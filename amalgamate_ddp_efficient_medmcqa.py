
import os
import sys
import argparse
import json
import torch
from tqdm import tqdm
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from transformers import AutoTokenizer
from transformers import BioGptForSequenceClassification
from transformers.optimization import get_linear_schedule_with_warmup

from sklearn.metrics import f1_score


from utils.yaml import load_config_yaml
from utils.utils import get_loss, set_seed_from_config, set_seed


from prepare_data.datasets import Dataset_medmcqa_toAmalgamate_unlabeled
from prepare_data.datasets import PairsDataset_medmcqa
from prepare_data.datasets import Dataset_medmcqa_toAmalgamate_unlabeled_disagreement




def new_single_ka_medmcqa(combination_name, knowledge_path_list, device, random_seed):
    # step-1: load the config file

    local_rank = int(os.getenv('LOCAL_RANK', -1))
    
    config_path = "configs/amalgamation/bo_ka_medmcqa.yaml"

    all_configs = load_config_yaml(config_path)
    config_dict = all_configs["config"]
    data_dict = all_configs["data"]
    student_dict = all_configs["student_model"]

    # set the seed
    set_seed(random_seed)
    print("random seed: {}".format(random_seed))


    # step-2: set the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config_dict["tokenizer_type"])

    # knowledge_path_list = [k_path for sublist in teachers_dict["teacher_knowledge_paths"].values() for k_path in sublist]
    print("knowledge_path_list: ", knowledge_path_list)

    # step-4
    print("Load data......")
    train_dataset = None
    train_dataset = Dataset_medmcqa_toAmalgamate_unlabeled(data_dict["train_files"], 
                                                    knowledge_path_list,
                                                    tokenizer,
                                                    config_dict["max_length"]
                                                    )
    val_dataset = PairsDataset_medmcqa(data_path=data_dict["test_files"],
                                            tokenizer=tokenizer, 
                                            max_length=config_dict["max_length"] )
    
    # distributed sampler
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    
    # init dataloaders
    # train_dataloader = DataLoader(train_dataset, batch_size=config_dict["train_batch_size"], shuffle=True, num_workers=2)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config_dict["train_batch_size"],
                                num_workers=2, pin_memory=True, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config_dict["eval_batch_size"], shuffle=False, num_workers=2)
    
    del train_dataset
    del val_dataset
    del train_sampler
    del tokenizer
    dist.barrier()

    # init model
    biogpt_student = BioGptForSequenceClassification.from_pretrained(student_dict["model_type_or_dir"], 
                                                                     num_labels=student_dict["num_labels"], 
                                                                     problem_type="multi_label_classification",
                                                                     output_hidden_states=True)
    
    if student_dict["checkpoint_restore"] is not None:  
        biogpt_student.load_state_dict(torch.load(student_dict["checkpoint_restore"]))

    biogpt_student.to(device) # model = model.cuda()
    biogpt_student = DDP(biogpt_student, device_ids=[local_rank], output_device=local_rank)

    # Optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(biogpt_student.parameters(), lr=config_dict["lr"], weight_decay=config_dict["weight_decay"])
    num_training_steps = len(train_dataloader)* config_dict["nb_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # step-5: 
    # train and eval
    print("----------------------------------------------------------------------------------")
    print("Start doing KA with the knowledge_type {}...".format(config_dict["knowledge_type"]))
    student_save_dir = student_dict["checkpoint_dir"]+combination_name
    if local_rank in [-1,0]:
        if not os.path.exists(student_save_dir):
            os.makedirs(student_save_dir)

    # record the best val_acc
    best_val_acc = 0.0

    for epoch in range(config_dict["nb_epochs"]):
        print("Local_rank:{} Epoch:{}".format(local_rank, epoch))
        
        # train
        total_loss = 0.0
        biogpt_student.train()
        for bt_i, batch in enumerate(tqdm(train_dataloader)):

            # batch_ids_int = torch.tensor([int(qid) for qid in batch["id"]]).to(device)
            # batch = {k: v.to(device) for k, v in batch.items() if k!="id"}
            input_dict = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            # outputs = biogpt_student(**input_dict)

            student_logits =  biogpt_student(**input_dict).logits
            
            teachers_logits = batch["teachers_logits"].to(device)

            # hard loss (teacher_labels:psuedo)
            hard_loss_function = torch.nn.CrossEntropyLoss(reduction="none")
            avg_hard_loss = torch.mean(hard_loss_function(student_logits, batch["labels"].to(device)), dim=0)

            # soft loss (correct): logits
            logits_loss_function = nn.KLDivLoss(reduction="none")
            logits_loss = torch.mean(torch.sum(logits_loss_function(torch.log_softmax(student_logits, dim=-1), torch.softmax(teachers_logits, dim=-1)), dim=-1))

            # loss                                     
            loss = avg_hard_loss+logits_loss

            optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            scheduler.step()
            dist.barrier()


        # print(q_loss_record)
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Local_rank {local_rank},Epoch {epoch}, Training loss: {avg_train_loss:.4f}")
        # print("The length of gpu_kept_qids {} on gpu {}".format(gpu_kept_qids.size(), local_rank))
        dist.barrier()

        # eval
        biogpt_student.eval()
        val_accuracy = 0.0
        val_total = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for bt_i, batch in enumerate(tqdm(val_dataloader)):
                batch = {k: v.to(device) for k, v in batch.items() if k!="id"}
                input_dict = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                }

                student_logits = biogpt_student(**input_dict).logits
                predicted_labels = torch.max(student_logits, 1)[1]
                labels = batch["labels"]

                val_preds.append(predicted_labels.cpu().numpy())
                val_labels.append(labels.cpu().numpy())
                val_total += labels.size(0)
                val_accuracy += (predicted_labels == labels).sum().item()

        avg_val_accuracy = val_accuracy / val_total
        print(f"Epoch {epoch}, Validation Accuracy: {avg_val_accuracy:.4f}")

        # Concatenate predictions and labels
        val_preds = np.concatenate(val_preds, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        # Calculate F1-macro score
        f1_macro = f1_score(val_labels, val_preds, average='macro')
        print(f"Epoch {epoch}, Validation F1-macro: {f1_macro:.4f}")

        # student_save_path = os.path.join(student_save_dir, "_".join(data_dict["topics"])+"_model.epoch"+str(epoch)+".pt")
        student_save_path = os.path.join(student_save_dir, "model.epoch{}.val_acc={:.4f}.pt".format(epoch, avg_val_accuracy))

        torch.save(biogpt_student.module.state_dict(), student_save_path)
        print("Save student: {}".format(student_save_path))

        # save the best model
        if avg_val_accuracy > best_val_acc:
            best_val_acc = avg_val_accuracy
            print("Current best val acuracy: ", best_val_acc)
        
        dist.barrier()
            
    return best_val_acc


def new_ka_medmcqa(combination_name, knowledge_path_list, device, random_seed):
    # step-1: load the config file
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    config_path = "configs/amalgamation/bo_ka_medmcqa.yaml"


    all_configs = load_config_yaml(config_path)
    config_dict = all_configs["config"]
    data_dict = all_configs["data"]
    student_dict = all_configs["student_model"]

    # set the seed
    set_seed(random_seed)
    print("random seed: {}".format(random_seed))

    # step-2: set the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config_dict["tokenizer_type"])

    # step-3:
    print("knowledge_path_list: ", knowledge_path_list)

    # step-4: 
    print("Load data......")
    train_dataset = Dataset_medmcqa_toAmalgamate_unlabeled_disagreement(data_dict["train_files"], 
                                                    knowledge_path_list,
                                                    tokenizer,
                                                    config_dict["max_length"]
                                                    )
    val_dataset = PairsDataset_medmcqa(data_path=data_dict["test_files"],
                                        tokenizer=tokenizer, 
                                        max_length=config_dict["max_length"] )

    
    # distributed sampler
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    # init dataloaders
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config_dict["train_batch_size"],
                                num_workers=4, pin_memory=True, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config_dict["eval_batch_size"], shuffle=False, num_workers=4)
    val_dataloader2 = DataLoader(val_dataset, sampler=val_sampler, batch_size=config_dict["eval_batch_size"],
                                num_workers=4, pin_memory=True, shuffle=False)
      
    del train_dataset
    del val_dataset
    del train_sampler
    del tokenizer


    # init model
    biogpt_student = BioGptForSequenceClassification.from_pretrained(student_dict["model_type_or_dir"], 
                                                                     num_labels=student_dict["num_labels"]*len(combination_name), 
                                                                     problem_type="multi_label_classification",
                                                                     output_hidden_states=True)
    
    if student_dict["checkpoint_restore"] is not None:  
        biogpt_student.load_state_dict(torch.load(student_dict["checkpoint_restore"]))

    # if local_rank == 0:
    biogpt_student.to(device) # model = model.cuda()
    biogpt_student = DDP(biogpt_student, device_ids=[local_rank], output_device=local_rank)
    # print(model.module.bert.config) 
    # Optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(biogpt_student.parameters(), lr=config_dict["lr"], weight_decay=config_dict["weight_decay"])
    num_training_steps = len(train_dataloader)* config_dict["nb_epochs"]
    # num_training_steps = int(data_dict["train_length"]/config_dict["train_batch_size"])* config_dict["nb_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # step-5:
    # train and eval
    print("----------------------------------------------------------------------------------")
    print("Start doing KA with the knowledge_type {}...".format(config_dict["knowledge_type"]))
    student_save_dir = student_dict["checkpoint_dir"]+combination_name
    if local_rank in [-1,0]:
        if not os.path.exists(student_save_dir):
            os.makedirs(student_save_dir)

    # record the best val_acc
    best_val_acc = 0.0

    for epoch in range(config_dict["nb_epochs"]):
        print("Local_rank:{} Epoch:{}".format(local_rank, epoch))
        
        # train
        total_loss = 0.0
        # gpu_kept_qids = torch.tensor([],dtype=int).to(device) 
        biogpt_student.train()
        for bt_i, batch in enumerate(tqdm(train_dataloader)):
            input_dict = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }

            student_logits =  biogpt_student(**input_dict).logits
            
            teachers_logits = batch["teachers_logits"].to(device)

            # soft loss: logits (correct) 
            logits_loss_function = nn.KLDivLoss(reduction="none")
            avg_logits_loss = torch.mean(torch.sum(logits_loss_function(torch.log_softmax(student_logits, dim=-1), torch.softmax(teachers_logits, dim=-1)), dim=-1))

            # hard loss: teacher_label
            hard_loss_function = torch.nn.CrossEntropyLoss(reduction="none")
            avg_hard_loss = torch.mean(hard_loss_function(torch.mean(student_logits.view(len(batch["id"]), -1,student_dict["num_labels"]), dim=1), batch["labels"].to(device)), dim=0)
            
            # loss                                         
            loss = avg_logits_loss+avg_hard_loss

            optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            scheduler.step()
            dist.barrier()


        # print(q_loss_record)
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Local_rank {local_rank},Epoch {epoch}, Training loss: {avg_train_loss:.4f}")
        # print("The length of gpu_kept_qids {} on gpu {}".format(gpu_kept_qids.size(), local_rank))
        dist.barrier()

       
        # -------------------------test with ddp----------------------
        print("Test again with ddp...")
        biogpt_student.eval()
        val_accuracy = 0
        val_total = 0
        with torch.no_grad():
            for bt_i, batch in enumerate(tqdm(val_dataloader2)):
                len_batch = len(batch["id"])
                batch = {k: v.to(device) for k, v in batch.items() if k!="id"}
                input_dict = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                }

                student_logits = torch.mean(biogpt_student(**input_dict).logits.view(len_batch, -1,student_dict["num_labels"]), dim=1)
                predicted_labels = torch.max(student_logits, 1)[1]
                labels = batch["labels"]

                val_total += labels.size(0)
                val_accuracy += (predicted_labels == labels).sum().item()


        correct_tensor = torch.tensor(val_accuracy).cuda()
        total_tensor = torch.tensor(val_total).cuda()

        torch.distributed.all_reduce(correct_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_tensor, op=torch.distributed.ReduceOp.SUM)

        correct_all = correct_tensor.item()
        total_all = total_tensor.item()
        
        avg_val_accuracy = correct_all / total_all

        if local_rank == 0:
            print(f"[ddp-test] Epoch {epoch}, Validation Accuracy: {avg_val_accuracy:.4f}")

        student_save_path = os.path.join(student_save_dir, "model.epoch{}.val_acc={:.4f}.pt".format(epoch, avg_val_accuracy))
        torch.save(biogpt_student.module.state_dict(), student_save_path)
        print("Save student: {}".format(student_save_path))

        # the best model
        if avg_val_accuracy > best_val_acc:
            best_val_acc = avg_val_accuracy
            print("Current best val acuracy: ", best_val_acc)
        
        dist.barrier()
            
    return best_val_acc




