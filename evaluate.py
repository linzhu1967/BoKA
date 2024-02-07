import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

from transformers import BioGptForSequenceClassification
from transformers import AutoTokenizer

import sys 
sys.path.append("/home/ylz/ylz_github/23-auto_ka/") 
from utils.yaml import load_config_yaml
from utils.utils import set_seed_from_config
from prepare_data.datasets import PairsDataset_pubmedqa
from train_teachers.train_biogpt_pubmedqa import LABEL_DICT

from prepare_data.datasets import PairsDataset_medmcqa


def set_device():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device_count = torch.cuda.device_count()
    print("Start device: ", device)
    print(" --- use {} GPUs --- ".format(device_count))

    return device,device_count

def evaluate_biogpt_on_pubmedqa():
    # step-1: load the config file
    config_path = "./configs/evaluate/evaluate_pubmedqa.yaml"
    
    all_configs = load_config_yaml(config_path)
    config_dict = all_configs["config"]
    students_dict = all_configs["student_models"]
    teachers_dict = all_configs["teacher_models"]
    data_dict = all_configs["data"]

    # set the seed
    random_seed = set_seed_from_config(config_dict)
    print("random seed: {}".format(random_seed))

    # set device
    device, device_num = set_device()

    # set the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config_dict["tokenizer_type"])

    for test_file in data_dict["test_files"]:
        
        # set the dataset and dataloader
        test_dataset = PairsDataset_pubmedqa(data_path=test_file,
                                                tokenizer=tokenizer, 
                                                label2id_dict=LABEL_DICT, 
                                                max_length=config_dict["max_length"] )

        test_dataloader = DataLoader(test_dataset, batch_size=config_dict["eval_batch_size"], shuffle=False, num_workers=2)

        # test
        print("---------------------------------------------------------------")
        print("Test on the topic: [{}]".format(test_file.split("/")[-2]))
        test_ckpt_list = []
        if teachers_dict["ckpt_paths"] is not None:
            test_ckpt_list.extend(teachers_dict["ckpt_paths"])
        
        if students_dict["ckpt_paths"] is not None:
            test_ckpt_list.extend(students_dict["ckpt_paths"])

        # record the accuracy of different models
        acc_dict = {}

        for ckpt in test_ckpt_list:
            biogpt_model = None
            if "student" in ckpt:
                biogpt_model = BioGptForSequenceClassification.from_pretrained(students_dict["model_type_or_dir"], 
                                                                            num_labels=students_dict["num_labels"], 
                                                                            problem_type="multi_label_classification")
                biogpt_model.load_state_dict(torch.load(ckpt, map_location='cuda:0'))

            elif "teacher" in ckpt:
                t_state_dict = torch.load(ckpt)["state_dict"]
                if "biogpt.biogpt." in list(t_state_dict.keys())[0]:
                    biogpt_state_dict = {k[7:]:v for k,v in t_state_dict.items()}
                elif "model.biogpt." in list(t_state_dict.keys())[0]:
                    biogpt_state_dict = {k[6:]:v for k,v in t_state_dict.items()}
                else:
                    print("Unknown teacher model type!")
                    exit()

                biogpt_model = BioGptForSequenceClassification.from_pretrained(teachers_dict["model_type_or_dir"], 
                                                                         num_labels=teachers_dict["num_labels"], 
                                                                         problem_type="multi_label_classification")
                biogpt_model.load_state_dict(biogpt_state_dict)

            if device_num == 1:
                biogpt_model.to(device)
            else:
                print("The device_num is {}, but the model is not distributed !".format(device_num))
            
            biogpt_model.eval()
            test_accuracy = 0.0
            test_total = 0
            test_preds = []
            test_labels = []

            with torch.no_grad():
                for bt_i, batch in enumerate(tqdm(test_dataloader)):
                    batch = {k: v.to(device) for k, v in batch.items() if k!="id"}
                    input_dict = {
                        "input_ids": batch["input_ids"],
                        "attention_mask": batch["attention_mask"],
                    }

                    student_logits = biogpt_model(**input_dict).logits
                    predicted_labels = torch.max(student_logits, 1)[1]
                    labels = batch["labels"]

                    test_preds.append(predicted_labels.cpu().numpy())
                    test_labels.append(labels.cpu().numpy())
                    test_total += labels.size(0)
                    test_accuracy += (predicted_labels == labels).sum().item()

            if "student" in ckpt:
                print("Student model {}".format(ckpt))
            else:
                print("Teacher model {}".format(ckpt))

            avg_test_accuracy = test_accuracy / test_total
            print("Accuracy: {}".format(avg_test_accuracy))

            test_preds = np.concatenate(test_preds, axis=0)
            test_labels = np.concatenate(test_labels, axis=0)
            f1_macro = f1_score(test_labels, test_preds, average="macro")
            print(f"F1-macro: {f1_macro:.4f}")

            acc_dict[ckpt.split("/")[-2]] = avg_test_accuracy

        print("---------------------------------------------------------------")
        print("Test on the topic: [{}]".format(test_file.split("/")[-2]))
        print("The accuracy of different models: ")
        for k,v in acc_dict.items():
            print(k, v)





def evaluate_biogpt_on_medmcqa():
    # step-1: load the config file
    config_path = "./configs/evaluate/evaluate_medmcqa.yaml"

    
    all_configs = load_config_yaml(config_path)
    config_dict = all_configs["config"]
    students_dict = all_configs["student_models"]
    teachers_dict = all_configs["teacher_models"]
    data_dict = all_configs["data"]

    # set the seed
    random_seed = set_seed_from_config(config_dict)
    print("random seed: {}".format(random_seed))

    # set device
    device, device_num = set_device()

    # set the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config_dict["tokenizer_type"])

    for test_file in data_dict["test_files"]:
        
        # set the dataset and dataloader
        test_dataset = PairsDataset_medmcqa(data_path=test_file,
                                                tokenizer=tokenizer, 
                                                max_length=config_dict["max_length"] )

        test_dataloader = DataLoader(test_dataset, batch_size=config_dict["eval_batch_size"], shuffle=False, num_workers=2)

        # test
        print("---------------------------------------------------------------")
        print("Test on the topic: [{}]".format(test_file.split("/")[-2]))
        test_ckpt_list = []
        if teachers_dict["ckpt_paths"] is not None:
            test_ckpt_list.extend(teachers_dict["ckpt_paths"])
        
        if students_dict["ckpt_paths"] is not None:
            test_ckpt_list.extend(students_dict["ckpt_paths"])


        # record the accuracy of different models
        acc_dict = {}
        for ckpt in test_ckpt_list:
            biogpt_model = None
            if "student" in ckpt:
                num_teacher = len(ckpt.split("/")[-2].split("_")[-1])
                biogpt_model = BioGptForSequenceClassification.from_pretrained(students_dict["model_type_or_dir"], 
                                                                            num_labels=students_dict["num_labels"]*num_teacher, 
                                                                            problem_type="multi_label_classification")
                biogpt_model.load_state_dict(torch.load(ckpt, map_location='cuda:0'))

            elif "teacher" in ckpt:
                t_state_dict = torch.load(ckpt)["state_dict"]
                if "biogpt.biogpt." in list(t_state_dict.keys())[0]:
                    biogpt_state_dict = {k[7:]:v for k,v in t_state_dict.items()}
                elif "model.biogpt." in list(t_state_dict.keys())[0]:
                    biogpt_state_dict = {k[6:]:v for k,v in t_state_dict.items()}
                else:
                    print("Unknown teacher model type!")
                    exit()

                biogpt_model = BioGptForSequenceClassification.from_pretrained(teachers_dict["model_type_or_dir"], 
                                                                         num_labels=teachers_dict["num_labels"], 
                                                                         problem_type="multi_label_classification")
                biogpt_model.load_state_dict(biogpt_state_dict)

            if device_num == 1:
                biogpt_model.to(device)
            else:
                print("The device_num is {}, but the model is not distributed !".format(device_num))
            
            biogpt_model.eval()
            test_accuracy = 0.0
            test_total = 0
            test_preds = []
            test_labels = []

            with torch.no_grad():
                for bt_i, batch in enumerate(tqdm(test_dataloader)):
                    len_batch = len(batch["id"])
                    batch = {k: v.to(device) for k, v in batch.items() if k!="id"}
                    input_dict = {
                        "input_ids": batch["input_ids"],
                        "attention_mask": batch["attention_mask"],
                    }

                    output_logits = biogpt_model(**input_dict).logits
                    # 学生模型的输出做平均
                    if output_logits.shape[-1] > students_dict["num_labels"]:
                        output_logits = torch.mean(output_logits.view(len_batch, -1, students_dict["num_labels"]), dim=1)
                    predicted_labels = torch.max(output_logits, 1)[1]
                    labels = batch["labels"]

                    test_preds.append(predicted_labels.cpu().numpy())
                    test_labels.append(labels.cpu().numpy())
                    test_total += labels.size(0)
                    test_accuracy += (predicted_labels == labels).sum().item()

            if "student" in ckpt:
                print("Student model {}".format(ckpt))
            else:
                print("Teacher model {}".format(ckpt))

            avg_test_accuracy = test_accuracy / test_total
            print("Accuracy: {}".format(avg_test_accuracy))

            test_preds = np.concatenate(test_preds, axis=0)
            test_labels = np.concatenate(test_labels, axis=0)
            f1_macro = f1_score(test_labels, test_preds, average="macro")
            print(f"F1-macro: {f1_macro:.4f}")

            acc_dict[ckpt.split("/")[-2]] = round(avg_test_accuracy,3)

        print("---------------------------------------------------------------")
        print("Test on the topic: [{}]".format(test_file.split("/")[-2]))
        print("The accuracy of different models: ")
        for k,v in acc_dict.items():
            print(v)

            

    


if __name__ == "__main__":

   # test on Pubmedqa
   evaluate_biogpt_on_pubmedqa()

   # test on Medmcqa
   evaluate_biogpt_on_medmcqa()

