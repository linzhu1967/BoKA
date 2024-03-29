
import os
import json
import torch
import numpy as np

from torch.utils.data import Dataset
from tqdm.auto import tqdm

# PubmedQA----------------------------------------------------------------------------------------------------------------------------------------------------------PubmedQA

class PairsDataset_pubmedqa(Dataset):
    def __init__(self, data_path, tokenizer, label2id_dict, max_length=1024):
        self.data_path_list = data_path if isinstance(data_path, list) else [data_path]
        self.tokenizer = tokenizer
        self.max_length = max_length

        print("Preloading dataset: ", self.data_path_list)
        self.data_dict = {}  # => dict that maps the id to the line offset (position of pointer in the file)
        for data_file in self.data_path_list:
            current_idx = len(self.data_dict)
            if os.path.isfile(data_file):
                with open(data_file) as reader:
                    for i, line in enumerate(tqdm(reader)):
                        if len(line) > 1:
                            line_dict = json.loads(line)
                            qid = line_dict["id"].strip()
                            query = line_dict["sentence1"].strip()
                            context = line_dict["sentence2"].strip()
                            label = line_dict["label"].strip()
                            if label in label2id_dict.keys():
                                label = label2id_dict[label]
                            else:
                                raise ValueError("For query {}, label {} not in label2id_dict".format(qid, label))
                            
                            self.data_dict[current_idx+i] = (qid, query, context, label)
            else:
                raise ValueError("provide valid data type for PairsDataset_pubmedqa")

        self.nb_ex = len(self.data_dict)


    def __len__(self):
        return self.nb_ex

    def __getitem__(self, idx):
        data = self.data_dict[idx]
        encoding = self.tokenizer(data[1], data[2], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'id': data[0],
            'input_ids': encoding['input_ids'].squeeze(), 
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(data[3], dtype=torch.long)
        }


class Dataset_pubmedqa_toAmalgamate_unlabeled(Dataset):
    # TODO
    def __init__(self, data_path, knowledge_path, tokenizer, label2id_dict, max_length=1024, valid_ids=[]):
        self.data_path_list = data_path if isinstance(data_path, list) else [data_path]
        self.knowledge_path_list = knowledge_path if isinstance(knowledge_path, list) else [knowledge_path]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.valid_ids = valid_ids

        knowledge_dict = self.preload_knowledge(self.knowledge_path_list)

        print("Preloading dataset: ", self.data_path_list)
        self.data_dict = {}  # => dict that maps the id to the line offset (position of pointer in the file)
        invalid_ids = []
        datadict_ids = []
        all_ids = []
        for data_file in self.data_path_list:
            current_idx = len(self.data_dict)
            if os.path.isfile(data_file):
                pos = 0
                with open(data_file) as reader:
                    for i, line in enumerate(tqdm(reader)): 
                        line_dict = json.loads(line)
                        qid = line_dict["id"].strip()
                        all_ids.append(qid)
                        if len(self.valid_ids) > 0:
                            if qid not in self.valid_ids:
                                invalid_ids.append(qid)
                                continue

                        datadict_ids.append(qid)
                        query = line_dict["sentence1"].strip()
                        context = line_dict["sentence2"].strip()
                        klg_list = knowledge_dict[qid]
                        if len(klg_list) == 1:
                            t_logits, t_label = klg_list[0]
                        else:
                            t_logits, t_last_hidden_state, t_label  = self.average_knowledge(klg_list, len(label2id_dict.keys()), self.max_length)
                        
                        # self.data_dict[current_idx+i] = (qid, query, context, t_label, t_logits, t_last_hidden_state)
                        self.data_dict[current_idx+pos] = (qid, query, context, t_label, t_logits)
                        pos+=1

            else:
                print("============================================================")
                print(data_file)
                raise ValueError("provide valid data type for PairsDataset_pubmedqa")


        self.nb_ex = len(self.data_dict)

    def preload_knowledge(self, knowledge_paths):
        knowledge_dict = {}
        for knowledge_path in knowledge_paths:
            with open(knowledge_path) as reader:
                for _, line in enumerate(tqdm(reader)):
                    line_dict = json.loads(line)
                    qid = line_dict["id"].strip()
                    logits = line_dict["logits"]
                    label = line_dict["label"]
                    # last_hidden_states = line_dict["last_hidden_states"]
                    if qid not in knowledge_dict.keys():
                        knowledge_dict[qid] = []
                    knowledge_dict[qid].append((logits, label))
                    # knowledge_dict[qid].append((logits, label, last_hidden_states))

    
        return knowledge_dict

    def average_knowledge(self, knowledge_list, label_num=3, hidden_size=1024):
        avg_logits = [0.0] * label_num
        avg_last_hidden_state = [0.0] * hidden_size

        for klg in knowledge_list:
            avg_logits.extend(klg[0])
            avg_last_hidden_state.extend(klg[2])
        avg_logits = np.average(np.array(avg_logits).reshape(-1,label_num), axis=0)
        avg_last_hidden_state = np.average(np.array(avg_last_hidden_state).reshape(-1,hidden_size), axis=0)
        avg_label = np.argmax(avg_logits)

        return avg_logits, avg_last_hidden_state, avg_label

    def __len__(self):
        return self.nb_ex

    def __getitem__(self, idx):
        data = self.data_dict[idx]
        encoding = self.tokenizer(data[1], data[2], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'id': data[0],
            'input_ids': encoding['input_ids'].squeeze(), 
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(data[3], dtype=torch.long),
            'teachers_logits': torch.tensor(data[4], dtype=torch.float)
            # 'teachers_last_hidden_state': torch.tensor(data[5], dtype=torch.float)
        }


class Dataset_pubmedqa_toAmalgamate_unlabeled_disagreement(Dataset):
    # TODO
    def __init__(self, data_path, knowledge_path, tokenizer, label2id_dict, max_length=1024):
        self.data_path_list = data_path if isinstance(data_path, list) else [data_path]
        self.knowledge_path_list = knowledge_path if isinstance(knowledge_path, list) else [knowledge_path]
        self.tokenizer = tokenizer
        self.max_length = max_length
        knowledge_dict = self.preload_knowledge(self.knowledge_path_list)
        print("Preloading dataset: ", self.data_path_list)
        orig_data_dict = self.preload_orig_data(self.data_path_list)

        self.data_dict = {}  # => dict that maps the id to the line offset (position of pointer in the file)
        for i, qid in enumerate(knowledge_dict.keys()):
            query, context = orig_data_dict[qid]
            t_logits, t_label, t_kldiv = knowledge_dict[qid][0]
            self.data_dict[i] = (qid, query, context, t_label, t_logits, t_kldiv)
        self.nb_ex = len(self.data_dict)

    def preload_knowledge(self, knowledge_paths):
        knowledge_dict = {}
        for knowledge_path in knowledge_paths:
            with open(knowledge_path) as reader:
                for _, line in enumerate(tqdm(reader)):
                    line_dict = json.loads(line)
                    qid = line_dict["id"].strip()
                    logits = line_dict["logits"]
                    label = line_dict["label"]
                    kldiv = line_dict["kldiv"]
                    if qid not in knowledge_dict.keys():
                        knowledge_dict[qid] = []
                    knowledge_dict[qid].append((logits, label, kldiv))
        return knowledge_dict

    def preload_orig_data(self, data_paths):
        orig_data_dict = {}
        for data_path in data_paths:
            if os.path.isfile(data_path):
                with open(data_path) as reader:
                    for i, line in enumerate(tqdm(reader)): 
                        line_dict = json.loads(line)
                        qid = line_dict["id"].strip()
                        query = line_dict["sentence1"].strip()
                        context = line_dict["sentence2"].strip()
                        orig_data_dict[qid] = (query, context)
            else:
                print("============================================================")
                print(data_path)
                raise ValueError("provide valid data type for pubmedqa")

        return orig_data_dict



    def __len__(self):
        return self.nb_ex

    def __getitem__(self, idx):
        data = self.data_dict[idx]
        encoding = self.tokenizer(data[1], data[2], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'id': data[0],
            'input_ids': encoding['input_ids'].squeeze(), 
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(data[3], dtype=torch.long),
            'teachers_logits': torch.tensor(data[4], dtype=torch.float),
            'teachers_kldiv': torch.tensor(data[5], dtype=torch.float)
        }




# MedMCQA----------------------------------------------------------------------------------------------------------------------------------------------------------MedMCQA
class PairsDataset_medmcqa(Dataset):
    def __init__(self, data_path, tokenizer, max_length=1024):
        self.data_path_list = data_path if isinstance(data_path, list) else [data_path]
        self.tokenizer = tokenizer
        self.max_length = max_length

        print("Preloading dataset: ", self.data_path_list)
        self.data_dict = {}  # => dict that maps the id to the line offset (position of pointer in the file)
        for data_file in self.data_path_list:
            current_idx = len(self.data_dict)
            with open(data_file) as reader:
                for i, line in enumerate(tqdm(reader)):
                    line_dict = json.loads(line)
                    qid = line_dict["id"].strip()
                    exp = line_dict["exp"].strip() if line_dict["exp"]!=None else ""
                    query = exp+"[Question] "+line_dict["question"].strip()
                    options = "[Option A] {opa} [Option B] {opb} [Option C] {opc} [Option D] {opd} [The Correct Answer Option]:".format(
                        opa=line_dict["opa"].strip(), opb=line_dict["opb"].strip(), opc=line_dict["opc"].strip(), opd=line_dict["opd"].strip())
                    label = line_dict["cop"]-1
                    self.data_dict[i+current_idx] = (qid, query, options, label)

        self.nb_ex = len(self.data_dict)

    def __len__(self):
        return self.nb_ex

    def __getitem__(self, idx):
        data = self.data_dict[idx]
        encoding = self.tokenizer(data[1], data[2], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'id': data[0],
            'input_ids': encoding['input_ids'].squeeze(), 
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(data[3], dtype=torch.long)
        }


class Dataset_medmcqa_toAmalgamate_unlabeled(Dataset):
    def __init__(self, data_path, knowledge_path, tokenizer, max_length=1024):
        self.data_path_list = data_path if isinstance(data_path, list) else [data_path]
        self.knowledge_path_list = knowledge_path if isinstance(knowledge_path, list) else [knowledge_path]
        self.tokenizer = tokenizer
        self.max_length = max_length

        knowledge_dict = self.preload_knowledge(self.knowledge_path_list)

        print("Preloading dataset: ", self.data_path_list)
        self.data_dict = {}  # => dict that maps the id to the line offset (position of pointer in the file)
        for data_file in self.data_path_list:
            current_idx = len(self.data_dict)
            if os.path.isfile(data_file):
                with open(data_file) as reader:
                    for i, line in enumerate(tqdm(reader)): 
                        line_dict = json.loads(line)
                        qid = line_dict["id"].strip()
                        exp = line_dict["exp"].strip() if line_dict["exp"]!=None else ""
                        query = exp+"[Question] "+line_dict["question"].strip()
                        options = "[Option A] {opa} [Option B] {opb} [Option C] {opc} [Option D] {opd} [The Correct Answer Option]:".format(
                            opa=line_dict["opa"].strip(), opb=line_dict["opb"].strip(), opc=line_dict["opc"].strip(), opd=line_dict["opd"].strip())
                        
                        t_logits, t_label = knowledge_dict[qid][0]
                        self.data_dict[current_idx+i] = (qid, query, options, t_label, t_logits)


            else:
                print("============================================================")
                print(data_file)
                raise ValueError("provide valid data type for PairsDataset_pubmedqa")


        self.nb_ex = len(self.data_dict)

    def preload_knowledge(self, knowledge_paths):
        knowledge_dict = {}
        for knowledge_path in knowledge_paths:
            with open(knowledge_path) as reader:
                for _, line in enumerate(tqdm(reader)):
                    line_dict = json.loads(line)
                    qid = line_dict["id"].strip()
                    logits = line_dict["logits"]
                    label = line_dict["label"]
                    if qid not in knowledge_dict.keys():
                        knowledge_dict[qid] = []
                    knowledge_dict[qid].append((logits, label))

    
        return knowledge_dict

    def __len__(self):
        return self.nb_ex

    def __getitem__(self, idx):
        data = self.data_dict[idx]
        encoding = self.tokenizer(data[1], data[2], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'id': data[0],
            'input_ids': encoding['input_ids'].squeeze(), 
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(data[3], dtype=torch.long),
            'teachers_logits': torch.tensor(data[4], dtype=torch.float)
        }


class Dataset_medmcqa_toAmalgamate_unlabeled_disagreement(Dataset):
    # TODO
    def __init__(self, data_path, knowledge_path, tokenizer, max_length=1024):
        self.data_path_list = data_path if isinstance(data_path, list) else [data_path]
        self.knowledge_path_list = knowledge_path if isinstance(knowledge_path, list) else [knowledge_path]
        self.tokenizer = tokenizer
        self.max_length = max_length
        knowledge_dict = self.preload_knowledge(self.knowledge_path_list)
        print("Preloading dataset: ", self.data_path_list)
        orig_data_dict = self.preload_orig_data(self.data_path_list)

        self.data_dict = {}  # => dict that maps the id to the line offset (position of pointer in the file)
        for i, qid in enumerate(knowledge_dict.keys()):
            query, options = orig_data_dict[qid]
            t_logits, t_label, t_kldiv = knowledge_dict[qid][0]
            self.data_dict[i] = (qid, query, options, t_label, t_logits, t_kldiv)
        self.nb_ex = len(self.data_dict)

    def preload_knowledge(self, knowledge_paths):
        knowledge_dict = {}
        for knowledge_path in knowledge_paths:
            with open(knowledge_path) as reader:
                for _, line in enumerate(tqdm(reader)):
                    line_dict = json.loads(line)
                    qid = line_dict["id"].strip()
                    logits = line_dict["logits"]
                    label = line_dict["label"]
                    kldiv = line_dict["kldiv"]
                    if qid not in knowledge_dict.keys():
                        knowledge_dict[qid] = []
                    knowledge_dict[qid].append((logits, label, kldiv))
        return knowledge_dict

    def preload_orig_data(self, data_paths):
        orig_data_dict = {}
        for data_path in data_paths:
            if os.path.isfile(data_path):
                with open(data_path) as reader:
                    for i, line in enumerate(tqdm(reader)): 
                        line_dict = json.loads(line)
                        qid = line_dict["id"].strip()
                        exp = line_dict["exp"].strip() if line_dict["exp"]!=None else ""
                        query = exp+"[Question] "+line_dict["question"].strip()
                        options = "[Option A] {opa} [Option B] {opb} [Option C] {opc} [Option D] {opd} [The Correct Answer Option]:".format(
                            opa=line_dict["opa"].strip(), opb=line_dict["opb"].strip(), opc=line_dict["opc"].strip(), opd=line_dict["opd"].strip())
                        orig_data_dict[qid] = (query, options)
            else:
                print("============================================================")
                print(data_path)
                raise ValueError("provide valid data type for pubmedqa")

        return orig_data_dict


    def __len__(self):
        return self.nb_ex

    def __getitem__(self, idx):
        data = self.data_dict[idx]
        encoding = self.tokenizer(data[1], data[2], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'id': data[0],
            'input_ids': encoding['input_ids'].squeeze(), 
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(data[3], dtype=torch.long),
            'teachers_logits': torch.tensor(data[4], dtype=torch.float),
            'teachers_kldiv': torch.tensor(data[5], dtype=torch.float)
        }




