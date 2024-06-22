
import numpy as np
from PIL import Image
from datasets import load_dataset
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from torch.utils.data import DataLoader, Dataset
import json
from torch.utils.data.distributed import DistributedSampler

from torch.utils.data import Dataset

# class MDataset(Dataset):
#     def __init__(self, data):
#         self.data = data # list
    
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]
    
    
class DataModule_dpo():

    def __init__(self, config: dict):
        self.config = config
        self.preference_data_path = config["data"]["preference_data_path"]
        self.VLfeedback_data_path = config["data"]["VLfeedback_data_path"]

        
        self.train_data = self.read_json(self.preference_data_path) + self.read_json(self.VLfeedback_data_path)
        
        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", cache_dir = './')

    def read_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    
    def preprocess_data(self, examples):
        end_token = self.processor.tokenizer.eos_token
        image_path = examples['image']
        question = "[INST] <image>\n{} [/INST]".format(examples['conversations'][0]['value'])
        preferred_response = "{} ".format(examples['conversations'][1]['value']['preferred']) + end_token
        rejected_response = "{} ".format(examples['conversations'][1]['value']['rejected']) + end_token
        image = Image.open(image_path)
        
        examples = {
            'id': examples['id'],
            'image': image,
            'text': {'preferred':[question, preferred_response],
                     'rejected':[question, rejected_response]}
        }
        return examples


    def collate_fn(self,batch):
        if len(batch) != 1:
            raise ValueError("Batch size must be 1")
        item = batch[0]
        preprocess_item = self.preprocess_data(item)
        preferred_input_ids = []
        rejected_input_ids = []
        
        preferred_attention_mask = []
        rejected_attention_mask = []
        
        preferred_labels = []
        rejected_labels = []
        
        #preferred
        for k, v in enumerate(preprocess_item['text']['preferred']):
            if k == 0: 
                new_v = self.processor.tokenizer.bos_token + v
                tokenized_question = self.processor.tokenizer(new_v, max_length = 1024, truncation=True, add_special_tokens=False)
                preferred_input_ids += tokenized_question['input_ids']
                preferred_attention_mask += tokenized_question['attention_mask']
                question_len = len(tokenized_question["input_ids"])
                preferred_labels += [-100] * question_len 
            else:
                tokenized_response = self.processor.tokenizer(v, max_length = 1024, truncation=True, add_special_tokens=False)
                preferred_input_ids += tokenized_response['input_ids']
                preferred_attention_mask += tokenized_response['attention_mask']
                preferred_labels += tokenized_response['input_ids']
                
        preferred_encoding = self.processor.image_processor(preprocess_item['image'], return_tensors="pt")
        preferred_encoding['input_ids'] = torch.LongTensor([preferred_input_ids])
        preferred_encoding['attention_mask'] = torch.LongTensor([preferred_attention_mask])
        preferred_encoding['labels'] = torch.LongTensor([preferred_labels])
        
        #rejected
        for k, v in enumerate(preprocess_item['text']['rejected']):
            if k == 0: 
                new_v = self.processor.tokenizer.bos_token + v
                tokenized_question = self.processor.tokenizer(new_v, max_length = 1024, truncation=True, add_special_tokens=False)
                rejected_input_ids += tokenized_question['input_ids']
                rejected_attention_mask += tokenized_question['attention_mask']
                rejected_labels += [-100] * len(tokenized_question["input_ids"])
            else: 
                tokenized_response = self.processor.tokenizer(v, max_length = 1024, truncation=True, add_special_tokens=False)
                rejected_input_ids += tokenized_response['input_ids']
                rejected_attention_mask += tokenized_response['attention_mask']
                rejected_labels += tokenized_response['input_ids']
                  
        rejected_encoding = self.processor.image_processor(preprocess_item['image'], return_tensors="pt")
        rejected_encoding['input_ids'] = torch.LongTensor([rejected_input_ids])
        rejected_encoding['attention_mask'] = torch.LongTensor([rejected_attention_mask])     
        rejected_encoding['labels'] = torch.LongTensor([rejected_labels])
        
        return preferred_encoding, rejected_encoding, question_len
    
    def dataloader(self):
            train_sampler = DistributedSampler(dataset=self.train_data)
            return torch.utils.data.DataLoader(
                self.train_data,
                batch_size=1,
                shuffle=False,
                num_workers=1,
                collate_fn=self.collate_fn,
                pin_memory=True,
                sampler=train_sampler
            )


