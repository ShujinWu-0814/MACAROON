
import numpy as np
from PIL import Image
from datasets import load_dataset
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from torch.utils.data import DataLoader, Dataset
import json
from torch.utils.data.distributed import DistributedSampler
import random


class DataModule_SFT():

    def __init__(self, config: dict):
        self.config = config

        self.preference_data_path = config["data"]["preference_data_path"]
        self.VLfeedback_data_path = config["data"]["VLfeedback_data_path"]
        preference_data = self.read_json(self.preference_data_path)
        general_data = self.read_json(self.VLfeedback_data_path)
        self.train_data = preference_data + general_data
        

        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", cache_dir = './')

    def read_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    
    def preprocess_data(self, examples):
        image_path = examples['image']

        end_token = self.processor.tokenizer.eos_token
        
        question = "[INST] <image>\n{} [/INST]".format(examples['conversations'][0]['value'])
        response = "{} ".format(examples['conversations'][1]['value']['preferred']) + end_token
        
        image = Image.open(image_path)
        examples = {
            'id': examples['id'],
            'image': image,
            'text': [question, response]
        }
        return examples


    def collate_fn(self,batch):
        if len(batch) != 1:
            raise ValueError("Batch size must be 1")
        item = batch[0]
        preprocess_item = self.preprocess_data(item)
        input_ids = []
        labels = []
        attention_mask = []
        for k, v in enumerate(preprocess_item['text']):
            if k == 0: 
                new_v = self.processor.tokenizer.bos_token + v
                tokenized_question = self.processor.tokenizer(new_v, max_length = 1024, truncation=True, add_special_tokens=False)
                input_ids += tokenized_question['input_ids']
                question_len = len(tokenized_question["input_ids"])
                labels += [-100] * question_len
                attention_mask += tokenized_question['attention_mask']
            else:
                tokenized_response = self.processor.tokenizer(v, max_length = 1024, truncation=True, add_special_tokens=False)
                input_ids += tokenized_response['input_ids']
                labels += tokenized_response['input_ids']
                attention_mask += tokenized_response['attention_mask']
                  
        encoding = self.processor.image_processor(preprocess_item['image'],return_tensors="pt")

        encoding['labels'] = torch.LongTensor([labels])
        encoding['input_ids'] = torch.LongTensor([input_ids])
        encoding['attention_mask'] = torch.LongTensor([attention_mask])
        return encoding, question_len
    
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
