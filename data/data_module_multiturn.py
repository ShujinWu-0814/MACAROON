
import numpy as np
from PIL import Image
from datasets import load_dataset
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from torch.utils.data import DataLoader, Dataset
import json
from torch.utils.data.distributed import DistributedSampler
import jsonlines

class DataModule_Multiturn():

    def __init__(self, config: dict):
        self.config = config
        # self.train_data_path = config["data"]["train_data_path"]
        self.preference_data_path = config["data"]["preference_data_path"]
        self.VLfeedback_data_path = config["data"]["VLfeedback_data_path"]
        
        VLfeedback_data = self.read_jsonl(self.VLfeedback_data_path)
        for item in VLfeedback_data:
            item['image'] = '/home/yangyic3/yangyic3/MultimodalAgent/data/' + item['image']
        self.train_data = self.read_json(self.preference_data_path) + VLfeedback_data
        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", cache_dir = './')

    def read_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    
    def read_jsonl(self, file_path):
        with open(file_path, 'r') as f:
            data = [json.loads(line) for line in f]
        return data
            
    
    def preprocess_data(self, examples):
        image_path = examples['image']
        begin_token = self.processor.tokenizer.bos_token
        end_token = self.processor.tokenizer.eos_token

        texts = []
        conversation_length = len(examples['conversations'])
        for i in range(conversation_length):
            if i % 2 == 0:
                if i == 0:
                    user = begin_token + "[INST] <image>\n{} [/INST]".format(examples['conversations'][i]['value'])
                else:
                    user = begin_token + "[INST] {} [/INST]".format(examples['conversations'][i]['value'])
                texts.append(user)
            else:
                assistant = "{} ".format(examples['conversations'][i]['value']) + end_token
                texts.append(assistant)
        image = Image.open(image_path)
        examples = {
            # 'id': examples['id'],
            'image': image,
            'text': texts
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
        conversation_length = len(preprocess_item['text'])
        for k, v in enumerate(preprocess_item['text']):
            if k != conversation_length - 1:
                tokenized_text = self.processor.tokenizer(v, max_length = 1024, truncation=True, add_special_tokens=False)
                input_ids += tokenized_text['input_ids']
                text_len = len(tokenized_text["input_ids"])
                labels += [-100] * text_len
                attention_mask += tokenized_text['attention_mask']
            else:
                tokenized_text = self.processor.tokenizer(v, max_length = 1024, truncation=True, add_special_tokens=False)
                input_ids += tokenized_text['input_ids']
                labels += tokenized_text['input_ids']
                attention_mask += tokenized_text['attention_mask']
                  
        encoding = self.processor.image_processor(preprocess_item['image'],return_tensors="pt")

        encoding['labels'] = torch.LongTensor([labels])
        encoding['input_ids'] = torch.LongTensor([input_ids])
        encoding['attention_mask'] = torch.LongTensor([attention_mask])
        return encoding, None
    
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
