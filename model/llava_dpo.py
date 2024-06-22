
import yaml
import torch.nn as nn
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor, LlavaNextImageProcessor
import torch
import torch.nn.functional as F
from PIL import Image

class llava_dpo(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, cache_dir = './')
        self.template = "[INST] <image>\n{} [/INST]"
        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", cache_dir = './')
    
    def forward(self,data,device):
        preferred = {k: v.to(device) for k, v in data[0].items()}
        preferred_outputs = self.model(**preferred)
        preferred_logits = preferred_outputs.logits # batch_size, seq_len, vocab_size
        preferred_labels = preferred["labels"]
        question_len = data[2] 
        
        probabilities = torch.log_softmax(preferred_logits, dim=-1) # batch_size, seq_len, vocab_size
        
       
        # Get the start index of the response (end of the question)
        response_start = question_len
        image_token_len = preferred_logits.shape[1] - preferred_labels.shape[1] 

        response_probs = probabilities[0, image_token_len + response_start-1:-1] # response_len, vocab_size
        response_labels = preferred_labels[0, response_start:] # response_len 
   
   

        
        # Compute the probability of the actual response
        # This multiplies the probabilities of each token in the response
        probs_ = response_probs[range(response_probs.shape[0]), response_labels]
        response_probability_prefer = torch.sum(probs_)
        ### reject
        rejected = {k: v.to(device) for k, v in data[1].items()}
        rejected_outputs = self.model(**rejected)
        rejected_logits = rejected_outputs.logits
        rejected_labels = rejected["labels"]
        
        probabilities = torch.log_softmax(rejected_logits, dim=-1) # batch_size, seq_len, vocab_size

        # Get the start index of the response (end of the question)
        response_start = question_len
        
        # Slice the probabilities to get only the response part
        # and the corresponding labels for the response
        response_probs = probabilities[0, image_token_len + response_start-1:-1]
        response_labels = rejected_labels[0, response_start:]
        
        # Compute the probability of the actual response
        # This multiplies the probabilities of each token in the response
        response_probability_reject = torch.sum(response_probs[range(response_probs.shape[0]), response_labels])
        return preferred_outputs, response_probability_prefer, response_probability_reject
        
        
        
    
    def inference(self, image_path, prompt, dataset=None):
        image = Image.open(image_path)
        prompt = self.template.format(prompt)
        inputs = self.processor(prompt, image, return_tensors="pt").to("cuda")
        output = self.model.generate(**inputs, max_new_tokens=1024)
        answer = self.processor.decode(output[0], skip_special_tokens=True)
        answer = answer.split('[/INST]')[1].strip()
        return answer
 