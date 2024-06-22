
import yaml
import torch.nn as nn
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor, LlavaNextImageProcessor
import torch
import torch.nn.functional as F
from PIL import Image

class llava(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, cache_dir = './')
        self.template = "[INST] <image>\n{} [/INST]"
        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", cache_dir = './')
    
    def forward(self,data,device):
        response = {k: v.to(device) for k, v in data[0].items()}
        outputs = self.model(**response)
        return outputs
        
        
    def inference(self, image_path, prompt, dataset=None):
        image = Image.open(image_path)
        prompt = self.template.format(prompt)
        inputs = self.processor(prompt, image, return_tensors="pt").to("cuda")
        output = self.model.generate(**inputs, max_new_tokens=1024)
        answer = self.processor.decode(output[0], skip_special_tokens=True)
        answer = answer.split('[/INST]')[1].strip()
        return answer
    