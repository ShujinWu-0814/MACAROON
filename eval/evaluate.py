import yaml
import argparse
from tqdm import tqdm
import torch
import jsonlines
from PIL import Image
import os
import MACAROON.data
import MACAROON.model
from peft import LoraConfig, get_peft_model

def save_results_tofile(output_path, save_results):
    with jsonlines.open(output_path, 'w') as writer:
        for result in save_results:
            writer.write(result)
    print(f"Results saved in {output_path}")

def read_jsonl(output_path):
    with jsonlines.open(output_path, 'r') as reader:
        save_results = [obj for obj in reader]
    return save_results

def main():
    # load config from args
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str)
    args = parser.parse_args()
    
    
    with open('../config/config_{}.yml'.format(args.method), "r") as f:
        # safeload yaml 
        config = yaml.safe_load(f)
            
        
    # import model and data
    model = getattr(MACAROON.model, config["model"]["model_module"])(config)
    incompatible_state_dict = torch.load('../model_output/output_model_{}/model_10000.pt'.format(args.method), map_location='cpu')

    lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", 
                    "k_proj",
                    "v_proj"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
    )
    
    if 'dpo' not in args.method:
        model = get_peft_model(model, lora_config)
        

    # state_dict = {}
    # for key in incompatible_state_dict.keys():
    #     new_key = key.split('.model.model')[0] + '.model' + key.split('.model.model')[1]
    #     state_dict[new_key] = incompatible_state_dict[key]
        
    model.load_state_dict(incompatible_state_dict)
    model.to('cuda')
    

    eval_data_module = getattr(
        MACAROON.data, config["data"]["eval_module"]
    )(config)
    
    
    if os.path.exists(config["eval_output_path"]):
        save_results = read_jsonl(config["eval_output_path"])
    else:
        save_results = []
    
    
    output_path = config["eval_output_path"]
    
    # evaluation loop
    test_dataloader = eval_data_module.test_dataloader()    
    for batch in tqdm(test_dataloader):
        img_path = batch['image_path']
        question = batch['question']
        result = {}
        with torch.no_grad():
            if 'CRL' in args.method:
                result['answer'] = model.inference(img_path, 'good '+ question)
            else:
                result['answer'] = model.inference(img_path, question)
        result['img_path'] = img_path
        result['question'] = question
        save_results.append(result)
    save_results_tofile(output_path, save_results)


if __name__ == '__main__':
    main()