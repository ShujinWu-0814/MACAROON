import MACAROON.data
import MACAROON.model
import yaml
import torch.nn as nn
from transformers import LlavaNextForConditionalGeneration
import torch
import wandb
import argparse
from tqdm import tqdm
import os
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

def main():
    wandb.login()
    
    argparser = argparse.ArgumentParser()
    # argparser.add_argument("--resume", action="store_true")
    argparser.add_argument("--local-rank", type=int, 
                           help="Local rank. Necessary for using the torch.distributed.launch utility.")
    argparser.add_argument("--method")
    
    args = argparser.parse_args()
    local_rank = args.local_rank
    # resume = args.resume
    method = args.method
    
    wandb.init(
        # set the wandb project where this run will be logged
        project="MACAROON",
        name="Llava {}".format(method),
        # track hyperparameters and run metadata
        config={
        "learning_rate": 1e-4,
        "architecture": "llava",
        "epochs": 1,
        }
    )

    with open('../config/config_{}.yml'.format(method), "r") as f:
    # safeload yaml 
        config = yaml.safe_load(f)
    
    
    epochs = config["training"]["epochs"]
    accu_grad_steps = config["training"]["accumulate_grad_batches"]
    output_dir = config['output_dir']
    learning_rate = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    resume_from_checkpoint = config.get("resume_from_checkpoint", None)
    # checkpoint_every_n_steps = config["training"]["checkpoint_every_n_steps"]
    


    # import model and data
    data_module = getattr(
        MACAROON.data, config["data"]["data_module"]
    )(config)
    
    
    
    torch.distributed.init_process_group(backend="nccl")
    device = torch.device("cuda:{}".format(local_rank))
    
    train_dataloader = data_module.dataloader()
    # eval_dataloader = data_module.dataloader('eval')

    model = getattr(MACAROON.model, config["model"]["model_module"])(config)

    # for parameter in model.model.parameters():
    #     parameter.requires_grad = False
        
    # layers_to_unfreeze = model.model.language_model.model.layers[-10:]
    # for layer in layers_to_unfreeze:
    #     for parameter in layer.parameters():
    #         parameter.requires_grad = True
    
    #lora
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
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    
    if resume_from_checkpoint is not None:
        model.load_state_dict(torch.load(resume_from_checkpoint, map_location='cpu'))
        
    
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler =  get_cosine_schedule_with_warmup(optimizer, 100,
                                                   epochs * len(train_dataloader) // accu_grad_steps)
    
    
    def train_one_epoch():
        running_loss = 0.0
        
        for i, batch in enumerate(tqdm(train_dataloader)):
            try:
                outputs = model.forward(batch,device)
            except Exception as e:
                print(e)
                continue
            loss = outputs.loss
            loss = loss/accu_grad_steps
            loss.backward()
            
            wandb.log({"Train Loss": loss * accu_grad_steps})
            if (i+1) % accu_grad_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                
            running_loss += loss.item()
            
            if i % 100 == 99:
                avg_loss_per_batch = running_loss/100
                print(f"Average loss for {i+1} batches: {avg_loss_per_batch}")
                running_loss = 0.0
                
            if (i+1) % 3000 == 0:
                if os.path.exists(output_dir):
                    pass
                else:
                    os.makedirs(output_dir, exist_ok=True)
                torch.save(model.module.state_dict(), os.path.join(output_dir, f"model_{i+1}.pt"))
                
            if (i+1) % 20000 == 0:
                break
            
        return avg_loss_per_batch


    for epoch in range(epochs):
        print("Local Rank: {}, Epoch: {}, Training ...".format(local_rank, epoch+1))
        model.train(True)
        avg_loss = train_one_epoch()
        
    torch.save(model.module.state_dict(), os.path.join(output_dir, f"model_final.pt"))
    
if __name__ == '__main__':
    main()
        
            

