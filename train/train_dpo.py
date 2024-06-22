import MACAROON.data
import MACAROON.model
import yaml
import torch.nn as nn
from transformers import LlavaNextForConditionalGeneration
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import wandb
import argparse
import torch.nn.functional as F
import os

def main():
    wandb.login()

    with open('../config/config_dpo.yml', "r") as 
        # safeload yaml 
        config = yaml.safe_load(f)

    # import model and data
    data_module = getattr(
        MACAROON.data, config["data"]["data_module"]
    )(config)



    wandb.init(
        # set the wandb project where this run will be logged
        project="MACAROON",
        name="Llava Finetune",
        # track hyperparameters and run metadata
        config={
        "learning_rate": 1e-4,
        "architecture": "llava",
        "epochs": 3,
        }
    )

    
    epochs = config["training"]["epochs"]
    accu_grad_steps = config["training"]["accumulate_grad_batches"]
    output_dir = config['output_dir']
    learning_rate = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    # checkpoint_every_n_steps = config["training"]["checkpoint_every_n_steps"]
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--resume", action="store_true")
    argparser.add_argument("--local-rank", type=int, 
                           help="Local rank. Necessary for using the torch.distributed.launch utility.")
    
    
    args = argparser.parse_args()
    local_rank = args.local_rank
    resume = args.resume
    
    torch.distributed.init_process_group(backend="nccl")
    device = torch.device("cuda:{}".format(local_rank))

    train_dataloader = data_module.dataloader()
    # eval_dataloader = data_module.dataloader('eval')

    policy_model = getattr(MACAROON.model, config["model"]["model_module"])(config)
    reference_model = getattr(MACAROON.model, config["model"]["model_module"])(config)

    for parameter in policy_model.model.parameters():
        parameter.requires_grad = False
        
    layers_to_unfreeze = policy_model.model.language_model.model.layers[-10:]
    for layer in layers_to_unfreeze:
        for parameter in layer.parameters():
            parameter.requires_grad = True

    policy_model.to(device)
    policy_model = torch.nn.parallel.DistributedDataParallel(policy_model, device_ids=[local_rank], output_device=local_rank)
    reference_model.to(device)

    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, policy_model.parameters()), lr=learning_rate, weight_decay=weight_decay)

    
    def compute_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float
                    ):
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        
        logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}
        
        losses = -F.logsigmoid(beta * logits)
        # print(logits, losses)
        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards
    
    
    
    
    def train_one_epoch():
        running_loss = 0.0
        optimizer.zero_grad()
        for i, batch in enumerate(tqdm(train_dataloader)):
            # optimizer.zero_grad()
            policy_preferred_outputs, policy_preferred, policy_rejected = policy_model(batch,device)
            with torch.no_grad():
                reference_preferred_outputs, reference_preferred, reference_rejected = reference_model(batch,device)
            loss1 = compute_loss(policy_preferred, policy_rejected, reference_preferred, reference_rejected, 0.1)[0]
            loss2 = policy_preferred_outputs.loss
            loss = loss1 + loss2
            loss = loss/accu_grad_steps
            loss.backward()
            wandb.log({"Train Loss": loss * accu_grad_steps})
            if (i+1) % accu_grad_steps == 0:
                optimizer.step()
                # clip grad norm
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 0.5)
                optimizer.zero_grad()
            running_loss += loss.item()
            
            if i % 100 == 99:
                avg_loss_per_batch = running_loss/100
                print(f"Average loss for {i+1} batches: {avg_loss_per_batch}")
                running_loss = 0.0
            if (i+1) % 2000 == 0:
                if os.path.exists(output_dir):
                    pass
                else:
                    os.makedirs(output_dir, exist_ok=True)
                torch.save(policy_model.module.state_dict(), os.path.join(output_dir, f"model_{i+1}.pt"))

                
            if (i+1) % 10000 == 0:
                break
                
            
        return avg_loss_per_batch


    for epoch in range(epochs):
        print("Local Rank: {}, Epoch: {}, Training ...".format(local_rank, epoch+1))
        policy_model.train(True)
        avg_loss = train_one_epoch()
       
        
    # save final
    torch.save(policy_model.module.state_dict(), os.path.join(output_dir, f"model_final.pt"))


if __name__ == '__main__':
    main()