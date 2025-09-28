import wandb
import torch

# 1. Login (first time)
wandb.login()  # or set WANDB_API_KEY in environment

# 2. Initialize project
wandb.init(project="my_decoder_model", name="experiment_01")

# 3. Example logging
loss = torch.tensor(0.5)
wandb.log({"train_loss": loss.item(), "epoch": 1})

# 4. Log model
# wandb.watch(model, log="all", log_freq=10)
