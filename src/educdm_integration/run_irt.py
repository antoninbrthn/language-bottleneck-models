import argparse
import wandb
import logging
import pandas as pd
from EduCDM import GDIRT, EMIRT  # or EMIRT


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="synthetic-11")
parser.add_argument("--max_n_trajectories", type=int, default=None)
parser.add_argument("--min_trajectory_length", type=int, default=None)
parser.add_argument("--max_trajectory_length", type=int, default=None)
parser.add_argument("--epochs", type=int, default=20)
args = parser.parse_args()

wandb.init(project="educdm", name=f"{args.dataset_name}-irt")
wandb.config.update({
    "model": "GDIRT",
    "dataset_name": args.dataset_name,
    "max_n_trajectories": args.max_n_trajectories,
    "min_trajectory_length": args.min_trajectory_length,
    "max_trajectory_length": args.max_trajectory_length,
    "epochs": args.epochs,
})

logging.getLogger().setLevel(logging.INFO)

# Load data
train = pd.read_csv(f"data/{args.dataset_name}/irt/train.csv")
valid = pd.read_csv(f"data/{args.dataset_name}/irt/valid.csv")
test = pd.read_csv(f"data/{args.dataset_name}/irt/test.csv")

# Prepare dataloaders (see official notebook for batchify)
import torch
from torch.utils.data import TensorDataset, DataLoader

def transform(df, batch_size=256):
    dataset = TensorDataset(
        torch.tensor(df["user_id"], dtype=torch.int64),
        torch.tensor(df["item_id"], dtype=torch.int64),
        torch.tensor(df["score"], dtype=torch.float)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_loader = transform(train)
valid_loader = transform(valid)
test_loader = transform(test)

n_items = max(train["item_id"].max(), valid["item_id"].max(), test["item_id"].max())
cdm = GDIRT(train["user_id"].max()+1, n_items+1)
cdm.train(train_loader, valid_loader, epoch=args.epochs)
cdm.save("irt.params")

cdm.load("irt.params")
auc, accuracy = cdm.eval(test_loader)
wandb.log({"auc": auc, "accuracy": accuracy})
print(f"auc: {auc:.6f}, accuracy: {accuracy:.6f}")
