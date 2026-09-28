import argparse
import wandb
import logging
import pandas as pd
from EduCDM import MIRT


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="synthetic-11")
parser.add_argument("--max_n_trajectories", type=int, default=None)
parser.add_argument("--min_trajectory_length", type=int, default=None)
parser.add_argument("--max_trajectory_length", type=int, default=None)
parser.add_argument("--epochs", type=int, default=20)
args = parser.parse_args()

wandb.init(project="educdm", name=f"{args.dataset_name}-mirt")
wandb.config.update({
    "model": "MIRT",
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
df_item = pd.read_csv(f"data/{args.dataset_name}/ncdm/item.csv")

item2knowledge = {}
knowledge_set = set()
for _, s in df_item.iterrows():
    item_id, knowledge_codes = s['item_id'], list(set(eval(s['knowledge_code'])))
    item2knowledge[item_id] = knowledge_codes
    knowledge_set.update(knowledge_codes)
knowledge_n = max(knowledge_set)

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
cdm = MIRT(train["user_id"].max()+1, n_items+1, knowledge_n)
cdm.train(train_loader, valid_loader, epoch=args.epochs)
cdm.save("mirt.params")

cdm.load("mirt.params")
auc, accuracy = cdm.eval(test_loader)
wandb.log({"auc": auc, "accuracy": accuracy})
print(f"auc: {auc:.6f}, accuracy: {accuracy:.6f}")
