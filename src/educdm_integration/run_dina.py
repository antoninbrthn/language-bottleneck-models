import argparse
import wandb
import logging
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from EduCDM import GDDINA

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="synthetic-11")
parser.add_argument("--max_n_trajectories", type=int, default=None)
parser.add_argument("--min_trajectory_length", type=int, default=None)
parser.add_argument("--max_trajectory_length", type=int, default=None)
parser.add_argument("--epochs", type=int, default=20)
args = parser.parse_args()

wandb.init(project="educdm", name=f"{args.dataset_name}-gddina")
wandb.config.update({
    "model": "GDDINA",
    "dataset_name": args.dataset_name,
    "max_n_trajectories": args.max_n_trajectories,
    "min_trajectory_length": args.min_trajectory_length,
    "max_trajectory_length": args.max_trajectory_length,
    "epochs": args.epochs,
})

logging.getLogger().setLevel(logging.INFO)

# Load data
train = pd.read_csv(f"data/{args.dataset_name}/ncdm/train.csv")
valid = pd.read_csv(f"data/{args.dataset_name}/ncdm/valid.csv")
test = pd.read_csv(f"data/{args.dataset_name}/ncdm/test.csv")
item_data = pd.read_csv(f"data/{args.dataset_name}/ncdm/item.csv")

# Determine the number of knowledge concepts
knowledge_num = max([max(eval(x)) for x in item_data["knowledge_code"]])  # assumes 1-based indexing

def code2vector(x):
    vector = [0] * knowledge_num
    for k in eval(x):
        vector[k - 1] = 1
    return vector

item_data["knowledge"] = item_data["knowledge_code"].apply(code2vector)
item_data.drop(columns=["knowledge_code"], inplace=True)

# Merge knowledge vector into each split
train = pd.merge(train, item_data, on="item_id")
valid = pd.merge(valid, item_data, on="item_id")
test = pd.merge(test, item_data, on="item_id")

def transform(x, y, z, k, batch_size=256):
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.int64),
        torch.tensor(y, dtype=torch.int64),
        torch.tensor(k, dtype=torch.float32),
        torch.tensor(z, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_loader = transform(train["user_id"], train["item_id"], train["score"], train["knowledge"])
valid_loader = transform(valid["user_id"], valid["item_id"], valid["score"], valid["knowledge"])
test_loader = transform(test["user_id"], test["item_id"], test["score"], test["knowledge"])

user_n = train["user_id"].max()
item_n = max(train["item_id"].max(), valid["item_id"].max(), test["item_id"].max())

cdm = GDDINA(user_n+1, item_n+1, knowledge_num)
cdm.train(train_loader, valid_loader, epoch=args.epochs)
cdm.save("gddina.params")

cdm.load("gddina.params")
auc, accuracy = cdm.eval(test_loader)
wandb.log({"auc": auc, "accuracy": accuracy})
print(f"auc: {auc:.6f}, accuracy: {accuracy:.6f}")
