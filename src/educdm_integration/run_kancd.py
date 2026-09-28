import argparse
import wandb
import logging
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from EduCDM import KaNCD

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="synthetic-11")
parser.add_argument("--max_n_trajectories", type=int, default=None)
parser.add_argument("--min_trajectory_length", type=int, default=None)
parser.add_argument("--max_trajectory_length", type=int, default=None)
parser.add_argument("--epochs", type=int, default=20)
args = parser.parse_args()

wandb.init(project="educdm", name=f"{args.dataset_name}-kancd")
wandb.config.update({
    "model": "KaNCD",
    "dataset_name": args.dataset_name,
    "max_n_trajectories": args.max_n_trajectories,
    "min_trajectory_length": args.min_trajectory_length,
    "max_trajectory_length": args.max_trajectory_length,
    "epochs": args.epochs,
})

logging.getLogger().setLevel(logging.INFO)

train = pd.read_csv(f"data/{args.dataset_name}/ncdm/train.csv")
valid = pd.read_csv(f"data/{args.dataset_name}/ncdm/valid.csv")
test = pd.read_csv(f"data/{args.dataset_name}/ncdm/test.csv")
df_item = pd.read_csv(f"data/{args.dataset_name}/ncdm/item.csv")

item2knowledge = {}
knowledge_set = set()
for _, s in df_item.iterrows():
    item_id, knowledge_codes = s['item_id'], list(set(eval(s['knowledge_code'])))
    item2knowledge[item_id] = knowledge_codes
    knowledge_set.update(knowledge_codes)
knowledge_n = max(knowledge_set)

def transform(user, item, item2knowledge, score, batch_size=32):
    knowledge_emb = torch.zeros((len(item), knowledge_n))
    for idx in range(len(item)):
        knowledge_emb[idx][torch.tensor(item2knowledge[item.iloc[idx]]) - 1] = 1.0
    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64) - 1,
        torch.tensor(item, dtype=torch.int64) - 1,
        knowledge_emb,
        torch.tensor(score, dtype=torch.float32)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)

train_loader = transform(train["user_id"], train["item_id"], item2knowledge, train["score"])
valid_loader = transform(valid["user_id"], valid["item_id"], item2knowledge, valid["score"])
test_loader = transform(test["user_id"], test["item_id"], item2knowledge, test["score"])

user_n = train["user_id"].max()
item_n = max(train["item_id"].max(), valid["item_id"].max(), test["item_id"].max())

cdm = KaNCD(exer_n=item_n, student_n=user_n, knowledge_n=knowledge_n, mf_type='gmf', dim=20)
cdm.train(train_loader, valid_loader, epoch_n=args.epochs, device="cuda")
cdm.save("kancd.snapshot")

cdm.load("kancd.snapshot")
auc, accuracy = cdm.eval(test_loader)
wandb.log({"auc": auc, "accuracy": accuracy})
print(f"auc: {auc:.6f}, accuracy: {accuracy:.6f}")
