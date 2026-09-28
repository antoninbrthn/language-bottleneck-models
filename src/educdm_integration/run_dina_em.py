import wandb
import logging
import numpy as np
import json
from EduCDM import EMDINA

wandb.init(project="educdm", name="synthetic-11-dina")
wandb.config.update({"model": "EMDINA"})

logging.getLogger().setLevel(logging.INFO)

# Load Q-matrix and data
q_m = np.loadtxt("data/synthetic-11/dina/q_m.csv", dtype=int, delimiter=',')
with open("data/synthetic-11/dina/train_data.json") as f:
    train_set = json.load(f)
with open("data/synthetic-11/dina/test_data.json") as f:
    test_set = json.load(f)

stu_num = max(x['user_id'] for x in train_set) + 1
prob_num = max(x['item_id'] for x in train_set) + 1
know_num = q_m.shape[1]

# Build response matrix
R = -1 * np.ones((stu_num, prob_num))
for log in train_set:
    R[log['user_id'], log['item_id']] = log['score']

cdm = EMDINA(R, q_m, stu_num, prob_num, know_num, skip_value=-1)
cdm.train(epoch=10, epsilon=1e-3)
cdm.save("dina.params")

cdm.load("dina.params")
rmse, mae = cdm.eval(test_set)
wandb.log({"rmse": rmse, "mae": mae})
print(f"RMSE: {rmse:.6f}, MAE: {mae:.6f}")
