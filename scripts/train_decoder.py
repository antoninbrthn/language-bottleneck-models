# Train decoder model using SFT

import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import wandb
import os
from peft import LoraConfig
from src.data.utils import get_data_from_config, resolve_config_path
from src.models.lb_models.utils import add_instructions_to_yprompt
from src.models.model_loader import load_any_model
from src.models.finetune_lora import DEFAULT_LORA_CONFIG
from src.evaluation.evaluate_model import ModelEvaluator
from src.utils.config import CONFIGS_DIR, DATA_DIR
from src.utils.misc import get_default_results_dir, get_run_name
import pickle

from datasets import Dataset
from trl import SFTConfig, SFTTrainer

# parse args for config_name
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config_name", type=str, default="config-train-decoder-xes")
args, unknown = parser.parse_known_args()
config_name = args.config_name

# Initialize Hydra and load the configuration
with initialize_config_dir(version_base=None, config_dir=CONFIGS_DIR):
    cfg = compose(config_name=config_name)
cfg_dict = OmegaConf.to_object(cfg)
print("Config:", cfg_dict)

# If config file has a data checkpoint, use that to load train_data and test_data
if "data_checkpoint_path" in cfg.dataset and cfg.dataset.data_checkpoint_path is not None:
    data_checkpoint_path = resolve_config_path(cfg.dataset.data_checkpoint_path)
    if not os.path.exists(data_checkpoint_path):
        raise ValueError(f"Data checkpoint path does not exist: {data_checkpoint_path}")
    print(f"Loading data from checkpoint: {data_checkpoint_path}")
    train_data_path = os.path.join(data_checkpoint_path, "train_data")
    test_data_path = os.path.join(data_checkpoint_path, "test_data")
    with open(train_data_path, "rb") as f:
        train_data = pickle.load(f)
    with open(test_data_path, "rb") as f:
        test_data = pickle.load(f)
    print(f"Train data size: {len(train_data)}, Test data size: {len(test_data)}")

    # Load encoder output for train_data
    encoder_output_path = cfg.dataset.get("encoder_output_path", None)
    if encoder_output_path is None:
        print("No encoder output path specified in config.dataset.encoder_output_path. Will run encoder on train set.")
        encoder_outputs = None
    else:
        encoder_output_path = resolve_config_path(encoder_output_path)
        if not os.path.exists(encoder_output_path):
            raise ValueError(f"Encoder output path does not exist: {encoder_output_path}")
        print(f"Loading encoder outputs from: {encoder_output_path}")
        with open(encoder_output_path, "rb") as f:
            encoder_outputs = pd.read_csv(encoder_output_path)
else:
    # Load and prepare data
    train_data, test_data, prompt_generator = get_data_from_config(cfg, verbose=True)
    print(f"Train data size: {len(train_data)}, Test data size: {len(test_data)}")
    # export train and test data to pickle
    data_checkpoint_path = os.path.join(DATA_DIR, cfg.dataset.name)
    # export cfg.dataset to config.yaml
    os.makedirs(data_checkpoint_path, exist_ok=True)
    with open(os.path.join(data_checkpoint_path, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    with open(os.path.join(data_checkpoint_path, "train_data"), "wb") as f:
        pickle.dump(train_data, f)
    with open(os.path.join(data_checkpoint_path, "test_data"), "wb") as f:
        pickle.dump(test_data, f)
    print(f"Saved train and test data to checkpoint: {data_checkpoint_path}")
    encoder_outputs = None  # will run encoder on train set below

# Load model
model = load_any_model(cfg.model)

# if encoder outputs were not found in checkpoint, compute them now and save to checkpoint
if encoder_outputs is None:
    print("Running encoder on train set to compute encoder outputs.")
    # Evaluate the model
    batch_size = cfg.evaluation.get("batch_size", 1)
    export_path = os.path.join(data_checkpoint_path, "encoder_outputs")
    os.makedirs(export_path, exist_ok=True)
    print("Batch size", batch_size)
    evaluator = ModelEvaluator(
        model=model,
        # test_data={k: v for i, (k, v) in enumerate(train_data.items()) if i < 15},
        test_data=train_data,
        batch_size=batch_size,
        verbose_freq=1,
        export_bool=True,
        export_path=export_path,
    )
    answers, accuracy, export_dir = evaluator.evaluate(config=cfg_dict)
    # load encoder_outputs from export_dir/evaluation_results.csv
    encoder_output_path = os.path.join(export_dir, "evaluation_results.csv")
    print(f"Loading encoder outputs from: {encoder_output_path}")
    with open(encoder_output_path, "rb") as f:
        encoder_outputs = pd.read_csv(encoder_output_path)
    print(f"Loaded encoder outputs with shape: {encoder_outputs.shape}")

prompts = []
true_labels = []
student_ids = []
for idx, row in encoder_outputs.iterrows():
    student_id = row["user_id"]
    bottleneck = row["output_bottleneck"]
    if type(bottleneck) is float:
        print(idx, bottleneck)
        continue
    decoder_prompts = eval(row["prompt"])[1]  # list of decoder prompts
    labels = eval(row["true_label"])  # list of true labels
    for dec_prompt, label in zip(decoder_prompts, labels):
        assert len(dec_prompt) > 0, "Decoder prompt is empty"
        assert len(bottleneck) > 0, "bottleneck is empty"
        full_prompt = add_instructions_to_yprompt(dec_prompt, bottleneck)
        # full_prompt = add_instructions_to_yprompt(dec_prompt, bottleneck, template=model.y_template)
        prompts.append(full_prompt)
        true_labels.append(label)
        student_ids.append(student_id)
print(f"Prepared {len(prompts)} decoder training examples.")

# Set up wandb
if int(os.environ.get("RANK", 0)) == 0:
    wandb_tags = cfg.get("wandb_tags", None)
    wandb.init(project="lbm", name=get_run_name(cfg_dict), config=cfg_dict, tags=wandb_tags)
    wandb.config.update(cfg_dict)

# Make an SFTDataset
full_dataset = Dataset.from_list([{"prompt": p, "completion": l} for p, l in zip(prompts, true_labels)])
# train val split

decoder_train_data, decoder_val_data = full_dataset.train_test_split(test_size=0.1).values()


def tokenize_sft(example):
    # Tokenize prompt alone (no EOS)
    prompt_enc = model.decoder.tokenizer(
        example["prompt"],
        add_special_tokens=False,
    )

    # Tokenize completion alone (NO EOS for now)
    completion_enc = model.decoder.tokenizer(
        example["completion"],
        add_special_tokens=False,
    )

    # Concatenate
    input_ids = prompt_enc["input_ids"] + completion_enc["input_ids"]

    attention_mask = [1] * len(input_ids)

    # Labels:
    #  - mask prompt tokens
    #  - supervise completion tokens
    labels = [-100] * len(prompt_enc["input_ids"]) + completion_enc["input_ids"]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


train_tok = [tokenize_sft(decoder_train_data[i]) for i in range(len(decoder_train_data))]
val_tok = [tokenize_sft(decoder_val_data[i]) for i in range(len(decoder_val_data))]
train_tok_dataset = Dataset.from_list(train_tok)
val_tok_dataset = Dataset.from_list(val_tok)

# Print example training samples
for i in range(3):
    print("Prompt:", decoder_train_data[i]["prompt"])
    print("Completion:", decoder_train_data[i]["completion"])
    print("-----")

output_dir = get_default_results_dir()
training_args = SFTConfig(
    output_dir=output_dir,
    **cfg.training,
)

model.decoder.model.enable_input_require_grads()

lora_config = LoraConfig(**cfg_dict["lora_config"]) if "lora_config" in cfg_dict else LoraConfig(**DEFAULT_LORA_CONFIG)
print(lora_config)

trainer = SFTTrainer(
    model=model.decoder.model,
    processing_class=model.decoder.tokenizer,
    args=training_args,
    train_dataset=train_tok_dataset,
    eval_dataset=val_tok_dataset,
    peft_config=lora_config,
)

trainer.train()

# Final eval
answers, accuracy, export_dir = evaluator.evaluate()
print(f"Model accuracy: {accuracy:.2f}")
wandb.config.update({"export_dir": export_dir})
wandb.log({"Final test accuracy": accuracy})
