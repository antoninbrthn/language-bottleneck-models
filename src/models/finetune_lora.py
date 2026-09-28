import numpy as np
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
import wandb
from transformers import TrainerCallback
from src.evaluation.evaluate_model import ModelEvaluator
from src.utils.config import LOGS_DIR, RESULTS_DIR
from src.models.hf_models import HFModel

class LoRADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, label = self.data[idx]
        # print warning if longer than max_length
        prompt_token_length = len(self.tokenizer(prompt)['input_ids'])
        if prompt_token_length > self.max_length:
            print(f"Warning: input sequence of length {prompt_token_length} longer than max_length: {self.max_length}. Will be truncated.")

        # Encode input sequence
        encoding = self.tokenizer(prompt, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt")

        # Encode label - we'll use -100 for padding tokens
        label_ids = torch.full((self.max_length,), -100)

        # Get the label token IDs
        label_tokens = self.tokenizer.encode(label, add_special_tokens=False)

        # Put the label tokens at the end of the sequence
        # This assumes the label is short (like "Yes" or "No")
        input_length = encoding['attention_mask'].sum().item()
        label_ids[input_length - len(label_tokens):input_length] = torch.tensor(label_tokens)

        return {'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(), 'labels': label_ids}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode predictions and labels
    predictions = np.argmax(predictions, axis=-1)
    labels = labels[labels != -100]  # Ignore padding tokens
    predictions = predictions[labels != -100]

    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

def get_hash():
    # make a hash for the run name in wandb
    import hashlib
    import time
    return hashlib.md5(str(time.time()).encode()).hexdigest()[:6]

DEFAULT_LORA_CONFIG = dict(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

class FineTuner:
    DEFAULT_TRAINING_CONFIG = dict(
        output_dir=RESULTS_DIR,
        # run_name="default_run",
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        save_total_limit=2,
        logging_dir=LOGS_DIR,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        # gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        fp16=True,
    )

    def __init__(self, hf_model, tokenizer, lora_config={}, training_config={}):
        self.model = hf_model
        self.tokenizer = tokenizer
        
        # LoRA configuration
        lora_config_values = DEFAULT_LORA_CONFIG.copy()
        lora_config_values.update(lora_config)
        self.lora_config = LoraConfig(**lora_config_values)
        # set from default values + possible overrides
        self.training_config = self.DEFAULT_TRAINING_CONFIG.copy()
        self.training_config.update(training_config)
        
        # Log LoRA config to wandb
        wandb.config.update({"lora_config": self.lora_config.to_dict()})
        
        # Prepare model for LoRA
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()

    def finetune(self, train_dataset, val_dataset, config):
        training_args = TrainingArguments(
            **{**self.training_config, **config['training']}
        )
        print('Training with config', training_args)
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EvaluationCallback(self.tokenizer, val_dataset.data)],
        )
        
        trainer.train()
        return self.model


class EvaluationCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_data):
        self.tokenizer = tokenizer
        self.eval_data = eval_data

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        
        evaluator = ModelEvaluator(
            model=HFModel(model, self.tokenizer),
            test_data=self.eval_data
        )
        _, accuracy = evaluator.evaluate()

        print(f"Epoch {int(state.epoch)} - Validation Accuracy: {accuracy:.2f}")
        wandb.log({"Validation Accuracy": accuracy, "epoch": int(state.epoch)})


