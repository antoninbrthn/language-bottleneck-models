# train_lbm_unsloth.py
import unsloth
from unsloth import FastLanguageModel
from datasets import load_dataset
import torch
from trl import GRPOConfig, GRPOTrainer
from hydra import initialize, compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
import wandb
import os 
from transformers import TrainerCallback
from peft import LoraConfig
from src.models.finetune_lbm import GRPODataset, create_reward_function, create_reward_function_steer, format_reward_func, misconception_reward_func
from src.data.utils import get_data_from_config
from src.models.model_loader import load_any_model
from src.models.finetune_lora import FineTuner, LoRADataset, DEFAULT_LORA_CONFIG
from src.evaluation.evaluate_model import ModelEvaluator
from src.utils.config import CONFIGS_DIR, RESULTS_DIR
from src.utils.misc import get_default_results_dir, get_run_name
from src.models.hf_models import HFModelChat
from src.prompts.prompt_utils import load_prompt_template
from src.data.synthetic_loader import SyntheticLoader
from src.synthetic_dataset.students import Student

# parse args for config_name
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config_name", type=str, default="config-train-lbm")
args = parser.parse_args()
config_name = args.config_name

# Initialize Hydra and load the configuration
with initialize_config_dir(version_base=None, config_dir=CONFIGS_DIR):
    cfg = compose(config_name=config_name)
cfg_dict = OmegaConf.to_object(cfg)
print('Config:', cfg_dict)

# Load and prepare data
train_data, test_data, prompt_generator = get_data_from_config(cfg, verbose=True)

# Load student data if training with student-specific information
student_dict = None
if "synthetic" in cfg.dataset.name and cfg.model.get("mode", "standard")=="steer":
    print("Loading student data for additional info processing...")
    loader = SyntheticLoader(**cfg.dataset)
    students_df, questions_df, data = loader.load_data(return_full=True)
    student_dict = {
        s.student_id: Student.from_series(s) for _, s in students_df.iterrows()
    }
    print(f"Loaded {len(student_dict)} students")

# Load original model (needed for decoder)
model = load_any_model(cfg.model)

# Get template information from config or use defaults
x_template_file = cfg.model.get("x_template_file", "base_x.yaml")
y_template_file = cfg.model.get("y_template_file", "base_y.yaml")

# Get additional_x_info from config if provided
additional_x_info = cfg.model.get("additional_x_info", None)
additional_bn_info = cfg.model.get("additional_bn_info", None)

# Load templates
x_template = load_prompt_template(filename=x_template_file)
y_template = load_prompt_template(filename=y_template_file)

print(f"Using X template from: {x_template_file}")
print(f"Using Y template from: {y_template_file}")
if additional_x_info:
    print(f"Using additional_x_info: {additional_x_info}")
if additional_bn_info:
    print(f"Using additional_bn_info: {additional_bn_info}")

# Set up wandb
wandb_tags = cfg.get("wandb_tags", None)
wandb.init(project="lbm",
           name=get_run_name(cfg_dict) + "-unsl",
           config=cfg_dict,
           tags=wandb_tags)
           
wandb.config.update(cfg_dict)

# Map reward function names to their implementations
REWARD_FUNCTIONS = {
    "acc": lambda: create_reward_function(model.decoder, 
                                         decoder_batch_size=cfg.model.decoder.get('batch_size', None),
                                         y_template=y_template),
    "acc-steer": lambda: create_reward_function_steer(model.decoder, 
                                         decoder_batch_size=cfg.model.decoder.get('batch_size', None),
                                         y_template=y_template,
                                         additional_bn_info=cfg.model.get("additional_bn_info", None),
                                         student_dict=student_dict),
    "format": format_reward_func,
    "misc": misconception_reward_func
}

# Get the list of reward functions to use from config
# Default to ["acc", "format"] if not specified
reward_func_names = cfg.get("training_loss", ["acc", "format"])
print(f"Using reward functions: {reward_func_names}")

# Create the list of reward functions based on config
reward_funcs = []
for name in reward_func_names:
    if name in REWARD_FUNCTIONS:
        func = REWARD_FUNCTIONS[name]
        # Call the function if it's a factory function (like "acc")
        if callable(func) and "acc" in name:
            reward_funcs.append(func())
        else:
            reward_funcs.append(func)
    else:
        print(f"Warning: Unknown reward function '{name}'. Skipping.")

# Set up Unsloth for the encoder model
encoder_model_name = model.encoder.model.config._name_or_path

# Get Unsloth model parameters
max_seq_length = cfg.dataset.max_context_length
lora_rank = cfg_dict['lora_config']['r'] if 'lora_config' in cfg_dict else 8

# Initialize FastLanguageModel
fast_model, vllm_tokenizer = FastLanguageModel.from_pretrained(
    model_name=encoder_model_name,
    max_seq_length=int(1.1*max_seq_length),  # more headroom
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.6,
)

# Get PEFT model with LoRA config
target_modules = cfg_dict['lora_config']['target_modules'] if 'lora_config' in cfg_dict else ["q_proj", "v_proj"]
lora_alpha = cfg_dict['lora_config']['lora_alpha'] if 'lora_config' in cfg_dict else 16

fast_model = FastLanguageModel.get_peft_model(
    fast_model,
    r=lora_rank,
    target_modules=target_modules,
    lora_alpha=lora_alpha,
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# Set up training arguments
output_dir = get_default_results_dir()
training_args = GRPOConfig(
    output_dir=output_dir,
    **cfg.training,
)

# Setup evaluator
evaluator = ModelEvaluator(
    model=model,
    test_data=test_data,
    batch_size=cfg.evaluation.batch_size
)

# Initial evaluation before training
if cfg.evaluation.eval_before:
    answers, accuracy, export_dir = evaluator.evaluate()
    print(f"Epoch {0} - Validation Accuracy: {accuracy:.2f}")
    wandb.log({"Validation Accuracy": accuracy, "epoch": 0})

# Prepare datasets with templates
train_dataset = GRPODataset(
    train_data, 
    vllm_tokenizer, 
    conversational=True, 
    max_prompt_length=cfg.dataset.max_context_length, 
    use_cot=cfg.model.encoder.use_cot, 
    max_bottleneck_tokens=cfg.model.max_bottleneck_tokens,
    x_template=x_template,
    additional_x_info=additional_x_info,
    student_dict=student_dict
)

val_dataset = GRPODataset(
    test_data, 
    vllm_tokenizer, 
    conversational=True, 
    max_prompt_length=cfg.dataset.max_context_length, 
    use_cot=cfg.model.encoder.use_cot, 
    max_bottleneck_tokens=cfg.model.max_bottleneck_tokens,
    x_template=x_template,
    additional_x_info=additional_x_info,
    student_dict=student_dict
)

# Callback to log the total price of the model at each epoch
class LogTotalPriceCallback(TrainerCallback):
    def on_log(self, args, state, control, **kwargs):
        try:
            wandb.log({"Total price": model.total_price})
        except Exception as e:
            print("Could not log total price:", e)

# Initialize GRPOTrainer with Unsloth model
trainer = GRPOTrainer(
    model=fast_model,
    processing_class=vllm_tokenizer,
    reward_funcs=reward_funcs,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[LogTotalPriceCallback()],
)

# Export initial checkpoint of the model under {output_dir}/checkpoint-0/
try:
    os.makedirs(os.path.join(output_dir, "checkpoint-0"), exist_ok=True)
    trainer.save_model(os.path.join(output_dir, "checkpoint-0"))
    print(f"Initial checkpoint saved to {output_dir}/checkpoint-0")
except Exception as e:
    print(f"Error saving initial checkpoint: {e}")

# Train the model
trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

# Replace the encoder model in the original model with the trained model for evaluation
# This is a shallow copy - we're just replacing the encoder model reference
model.encoder.model = fast_model

# Final evaluation
answers, accuracy, export_dir = evaluator.evaluate()
print(f"Model accuracy: {accuracy:.2f}")
wandb.config.update({"export_dir": export_dir})
wandb.log({"Final test accuracy": accuracy}) 