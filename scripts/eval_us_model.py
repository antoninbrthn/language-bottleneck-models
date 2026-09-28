import sys
from datasets import load_dataset
import torch
from trl import GRPOConfig, GRPOTrainer
from hydra import initialize, compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
import wandb
import os 
from transformers import TrainerCallback
from peft import LoraConfig
from src.models.finetune_lbm import GRPODataset, create_reward_function, format_reward_func
from src.data.utils import get_data_from_config
from src.models.model_loader import load_any_model
from src.models.finetune_lora import FineTuner, LoRADataset, DEFAULT_LORA_CONFIG
from src.evaluation.evaluate_model import ModelEvaluator
from src.utils.config import CONFIGS_DIR, RESULTS_DIR
from src.utils.misc import get_default_results_dir, get_run_name

# overrides=[]
# config_name="config-zs-trained-us"
def run_evaluation(config_name="config-zs-trained-us", overrides=[]):
    with initialize_config_dir(version_base=None, config_dir=CONFIGS_DIR):
        cfg = compose(config_name=config_name, overrides=overrides)

    cfg_dict = OmegaConf.to_object(cfg)
    print('Config:', cfg_dict)


    # Load and prepare data
    train_data, test_data, prompt_generator = get_data_from_config(cfg, verbose=True)

    # Load original model (needed for decoder)
    model = load_any_model(cfg.model)

    # Set up wandb
    wandb_tags = cfg.get("wandb_tags", None)
    wandb.init(project="lbm",
            name=get_run_name(cfg_dict) + "-unsl",
            config=cfg_dict,
            tags=wandb_tags)
            
    wandb.config.update(cfg_dict)

    # Setup evaluator
    evaluator = ModelEvaluator(
        model=model,
        test_data=test_data,
        batch_size=cfg.evaluation.batch_size,
        export_bool=cfg.evaluation.get("export_results", False),
        export_path=cfg.evaluation.get("export_path", "results"),
        verbose_freq=cfg.evaluation.get("verbose_freq", 0.1),
    )

    # Final evaluation
    answers, accuracy, export_dir = evaluator.evaluate()
    print(f"Model accuracy: {accuracy:.2f}")
    wandb.config.update({"export_dir": export_dir})
    wandb.log({"Final test accuracy": accuracy}) 


if __name__ == "__main__":
    config_name = sys.argv[1]
    overrides = sys.argv[2:]
    run_evaluation(config_name, overrides)
