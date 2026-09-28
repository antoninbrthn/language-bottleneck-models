from hydra import compose, initialize_config_dir
from transformers import AutoTokenizer
from omegaconf import OmegaConf
import numpy as np
import wandb
import sys

from src.data.synthetic_loader import SyntheticLoader
from src.models.model_loader import load_any_model
from src.data.utils import get_data_from_config
from src.models.api_models import load_api_model
from src.evaluation.evaluate_model import ModelEvaluator
from src.utils.config import CONFIGS_DIR
from src.prompts.prompt_generator import PromptGenerator
from src.utils.misc import get_run_name

def run_evaluation(config_name="config-ground-truth", overrides=[]):
    with initialize_config_dir(version_base=None, config_dir=CONFIGS_DIR):
        cfg = compose(config_name=config_name, overrides=overrides)

    cfg_dict = OmegaConf.to_object(cfg)
    print('Config:', cfg_dict)

    # Load the OpenAI model using the model name from the config
    model = load_any_model(cfg.model)

    # Load and prepare data
    train_data, test_data, prompt_generator = get_data_from_config(cfg, verbose=True)
    # load all data
    loader = SyntheticLoader(**cfg.dataset)
    students_df, questions_df, data = loader.load_data(return_full=True)
    # Example of getting the ground truth knowledge state for a specific user
    user_id = '1340'
    dt_stud = students_df[students_df['student_id'] == int(user_id)].iloc[0]
    dt_stud.knowledge_state

    
    # Set up wandb
    wandb_tags = cfg.get("wandb_tags", None)
    wandb.init(project="lbm",
            name=get_run_name(cfg_dict),
            config=cfg_dict,
            tags=wandb_tags)

    wandb.config.update(cfg_dict)

    extra_args = {'students_df': students_df}

    # Evaluate the model
    print('Batch size', cfg.evaluation.get("batch_size", 1))
    import time
    st = time.time()
    evaluator = ModelEvaluator(
        model=model,
        test_data=test_data,
        batch_size=cfg.evaluation.get("batch_size", 1),
        verbose_freq=1,
        export_bool=cfg.evaluation.get("export_results", False),
        export_path=cfg.evaluation.get("export_path", "exports"),
        **extra_args,
    )
    answers, accuracy, export_dir = evaluator.evaluate(config=cfg_dict)
    print('Export dir:', export_dir)
    wandb.config.update({"export_dir": export_dir})
    print(f"Model accuracy: {accuracy:.2f}")
    print('Took:', time.time()-st)
    wandb.log({"Test accuracy": accuracy})
    if hasattr(model, 'total_price'):
        wandb.log({"Total price": model.total_price})
        

if __name__ == "__main__":
    # Convert command line arguments to override list

    # overrides = sys.argv[1:]
    # config_name, overrides
    config_name = sys.argv[1]
    overrides = sys.argv[2:]
    run_evaluation(config_name, overrides)
