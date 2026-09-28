from hydra import compose, initialize_config_dir
from transformers import AutoTokenizer
from omegaconf import OmegaConf
import numpy as np
import wandb
import sys

from src.data.synthetic_loader import SyntheticLoader
from src.models.model_loader import load_any_model
from src.data.utils import get_data_from_config
from src.evaluation.evaluate_model import ModelEvaluator
from src.utils.config import CONFIGS_DIR
from src.prompts.prompt_generator import PromptGenerator
from src.utils.misc import get_run_name

def run_evaluation(config_name="config-lbm-3", overrides=[]):
    print('Using config:', config_name, "with overrides:", overrides)
    with initialize_config_dir(version_base=None, config_dir=CONFIGS_DIR):
        cfg = compose(config_name=config_name, overrides=overrides)

    cfg_dict = OmegaConf.to_object(cfg)
    print('Config:', cfg_dict)

    # Load the OpenAI model using the model name from the config
    model = load_any_model(cfg.model)

    # Load and prepare data
    train_data, test_data, prompt_generator = get_data_from_config(cfg, verbose=True)
    if hasattr(model, "set_contains"):
        assert prompt_generator.contains is not None, "Prompt generator should have contains attribute for iterative model"
        model.set_contains(prompt_generator.contains)        # one-time injection
    # same for lbm.set_student_dict(students_dict)
    if hasattr(model, "set_student_dict"):
        # load synthetic
        assert "synthetic" in cfg.dataset.name, "Synthetic dataset required for LBMActive"
        from src.synthetic_dataset.students import Student

        loader = SyntheticLoader(**cfg.dataset)
        students_df, questions_df, data = loader.load_data(return_full=True)
        students_dict = {
            s.student_id: Student.from_series(s) for _, s in students_df.iterrows()
        }
        model.set_student_dict(students_dict) 

    # Set up wandb
    wandb_tags = cfg.get("wandb_tags", None)
    wandb.init(project="lbm",
            name=get_run_name(cfg_dict),
            config=cfg_dict,
            tags=wandb_tags)

    wandb.config.update(cfg_dict)


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
    if len(sys.argv) < 2:
        run_evaluation()  # default params
    else:
        config_name = sys.argv[1]
        overrides = sys.argv[2:]
        run_evaluation(config_name, overrides)
