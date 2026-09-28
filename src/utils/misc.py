import os

from src.utils.config import RESULTS_DIR


def get_run_name(cfg_dict):
    dataset_name = cfg_dict['dataset']['name']

    if cfg_dict['model'].get('type', None) == 'lbm':
        enc_name = cfg_dict['model']['encoder']['name'].split("/")[-1]
        enc_cot = 'T' if cfg_dict['model']['encoder']['use_cot'] else 'F'
        dec_name = cfg_dict['model']['decoder']['name'].split("/")[-1]
        dec_cot = 'T' if cfg_dict['model']['decoder']['use_cot'] else 'F'
        run_name = f"{dataset_name}_LBM_{enc_name}_cot{enc_cot}_{dec_name}_cot{dec_cot}"
        return run_name
    model_name = cfg_dict['model']['name'].split("/")[-1]
    if 'use_cot' not in cfg_dict['model']:
        run_name = f"{dataset_name}_{model_name}"
    else:
        use_cot = 'T' if cfg_dict['model']['use_cot'] else 'F'
        run_name = f"{dataset_name}_{model_name}_cot{use_cot}"
    return run_name

def get_default_results_dir():
    import wandb    
    # if wandb is active, use `results/trained_models/lbm_{wandb_run_id}`
    if wandb.run:
        wandb_run_id = wandb.run.id
        output_dir = os.path.join(RESULTS_DIR, "trained_models", f"lbm_{wandb_run_id}")
    else:  # otherwise, use `results/trained_models/lbm_{timestamp}`
        import time
        run_id = time.strftime('%Y%m%d-%H%M%S')
        output_dir = os.path.join(RESULTS_DIR, "trained_models", f"lbm_{run_id}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def flatten_dict(d):
    "{'a': {'b': 1, ..}} -> {'a.b': 1, ..}. Make recursive."
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in flatten_dict(v).items():
                out[f"{k}.{k2}"] = v2
        else:
            out[k] = v
    return out
