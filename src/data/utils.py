from src.data.utils_iterative import load_and_prepare_data_iterative
from src.utils.config import DATA_DIR
from src.data.dataset_loader import DatasetLoader
from src.prompts.prompt_generator import PromptGenerator
import numpy as np
import os
import pickle

get_txt_answer = lambda x: "Yes" if x == 1 else "No"


def resolve_config_path(path_value: str) -> str:
    """Resolve config paths with optional placeholders and env vars.

    Supports:
    - ``{{DATA_DIR}}`` token replacement
    - shell env vars (e.g. ``$DATA_DIR``)
    - user home expansion (``~``)
    - fallback to ``DATA_DIR`` for relative paths
    """
    if path_value is None:
        return None

    resolved = str(path_value).replace("{{DATA_DIR}}", DATA_DIR)
    resolved = os.path.expanduser(os.path.expandvars(resolved))
    if not os.path.isabs(resolved):
        resolved = os.path.join(DATA_DIR, resolved)
    return resolved


def load_and_prepare_data(loader: DatasetLoader, prompt_generator: PromptGenerator, n_y_prompts=1):
    """Create prompts for a standard model.

    Args:
        loader: DatasetLoader instance
        prompt_generator: PromptGenerator instance
        n_y_prompts: Number of new questions to generate per input

    Returns:
        tuple: (train_data, test_data) where train_data and test_data are dictionaries
               with user_id keys containing (prompt, label) pairs
    """
    data = loader.load_data()
    trajectories, user_ids = loader.preprocess(data)
    train_data, test_data = loader.split_data(trajectories)
    train_ids, test_ids = loader.split_data(user_ids)

    # Convert to dictionaries with user_id keys
    train_data_dict = {}
    test_data_dict = {}

    if n_y_prompts > 1:  # Use multiple y prompts
        for user_id, trajectory in zip(train_ids, train_data):
            prompts = prompt_generator.create_prompt_multiple_new_qs(trajectory[:-n_y_prompts], trajectory[-n_y_prompts:])
            labels = [get_txt_answer(t["correct"]) for t in trajectory[-n_y_prompts:]]
            train_data_dict[user_id] = (prompts, labels)

        for user_id, trajectory in zip(test_ids, test_data):
            prompts = prompt_generator.create_prompt_multiple_new_qs(trajectory[:-n_y_prompts], trajectory[-n_y_prompts:])
            labels = [get_txt_answer(t["correct"]) for t in trajectory[-n_y_prompts:]]
            test_data_dict[user_id] = (prompts, labels)
    else:
        for user_id, trajectory in zip(train_ids, train_data):
            prompt = prompt_generator.create_prompt(trajectory[:-1], trajectory[-1])
            label = get_txt_answer(trajectory[-1]["correct"])
            train_data_dict[user_id] = (prompt, label)

        for user_id, trajectory in zip(test_ids, test_data):
            prompt = prompt_generator.create_prompt(trajectory[:-1], trajectory[-1])
            label = get_txt_answer(trajectory[-1]["correct"])
            test_data_dict[user_id] = (prompt, label)

    return train_data_dict, test_data_dict


def load_and_prepare_data_bottleneck(loader: DatasetLoader, prompt_generator: PromptGenerator, n_y_prompts=1):
    """Create prompts for a bottleneck model, separating the observed data from the new question.

    Args:
        loader: DatasetLoader instance
        prompt_generator: PromptGenerator instance
        n_y_prompts: Number of new questions to generate per input

    Returns:
        tuple: (train_data, test_data) where train_data and test_data are dictionaries
               with user_id keys containing (prompt, label) pairs
    """
    data = loader.load_data()
    trajectories, user_ids = loader.preprocess(data)
    train_data, test_data = loader.split_data(trajectories)
    train_ids, test_ids = loader.split_data(user_ids)

    # Convert to dictionaries with user_id keys
    train_data_dict = {}
    test_data_dict = {}

    if n_y_prompts > 1:  # Use multiple y prompts
        for user_id, trajectory in zip(train_ids, train_data):
            prompt = prompt_generator.create_prompt_bottleneck_multiple_new_qs(trajectory[:-n_y_prompts], trajectory[-n_y_prompts:])
            labels = [get_txt_answer(t["correct"]) for t in trajectory[-n_y_prompts:]]
            train_data_dict[user_id] = (prompt, labels)

        for user_id, trajectory in zip(test_ids, test_data):
            prompt = prompt_generator.create_prompt_bottleneck_multiple_new_qs(trajectory[:-n_y_prompts], trajectory[-n_y_prompts:])
            labels = [get_txt_answer(t["correct"]) for t in trajectory[-n_y_prompts:]]
            test_data_dict[user_id] = (prompt, labels)

    else:
        for user_id, trajectory in zip(train_ids, train_data):
            prompt = prompt_generator.create_prompt_bottleneck(trajectory[:-1], trajectory[-1])
            label = get_txt_answer(trajectory[-1]["correct"])
            train_data_dict[user_id] = (prompt, label)

        for user_id, trajectory in zip(test_ids, test_data):
            prompt = prompt_generator.create_prompt_bottleneck(trajectory[:-1], trajectory[-1])
            label = get_txt_answer(trajectory[-1]["correct"])
            test_data_dict[user_id] = (prompt, label)

    return train_data_dict, test_data_dict


def load_data_from_checkpoint(data_checkpoint_path):
    """
    Load train and test data from pickle files in checkpoint directory.
    """
    data_checkpoint_path = resolve_config_path(data_checkpoint_path)
    print(f"Loading data from checkpoint: {data_checkpoint_path}")
    if not os.path.exists(data_checkpoint_path):
        print(f"Using DATA_DIR from config.py to locate data checkpoint path.")
        data_checkpoint_path = os.path.join(DATA_DIR, data_checkpoint_path)
    if not os.path.exists(data_checkpoint_path):
        raise ValueError(f"Data checkpoint path does not exist: {data_checkpoint_path}")
    train_data_path = os.path.join(data_checkpoint_path, "train_data")
    test_data_path = os.path.join(data_checkpoint_path, "test_data")
    with open(train_data_path, "rb") as f:
        train_data = pickle.load(f)
    with open(test_data_path, "rb") as f:
        test_data = pickle.load(f)
    return train_data, test_data


def get_data_from_config(cfg, verbose=False):
    """Load and prepare data based on configuration.

    Args:
        cfg: Configuration object containing dataset and model settings
        verbose (bool): If True, print additional information about data loading

    Returns:
        tuple: (train_data, test_data, prompt_generator) where train_data and test_data are dictionaries
               with user_id keys containing (prompt, label) pairs
    """
    from src.data.eedi import EediLoader
    from src.data.eedi_filtered import EediFilteredLoader
    from src.data.xes3g5m_filtered import XES3G5MFilteredLoader
    from src.data.xes3g5m import XES3G5MLoader
    from src.data.synthetic_loader import SyntheticLoader
    from src.prompts.prompt_generator import PromptGenerator

    if "data_checkpoint_path" in cfg.dataset and cfg.dataset.data_checkpoint_path is not None:
        data_checkpoint_path = cfg.dataset.data_checkpoint_path
        train_data, test_data = load_data_from_checkpoint(data_checkpoint_path)
        prompt_generator = None  # Not needed when loading from checkpoint
        return train_data, test_data, prompt_generator

    # Initialize appropriate loader based on dataset name
    if cfg.dataset.name == "EEDI":
        loader = EediLoader(**cfg.dataset)
    elif cfg.dataset.name == "EEDIFiltered":
        loader = EediFilteredLoader(**cfg.dataset)
    elif cfg.dataset.name == "XES3G5MFiltered":
        loader = XES3G5MFilteredLoader(**cfg.dataset)
    elif cfg.dataset.name == "XES3G5M":
        loader = XES3G5MLoader(**cfg.dataset)
    elif "synthetic" in cfg.dataset.name:
        loader = SyntheticLoader(**cfg.dataset)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset.name}")

    prompt_contains = list(loader.contains) if hasattr(loader, "contains") else []
    use_construct = getattr(cfg.dataset, "use_construct", True)
    if not use_construct:
        print("Not using construct in prompts, removing from prompt_contains")
        prompt_contains = [c for c in prompt_contains if c not in ("construct_id", "construct_text")]
    prompt_generator = PromptGenerator(contains=prompt_contains)

    # Load and prepare data based on model type
    is_lbm = cfg.model.get("type", None) == "lbm"
    is_iterative = cfg.model.get("mode", "standard") == "iterative"
    is_steer = cfg.model.get("mode", "standard") == "steer"
    is_multiple_qs = cfg.dataset.get("n_y_prompts", 1) > 1
    is_training = bool(cfg.get("training", False))

    if (is_iterative or is_steer) and not is_training:
        train_data, test_data = load_and_prepare_data_iterative(loader, prompt_generator, n_y_prompts=cfg.dataset.get("n_y_prompts", 1))
        print("Loaded x questions as list of dicts (for LBM iter/steer)")
    elif is_lbm:
        train_data, test_data = load_and_prepare_data_bottleneck(loader, prompt_generator, n_y_prompts=cfg.dataset.get("n_y_prompts", 1))
        print("Loaded x questions as list of str (for vanilla LBM)")
    else:
        train_data, test_data = load_and_prepare_data(loader, prompt_generator, n_y_prompts=cfg.dataset.get("n_y_prompts", 1))
        print("Loaded all questions as list of str (for non LBM models)")

    if verbose and not (is_iterative or is_steer):
        print(f"Loaded {len(train_data)} train/{len(test_data)} test trajectories")
        # Print average prompt length in train_data in number of words
        if is_multiple_qs:
            nb_words = [len(t.split()) for (t, _), _ in train_data.values()] if is_lbm else [len(t[0].split()) for t, _ in train_data.values()]
        else:
            nb_words = [len(t.split()) for (t, _), _ in train_data.values()] if is_lbm else [len(t.split()) for t, _ in train_data.values()]
        print("Average nb of tokens in prompt:", np.mean(nb_words) * 0.75, "+/-", np.std(nb_words) * 0.75)
        print("95% quantile of prompt length in tokens:", np.quantile(nb_words, 0.95))

    return train_data, test_data, prompt_generator
