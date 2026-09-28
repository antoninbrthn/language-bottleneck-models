from pathlib import Path

import yaml

from src.data.synthetic_loader import SyntheticLoader
from src.utils.config import CONFIGS_DIR


def test_all_configs_are_valid_yaml():
    config_paths = sorted(Path(CONFIGS_DIR).glob("*.yaml"))
    assert config_paths
    for config_path in config_paths:
        with config_path.open() as handle:
            config = yaml.safe_load(handle)
        assert isinstance(config, dict)
        assert "dataset" in config
        assert "model" in config


def test_included_synthetic_dataset_loads():
    loader = SyntheticLoader(
        name="synthetic-11",
        min_trajectory_length=1,
        max_trajectory_length=3,
        max_n_trajectories=2,
    )
    data = loader.load_data()
    trajectories, user_ids = loader.preprocess(data)

    assert not data.empty
    assert len(trajectories) == len(user_ids) == 2
    assert len(trajectories[0]) == 3
