"""Configuration loader for lrppo"""

from pathlib import Path
import yaml


def load_config(path):
    """Load YAML config from `path` and return a dict.

    TODO: add validation and default values for missing fields.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file does not exist: {p}")
    with p.open('r') as f:
        cfg = yaml.safe_load(f)
    return cfg
