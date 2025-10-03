import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

def load_config():
    """Loads the YAML configuration file."""
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    return config

config = load_config()

def get_config():
    """Returns the loaded configuration."""
    return config