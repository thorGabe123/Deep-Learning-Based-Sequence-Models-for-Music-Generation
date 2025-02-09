from pathlib import Path
import yaml
from types import SimpleNamespace

# Load YAML Config
def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# Convert dictionary to a SimpleNamespace object
def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d

# Load config
config_dict = load_config()
config = dict_to_namespace(config_dict)