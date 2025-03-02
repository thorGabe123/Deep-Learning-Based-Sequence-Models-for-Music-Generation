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
config = dict_to_namespace(load_config())

# Compute VOCAB_SIZE dynamically
vocab_size = sum(vars(config.discretization).values()) + 1

# Compute START_IDX dynamically
start_idx = {}
offset = 1  # Start index

for key, value in vars(config.discretization).items():
    start_idx[key] = offset
    offset += value  # Move the index forward

