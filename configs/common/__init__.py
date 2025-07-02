from pathlib import Path
import yaml
import json
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

def load_tokenizations():
    # metadata_path = "/mnt/e/github/dataset/midi_dataset/tokenizations.json"
    metadata_path = "E:/GitHub/dataset/midi_dataset/tokenization.json"
    with open(metadata_path, "r") as file:
        return json.load(file)

# Load config 
config = dict_to_namespace(load_config())

# Load config
tokenizations = dict_to_namespace(load_tokenizations())

# Compute VOCAB_SIZE dynamically
vocab_size = sum([
    config.discretization.pitch * config.discretization.channel,
    config.discretization.dyn,
    config.discretization.length,
    config.discretization.time,
    config.discretization.tempo
])

metadata_vocab_size = tokenizations.VOCAB_SIZE

# Compute START_IDX dynamically
offset = 0
start_idx = {}

start_idx['pitch'] = offset
offset += config.discretization.pitch * config.discretization.channel

start_idx['dyn'] = offset
offset += config.discretization.dyn

start_idx['length'] = offset
offset += config.discretization.length

start_idx['time'] = offset
offset += config.discretization.time

start_idx['tempo'] = offset