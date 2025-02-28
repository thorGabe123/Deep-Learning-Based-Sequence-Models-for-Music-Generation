from pathlib import Path
import yaml

config_path = Path(__file__).parent / "config.yaml"
with open(config_path, "r") as file:
    config_dict = yaml.safe_load(file)
