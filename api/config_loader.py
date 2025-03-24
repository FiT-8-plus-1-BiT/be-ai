import yaml
from pathlib import Path

def load_config(profile="local"):
    config_path = Path(__file__).resolve().parent / "jwt_module" / f"application-{profile}.yml"
    print("🔍 config path:", config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config