from pathlib import Path
import yaml

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # atajos útiles
    batch_size = cfg["data"]["batch_size"]
    block_size = cfg["data"]["block_size"]
    target_tokens = cfg["training"]["target_tokens"]
    num_gpus = cfg["training"]["num_gpus"]

    grad_acc_global = target_tokens // (batch_size * block_size)
    grad_acc_global = (grad_acc_global // num_gpus) * num_gpus
    cfg["training"]["gradient_accumulation_steps"] = grad_acc_global

    return cfg