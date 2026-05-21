from pockliggpt.training.loaders import SequenceAddLoader


def build_model_classes(model_type: str):
    if model_type == "sequence_add":
        from pockliggpt.models.model_sequence import GPTConfig, GPT
        return GPTConfig, GPT

    raise ValueError(f"Unsupported model type: {model_type}")


def build_loader(cfg: dict, device: str, device_type: str):
    loader_type = cfg["data"]["loader_type"]

    if loader_type == "sequence_add":
        return SequenceAddLoader(cfg, device=device, device_type=device_type)

    raise ValueError(f"Unsupported loader type: {loader_type}")