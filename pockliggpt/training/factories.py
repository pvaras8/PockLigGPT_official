from pockliggpt.training.loaders import SequenceAddLoader, SequenceLoader


def build_model_classes():
    from pockliggpt.models.model_sequence import GPT, GPTConfig

    return GPTConfig, GPT


def build_loader(cfg: dict, device: str, device_type: str):
    loader_type = cfg["data"]["loader_type"]

    if loader_type == "sequence":
        return SequenceLoader(cfg, device=device, device_type=device_type)

    if loader_type == "sequence_add":
        return SequenceAddLoader(cfg, device=device, device_type=device_type)

    raise ValueError(f"Unsupported loader type: {loader_type}")
