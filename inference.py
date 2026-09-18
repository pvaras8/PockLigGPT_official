#!/usr/bin/env python3
"""Generate pocket-conditioned ligands from a PPO checkpoint."""

import argparse
import os
import pickle
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import selfies as sf
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from pockliggpt.rl.agent import PPOAgent
from pockliggpt.rl.conditioning import validate_conditioning_assets
from pockliggpt.rl.model_adapters import build_model_adapter, get_torch_dtype
from pockliggpt.rl.prompts import PromptBuilder, PromptDataset, strip_to_ligand


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera ligandos condicionados por pocket con un checkpoint PPO"
    )
    parser.add_argument("--config", required=True, help="YAML de inferencia")
    parser.add_argument("--checkpoint", default=None, help="Sobrescribe model.ppo_checkpoint_path")
    parser.add_argument("--num-ligands", "-n", type=int, default=None)
    parser.add_argument("--output", default=None, help="Sobrescribe output.csv_path")
    parser.add_argument("--pocket-str-path", default=None)
    parser.add_argument("--pocket-emb-path", default=None)
    parser.add_argument(
        "--base-checkpoint",
        default=None,
        help="Sobrescribe model.base_checkpoint_path del YAML",
    )
    parser.add_argument(
        "--seed-smiles-csv",
        default=None,
        help="Sobrescribe data.seed_smiles_csv",
    )
    parser.add_argument("--seed-smiles-column", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="auto, cpu, cuda o, por ejemplo, cuda:0",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default=None,
        help="Por defecto usa system.dtype del YAML",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--max-attempt-multiplier",
        type=int,
        default=None,
        help="Máximo de intentos por ligando válido solicitado",
    )
    return parser.parse_args()


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Se pidió CUDA, pero torch.cuda.is_available() es False")
    return requested


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_tokenizer(meta_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"No existe tokenizer.meta_path: {meta_path}")
    with open(meta_path, "rb") as handle:
        meta = pickle.load(handle)
    return meta["stoi"], meta["itos"]


def _clean_outer_prefix(state_dict: dict) -> dict:
    """Remove prefixes commonly introduced by DDP/torch.compile."""
    cleaned = {}
    for key, value in state_dict.items():
        while key.startswith("module."):
            key = key[len("module.") :]
        key = key.replace("model._orig_mod.", "model.", 1)
        cleaned[key] = value
    return cleaned


def _load_ppo_weights(agent: PPOAgent, checkpoint_path: str, device: str) -> None:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No existe el checkpoint PPO: {checkpoint_path}")

    # Mantener el checkpoint (que también contiene el estado del optimizador)
    # en CPU evita ocupar innecesariamente varios GB adicionales de VRAM.
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("El checkpoint PPO debe ser un diccionario")

    if "model_state_dict" in checkpoint:
        state_dict = _clean_outer_prefix(checkpoint["model_state_dict"])
        agent.load_state_dict(state_dict, strict=True)
    elif "model" in checkpoint:
        # Permite usar un checkpoint de modelo convencional como alternativa.
        state_dict = PPOAgent._clean_state_dict(checkpoint["model"])
        agent.model.load_state_dict(state_dict, strict=True)
    else:
        raise KeyError(
            "El checkpoint no contiene 'model_state_dict' (PPO) ni 'model'"
        )


def _decode_generated(ids: Iterable[int], itos: Dict[int, str]) -> str | None:
    text = "".join(itos.get(int(token_id), "") for token_id in ids)
    ligand_selfies = strip_to_ligand(text)
    if not ligand_selfies:
        return None
    try:
        smiles = sf.decoder(ligand_selfies)
    except Exception:
        return None
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return smiles.strip()


def _apply_overrides(cfg, args: argparse.Namespace) -> None:
    # Esta herramienta es deliberadamente pocket-conditioned, incluso si el YAML
    # conservaba conditioning.enabled=false de una ejecución no condicionada.
    cfg.conditioning.enabled = True
    if args.pocket_str_path:
        cfg.conditioning.pocket_str_path = args.pocket_str_path
        cfg.conditioning.pocket_str = ""
    if args.pocket_emb_path:
        cfg.conditioning.pocket_emb_path = args.pocket_emb_path
    if args.base_checkpoint:
        cfg.model.base_checkpoint_path = args.base_checkpoint
    if args.seed_smiles_csv:
        cfg.data.seed_smiles_csv = args.seed_smiles_csv
    if args.seed_smiles_column:
        cfg.data.seed_smiles_column = args.seed_smiles_column


def _build_prompt_dataset(cfg, stoi, adapter) -> PromptDataset:
    csv_path = str(cfg.data.get("seed_smiles_csv", cfg.data.get("smiles_csv", "")))
    column = str(cfg.data.get("seed_smiles_column", cfg.data.get("smiles_column", "")))
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"No existe data.smiles_csv: {csv_path}")
    smiles_df = pd.read_csv(csv_path)
    if column not in smiles_df.columns:
        raise KeyError(f"El CSV semilla no contiene la columna '{column}'")
    smiles = smiles_df[column].dropna().astype(str).str.strip().tolist()
    dataset = PromptDataset(smiles, PromptBuilder(stoi, adapter))
    if len(dataset) == 0:
        raise ValueError("No se pudo construir ningún prompt desde el CSV semilla")
    return dataset


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    _apply_overrides(cfg, args)

    num_ligands = int(
        args.num_ligands
        if args.num_ligands is not None
        else cfg.generation.get("num_ligands", 0)
    )
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else cfg.generation.get("batch_size", 64)
    )
    max_attempt_multiplier = int(
        args.max_attempt_multiplier
        if args.max_attempt_multiplier is not None
        else cfg.generation.get("max_attempt_multiplier", 10)
    )
    checkpoint_path = str(
        args.checkpoint
        or cfg.model.get("ppo_checkpoint_path", "")
    ).strip()
    base_checkpoint_path = str(
        cfg.model.get("base_checkpoint_path", cfg.model.get("checkpoint_path", ""))
    ).strip()
    output_path_raw = str(
        args.output
        or cfg.get("output", {}).get("csv_path", "generated_ligands.csv")
    ).strip()

    if num_ligands <= 0:
        raise ValueError("--num-ligands debe ser mayor que 0")
    if batch_size <= 0:
        raise ValueError("--batch-size debe ser mayor que 0")
    if max_attempt_multiplier <= 0:
        raise ValueError("--max-attempt-multiplier debe ser mayor que 0")
    if not checkpoint_path:
        raise ValueError("Falta model.ppo_checkpoint_path o --checkpoint")
    if not base_checkpoint_path:
        raise ValueError("Falta model.base_checkpoint_path o --base-checkpoint")
    if not output_path_raw:
        raise ValueError("Falta output.csv_path o --output")

    validate_conditioning_assets(cfg)

    seed = int(
        args.seed
        if args.seed is not None
        else cfg.get("system", {}).get("seed", cfg.get("seed", 42))
    )
    _set_seed(seed)
    requested_device = args.device or str(cfg.system.get("device", "auto"))
    device = _resolve_device(requested_device)
    dtype_name = args.dtype or str(cfg.system.dtype)
    if device == "cpu" and dtype_name != "float32":
        print(f"Aviso: usando float32 en CPU en lugar de {dtype_name}")
        dtype_name = "float32"
    dtype = get_torch_dtype(dtype_name)

    stoi, itos = _load_tokenizer(str(cfg.tokenizer.meta_path))
    adapter = build_model_adapter(cfg)
    prompt_dataset = _build_prompt_dataset(cfg, stoi, adapter)

    # PPOAgent necesita el checkpoint base para reconstruir GPT; después se
    # sustituyen sus parámetros por los aprendidos en PPO.
    agent = PPOAgent(
        cfg=cfg,
        adapter=adapter,
        stoi=stoi,
        model_checkpoint_path=base_checkpoint_path,
        trainable=True,
        device=device,
        dtype=dtype,
    )
    _load_ppo_weights(agent, checkpoint_path, device)
    agent.eval()
    agent.requires_grad_(False)

    rng = np.random.default_rng(seed)
    smiles_out: List[str] = []
    attempts = 0
    max_attempts = max(num_ligands, num_ligands * max_attempt_multiplier)

    progress = tqdm(total=num_ligands, desc="Ligandos válidos")
    with torch.inference_mode():
        while len(smiles_out) < num_ligands and attempts < max_attempts:
            remaining_attempts = max_attempts - attempts
            current_batch_size = min(batch_size, remaining_attempts)
            indices = rng.integers(0, len(prompt_dataset), size=current_batch_size)
            prompts = torch.tensor(
                [prompt_dataset[int(i)] for i in indices],
                dtype=torch.long,
                device=device,
            )

            trajectories = agent.generate(prompts, epoch=None)
            attempts += trajectories.shape[0]
            for trajectory in trajectories.detach().cpu().tolist():
                smiles = _decode_generated(trajectory, itos)
                if smiles is None:
                    continue
                smiles_out.append(smiles)
                progress.update(1)
                if len(smiles_out) == num_ligands:
                    break
    progress.close()

    if len(smiles_out) < num_ligands:
        raise RuntimeError(
            f"Solo se obtuvieron {len(smiles_out)} ligandos válidos tras "
            f"{attempts} intentos. Aumenta --max-attempt-multiplier."
        )

    output_path = Path(output_path_raw)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"SMILES": smiles_out}).to_csv(output_path, index=False)
    print(f"Guardados {len(smiles_out)} ligandos en {output_path.resolve()}")


if __name__ == "__main__":
    main()
