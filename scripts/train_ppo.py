# scripts/train_ppo.py
import argparse
from datetime import datetime
import json
import os
import sys

from omegaconf import OmegaConf

from pockliggpt.rl.model_adapters import build_model_adapter
from pockliggpt.rl.trainer import run_ppo_training


DEFAULT_DOCKING_VARS_FILE = "config/docking/vars_meeko.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO model for pocket-conditioned molecular generation"
    )
    parser.add_argument("--config", type=str, required=True)

    parser.add_argument("--pdbqt-path", type=str, default="", help="Ruta al receptor PDBQT")
    parser.add_argument("--center", type=str, default="", help='Centro docking como "x,y,z"')

    parser.add_argument(
        "--use-pocket-context",
        choices=["auto", "on", "off"],
        default="auto",
        help="Usar condicionamiento textual de pocket",
    )
    parser.add_argument(
        "--use-pocket-embedding",
        choices=["auto", "on", "off"],
        default="auto",
        help="Usar embedding ProtT5 de pocket",
    )
    parser.add_argument("--pocket-str-path", type=str, default="", help="Ruta a pocket_str.txt")
    parser.add_argument("--pocket-emb-path", type=str, default="", help="Ruta a pocket_emb.npy")

    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="No solicita confirmacion interactiva y continua automaticamente",
    )
    return parser.parse_args()


def _ensure_section(cfg, section: str):
    if section not in cfg or cfg[section] is None:
        cfg[section] = {}
    return cfg[section]


def _parse_center(center_raw: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in center_raw.split(",")]
    if len(parts) != 3:
        raise ValueError("--center debe tener formato 'x,y,z'")
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError as exc:
        raise ValueError("--center debe contener solo valores numericos") from exc


def _is_nonempty_text(value) -> bool:
    return isinstance(value, str) and len(value.strip()) > 0


def _apply_auto_on_off(current: bool, mode: str) -> bool:
    if mode == "on":
        return True
    if mode == "off":
        return False
    return current


def _apply_cli_conditioning_overrides(cfg, args: argparse.Namespace) -> None:
    prompt_cfg = _ensure_section(cfg, "prompt")
    cond_cfg = _ensure_section(cfg, "conditioning")

    prompt_cfg["include_pocket_tokens"] = _apply_auto_on_off(
        bool(prompt_cfg.get("include_pocket_tokens", True)),
        args.use_pocket_context,
    )
    prompt_cfg["include_pocket_embedding"] = _apply_auto_on_off(
        bool(prompt_cfg.get("include_pocket_embedding", True)),
        args.use_pocket_embedding,
    )

    if _is_nonempty_text(args.pocket_str_path):
        cond_cfg["pocket_str_path"] = args.pocket_str_path.strip()
    if _is_nonempty_text(args.pocket_emb_path):
        cond_cfg["pocket_emb_path"] = args.pocket_emb_path.strip()


def _apply_cli_docking_overrides(cfg, args: argparse.Namespace) -> None:
    pdbqt_path = args.pdbqt_path.strip()
    center_raw = args.center.strip()

    if not pdbqt_path and not center_raw:
        return
    if not pdbqt_path or not center_raw:
        raise ValueError("Debes indicar ambos --pdbqt-path y --center")

    receptor_abs = os.path.abspath(pdbqt_path)
    if not receptor_abs.lower().endswith(".pdbqt"):
        raise ValueError("El receptor debe ser .pdbqt")
    if not os.path.exists(receptor_abs):
        raise FileNotFoundError(f"No existe receptor: {receptor_abs}")

    center_x, center_y, center_z = _parse_center(center_raw)

    vars_file = (
        cfg.get("reward", {})
        .get("docking", {})
        .get("vars_file", DEFAULT_DOCKING_VARS_FILE)
    )
    vars_path = os.path.abspath(str(vars_file))
    if not os.path.exists(vars_path):
        raise FileNotFoundError(f"No existe vars_file de docking: {vars_path}")

    with open(vars_path, "r", encoding="utf-8") as handle:
        vars_cfg = json.load(handle)

    vars_cfg["filename_of_receptor"] = receptor_abs
    vars_cfg["center_x"] = center_x
    vars_cfg["center_y"] = center_y
    vars_cfg["center_z"] = center_z

    with open(vars_path, "w", encoding="utf-8") as handle:
        json.dump(vars_cfg, handle, indent=2)
        handle.write("\n")

    print("Docking actualizado desde CLI:")
    print(f"- vars_file: {vars_path}")
    print(f"- filename_of_receptor: {receptor_abs}")
    print(f"- center: {center_x}, {center_y}, {center_z}")


def _build_experiment_layout(cfg) -> None:
    reward_cfg = _ensure_section(cfg, "reward")
    docking_cfg = _ensure_section(reward_cfg, "docking")

    vars_file = docking_cfg.get("vars_file", DEFAULT_DOCKING_VARS_FILE)
    vars_path = os.path.abspath(str(vars_file))
    if not os.path.exists(vars_path):
        raise FileNotFoundError(f"No existe vars_file de docking: {vars_path}")

    with open(vars_path, "r", encoding="utf-8") as handle:
        vars_cfg = json.load(handle)

    receptor_path = os.path.abspath(str(vars_cfg.get("filename_of_receptor", "")))
    receptor_name = os.path.splitext(os.path.basename(receptor_path))[0] or "unknown_receptor"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = os.path.abspath(os.path.join("experiments", "rl", receptor_name, timestamp))
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    logs_dir = os.path.join(run_dir, "logs")
    reward_dir = os.path.join(run_dir, "reward")
    docking_dir = os.path.join(run_dir, "docking")
    trajectories_dir = os.path.join(run_dir, "trajectories")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(reward_dir, exist_ok=True)
    os.makedirs(docking_dir, exist_ok=True)
    os.makedirs(trajectories_dir, exist_ok=True)

    output_cfg = _ensure_section(cfg, "output")
    default_ckpt_name = os.path.basename(str(output_cfg.get("ppo_ckpt_path", "ppo_best.pt")))
    default_loss_name = os.path.basename(
        str(output_cfg.get("loss_history_file", "loss_history.txt"))
    )

    output_cfg["ppo_ckpt_path"] = os.path.join(ckpt_dir, default_ckpt_name)
    output_cfg["loss_history_file"] = os.path.join(logs_dir, default_loss_name)
    output_cfg["trajectories_dir"] = trajectories_dir

    reward_cfg["results_output_prefix"] = os.path.join(reward_dir, "reward_results")
    docking_cfg["smiles_output_file"] = os.path.join(run_dir, "smiles_input.smi")

    vars_cfg["final_folder"] = docking_dir
    with open(vars_path, "w", encoding="utf-8") as handle:
        json.dump(vars_cfg, handle, indent=2)
        handle.write("\n")

    print("Experimento RL organizado en:")
    print(f"- run_dir: {run_dir}")
    print(f"- checkpoint: {output_cfg['ppo_ckpt_path']}")
    print(f"- loss_history: {output_cfg['loss_history_file']}")
    print(f"- trajectories: {output_cfg['trajectories_dir']}")
    print(f"- reward_prefix: {reward_cfg['results_output_prefix']}")
    print(f"- docking_final_folder: {docking_dir}")


def _has_any_conditioning_input(cfg) -> bool:
    conditioning = cfg.get("conditioning", {})
    has_text = _is_nonempty_text(conditioning.get("pocket_str", "")) or _is_nonempty_text(
        conditioning.get("pocket_str_path", "")
    )
    has_embedding = _is_nonempty_text(conditioning.get("pocket_emb_path", ""))
    return has_text or has_embedding


def _handle_missing_conditioning(cfg, no_prompt: bool) -> None:
    prompt_cfg = cfg.get("prompt", {})
    include_tokens = bool(prompt_cfg.get("include_pocket_tokens", True))
    include_embedding = bool(prompt_cfg.get("include_pocket_embedding", True))

    # If both conditioning channels are disabled intentionally, skip warning.
    if not include_tokens and not include_embedding:
        return
    if _has_any_conditioning_input(cfg):
        return

    print(
        "No has introducido condicionamiento textual ni tampoco embedding de ProtT5, desea continuar?"
    )

    should_continue = True
    if not no_prompt and sys.stdin.isatty():
        reply = input("Continuar entrenamiento RL sin condicionamiento? [Y/n]: ").strip().lower()
        if reply in {"n", "no"}:
            should_continue = False

    if not should_continue:
        raise SystemExit("Entrenamiento cancelado por el usuario")

    prompt_cfg = _ensure_section(cfg, "prompt")
    prompt_cfg["include_pocket_tokens"] = False
    prompt_cfg["include_pocket_embedding"] = False
    print("Continuando RL sin condicionamiento de pocket.")


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    _apply_cli_conditioning_overrides(cfg, args)
    _apply_cli_docking_overrides(cfg, args)
    _build_experiment_layout(cfg)
    _handle_missing_conditioning(cfg, no_prompt=args.no_prompt)

    adapter = build_model_adapter(cfg)
    run_ppo_training(cfg, adapter)


if __name__ == "__main__":
    main()