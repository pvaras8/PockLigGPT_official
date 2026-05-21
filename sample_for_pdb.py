#!/usr/bin/env python3
"""Small helper to configure RL docking inputs for a specific receptor.

This script updates config/docking/vars_meeko.json with:
- filename_of_receptor
- center_x, center_y, center_z

Only .pdbqt receptors are allowed because Meeko/Vina use PDBQT receptors.
"""

import argparse
import json
import os
from typing import Tuple


def parse_center(center_raw: str) -> Tuple[float, float, float]:
    """Parse center string in format 'x,y,z' (spaces allowed)."""
    parts = [p.strip() for p in center_raw.split(",")]
    if len(parts) != 3:
        raise ValueError(
            "Invalid --center format. Expected 'x,y,z', for example: 32.0,28.0,36.0"
        )

    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError as exc:
        raise ValueError("--center values must be numeric") from exc


def validate_pdbqt_path(pdbqt_path: str) -> str:
    """Validate receptor path exists and has .pdbqt extension."""
    abs_path = os.path.abspath(pdbqt_path)

    if not abs_path.lower().endswith(".pdbqt"):
        raise ValueError(
            "Only .pdbqt receptors are supported (Meeko/Vina requirement)."
        )

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Receptor file not found: {abs_path}")

    return abs_path


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set receptor PDBQT path and docking box center for RL setup."
    )
    parser.add_argument(
        "--pdbqt_path",
        required=True,
        help="Path to receptor .pdbqt file",
    )
    parser.add_argument(
        "--center",
        required=True,
        help="Docking center as 'x,y,z' (spaces allowed)",
    )
    parser.add_argument(
        "--vars_json",
        default="config/docking/vars_meeko.json",
        help="Path to vars_meeko.json (default: config/docking/vars_meeko.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    vars_json_path = os.path.abspath(args.vars_json)
    if not os.path.exists(vars_json_path):
        raise FileNotFoundError(f"Config file not found: {vars_json_path}")

    receptor_pdbqt = validate_pdbqt_path(args.pdbqt_path)
    center_x, center_y, center_z = parse_center(args.center)

    cfg = load_json(vars_json_path)
    cfg["filename_of_receptor"] = receptor_pdbqt
    cfg["center_x"] = center_x
    cfg["center_y"] = center_y
    cfg["center_z"] = center_z
    save_json(vars_json_path, cfg)

    print("Updated docking config:")
    print(f"- vars_json: {vars_json_path}")
    print(f"- filename_of_receptor: {receptor_pdbqt}")
    print(f"- center: {center_x}, {center_y}, {center_z}")
    print("")
    print("Now run RL training with:")
    print("python scripts/train_ppo.py --config config/rl/sequence_add.yaml")


if __name__ == "__main__":
    main()
