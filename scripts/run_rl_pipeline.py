#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys


def _fix_negative_option_values(argv: list[str]) -> list[str]:
    """Replace '--opt -value' with '--opt=-value' so argparse doesn't treat
    negative numbers as unknown flags."""
    result = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if (
            token.startswith("--")
            and i + 1 < len(argv)
            and re.match(r"^-\d", argv[i + 1])
        ):
            result.append(f"{token}={argv[i + 1]}")
            i += 2
        else:
            result.append(token)
            i += 1
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end RL flow: set receptor center and start PPO training"
    )
    parser.add_argument("--pdbqt-path", required=True, help="Path to receptor PDBQT")
    parser.add_argument("--center", required=True, help='Docking center as "x,y,z"')
    parser.add_argument(
        "--config",
        default="config/rl/sequence_add.yaml",
        help="PPO config path (default: config/rl/sequence_add.yaml)",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to scripts/train_ppo.py (prefix with --)",
    )
    return parser.parse_args(_fix_negative_option_values(sys.argv[1:]))


def run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def normalize_extra_args(extra_args: list[str]) -> list[str]:
    # argparse.REMAINDER keeps an optional leading '--' separator.
    if extra_args and extra_args[0] == "--":
        return extra_args[1:]
    return extra_args


def main() -> None:
    args = parse_args()
    extra_args = normalize_extra_args(args.extra_args)

    train_cmd = [
        sys.executable,
        "scripts/train_ppo.py",
        "--config",
        args.config,
        "--pdbqt-path",
        args.pdbqt_path,
        "--center",
        args.center,
    ]
    train_cmd.extend(extra_args)
    run_cmd(train_cmd)


if __name__ == "__main__":
    main()
