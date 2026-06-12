#!/usr/bin/env python3
"""Backward-compatible entry point for the pocket-conditioned PPO pipeline."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    runpy.run_path(str(ROOT / "scripts" / "run_rl_pipeline.py"), run_name="__main__")
