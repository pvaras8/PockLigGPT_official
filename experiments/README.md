# Experiments

This folder contains training outputs (checkpoints, logs, etc.).

## Structure

- `pretrain/` → pretraining runs
- `finetune/` → fine-tuning runs
- `rl/` → reinforcement learning runs

Each experiment should have its own folder:

experiments/
  pretrain/
    zinc20_sequence_bs32_lr3e4/

## Notes

- This folder is **ignored by git** (see `.gitignore`)
- No checkpoints are tracked in the repository
- Config files used for experiments are stored in `config/`

## Reproducibility

To reproduce an experiment:

1. Use the corresponding YAML config in `config/training/`
2. Run:

```bash
python scripts/train.py --config <config.yaml>