# Datasets

This folder contains all data-related assets used in PockLigGPT.

The structure is divided into three main parts:

- `raw/`: original datasets (e.g., ChEMBL, CrossDocked, ZINC20)
- `processed/`: tokenized datasets used for training (`.bin`)
- `tokenizers/`: vocabulary files (`meta.pkl`) used by SELFIES tokenizers

⚠️ Note: Data files are not included in this repository due to size constraints.
They must be generated or downloaded separately.