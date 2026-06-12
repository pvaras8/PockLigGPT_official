# Datasets

This folder contains all data-related assets used in PockLigGPT.

The structure is divided into three main parts:

- `raw/`: original datasets (for example ChEMBL, CrossDocked, and ZINC20)
- `processed/`: generated training datasets and embedding indexes
- `tokenizer/`: vocabulary files used by the molecular tokenizers

Large data files are not committed to Git. Generate them with the preprocessing
notebooks/scripts or distribute them through an external dataset host.
