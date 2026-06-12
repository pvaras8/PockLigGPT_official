---
pretty_name: PockLigGPT Training Data
tags:
  - chemistry
  - drug-discovery
  - molecular-generation
  - protein-ligand
  - crossdocked
---

# PockLigGPT Training Data

Preprocessed training assets for
[PockLigGPT](https://github.com/pablovaras/PockLigGPT_official).

## Files

```text
chembl/
├── train_chembl.bin
└── val_chembl.bin

crossdocked/
├── crossdocked_clean_pocket_selfies_with_tokens.parquet
├── per_residue_index.parquet
└── per_residue_pack.npz.part-*

tokenizer/
└── meta_chembl_db_aa_2_proto4.pkl
```

The CrossDocked training Parquet contains pocket metadata, SELFIES,
fixed-length `token_ids`, and offsets into the residue embedding stack.

The split NPZ contains `emb_stack` with shape `(8_337_780, 1024)` and dtype
`float16`. Each row is a ProtT5 residue embedding. The index Parquet maps each
pocket to its `start` and `length` values.

## Download

```bash
hf download pablovp8/PockLigGPT-training-data \
  --repo-type dataset \
  --include "chembl/*" \
  --include "crossdocked/*" \
  --local-dir datasets/processed

hf download pablovp8/PockLigGPT-training-data \
  --repo-type dataset \
  --include "tokenizer/*" \
  --local-dir datasets
```

## Prepare CrossDocked embeddings

Reassemble the downloaded archive:

```bash
python scripts/assemble_embedding_pack.py \
  --parts-dir datasets/processed/crossdocked \
  --output datasets/processed/crossdocked/per_residue_pack.npz \
  --delete-parts
```

Then convert it once to an NPY file before finetuning so NumPy can memory-map
the embedding stack:

```bash
python scripts/convert_embedding_pack.py \
  --input datasets/crossdocked/per_residue_pack.npz \
  --output datasets/crossdocked/per_residue_pack.npy \
  --delete-source
```

When downloading directly into the PockLigGPT repository, place the files
under `datasets/processed/crossdocked/` or update the paths in
`config/training/finetune_2/crossdocked_sequence_add.yaml`.

## Training stages

- ZINC20 binary files are used for pretraining.
- ChEMBL binary files are used for finetune 1. They contain 1,061,229
  training sequences and 117,916 validation sequences with block size 156.
- CrossDocked Parquet and ProtT5 embeddings are used for finetune 2.

ZINC20 binaries are not distributed here because of their size. Generate them
from the source dataset with `config/tokenization/zinc20.yaml`.

## Notes

The tokenizer metadata is a Python pickle file. Only load pickle files from
sources you trust.

Users are responsible for complying with the licenses and terms of the
upstream molecular and structural datasets.
