# Processed datasets

Generated CrossDocked finetuning assets belong in:

```text
datasets/processed/crossdocked/
├── crossdocked_clean_pocket_selfies_with_tokens.parquet
├── per_residue_index.parquet
└── per_residue_pack.npy
```

The training Parquet stores `token_ids`, `start`, `length`, and `seq_pocket`.
The NPY stores the two-dimensional residue embedding stack and supports true
memory-mapped loading. The loader also accepts legacy NPZ packs containing an
array named `emb_stack`, but NPZ files cannot be memory-mapped efficiently.

These generated files are intentionally ignored by Git. The embedding pack is
much larger than GitHub's regular file limit; publish it through a dataset host
such as Hugging Face Hub, or regenerate it with
`notebooks/prott5_crossdocked_embeddings_en.ipynb`.

The notebook creates `per_residue_index.parquet` and
`per_residue_pack.npy`. The final tokenized Parquet is then generated with:

```bash
python scripts/tokenize_dataset.py \
  --config config/tokenization/crossdocked.yaml
```
