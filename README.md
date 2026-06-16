# PockLigGPT

PockLigGPT is a molecular generation framework based on GPT architectures and reinforcement learning (RL) for sequence-based conditioned design.

---

## 🌐 Online access

👉 https://pockliggpt.streamlit.app

Researchers and industry partners can submit target proteins (PDB) and request molecule generation or full computational studies.

---

## 🔗 Model Weights

👉 https://huggingface.co/pablovp8/PockLigGPT

Pretrained and fine-tuned checkpoints are available for direct use.

## 📦 Training Data

👉 https://huggingface.co/datasets/pablovp8/PockLigGPT-training-data

Download the prepared ChEMBL binaries, CrossDocked assets, and tokenizer into
the paths expected by the training configurations:

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

The CrossDocked embedding archive is split for reliable distribution.
Reassemble it and create the memory-mapped NPY before finetune 2:

```bash
python scripts/assemble_embedding_pack.py \
  --parts-dir datasets/processed/crossdocked \
  --output datasets/processed/crossdocked/per_residue_pack.npz \
  --delete-parts

python scripts/convert_embedding_pack.py \
  --input datasets/processed/crossdocked/per_residue_pack.npz \
  --output datasets/processed/crossdocked/per_residue_pack.npy \
  --delete-source
```

---

## 🚀 Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## ⚡ RL command

```bash
python3 scripts/run_rl_pipeline.py \
  --pdbqt-path /path/to/receptor.pdbqt \
  --center "CENTER_X,CENTER_Y,CENTER_Z"
```

---

## 🧭 Supported Workflows

PockLigGPT supports three main workflows:

1. **Full training pipeline**
   Pretraining → Finetuning → RL

2. **Use pretrained checkpoints + RL**

3. **Pocket-conditioned RL with real protein inputs**

---

## 📂 Training Workflow

The complete training sequence is:

```text
ZINC20 pretraining
    ↓ ckpt_zinc20.pt
ChEMBL finetune 1
    ↓ ckpt_zinc20_chembl.pt
CrossDocked finetune 2
    ↓ ckpt_zinc20_chembl_cross_sequence_add.pt
Reinforcement learning
```

This repository uses **ZINC20** for pretraining. ZINC250K is only the small
SMILES dataset used by the default RL workflow.

Large training datasets are not included in Git.

### 1) ZINC20 Pretraining

Place the ZINC20 source files under `datasets/raw/zinc20/`, then create the
fixed-length binary datasets:

```bash
python scripts/tokenize_dataset.py \
  --config config/tokenization/zinc20.yaml
```

Outputs:

```text
datasets/processed/zinc20/train_zinc20.bin
datasets/processed/zinc20/val_zinc20.bin
```

Train from scratch:

```bash
torchrun --standalone --nproc_per_node=4 scripts/train.py \
  --config config/training/pretrain/zinc20_sequence_pretrain.yaml
```

Output checkpoint:

```text
checkpoints/ckpt_zinc20.pt
```

### 2) ChEMBL Finetune 1

Either download the prepared `.bin` files from the training-data repository,
or place the ChEMBL CSV files configured in `config/tokenization/chembl.yaml`
under `datasets/raw/chembl/` and tokenize them:

```bash
python scripts/tokenize_dataset.py \
  --config config/tokenization/chembl.yaml
```

Outputs:

```text
datasets/processed/chembl/train_chembl.bin
datasets/processed/chembl/val_chembl.bin
```

Finetune from the ZINC20 checkpoint:

```bash
torchrun --standalone --nproc_per_node=4 scripts/train.py \
  --config config/training/finetune_1/chembl_sequence_finetune_1.yaml
```

Output checkpoint:

```text
checkpoints/ckpt_zinc20_chembl.pt
```

### 3) CrossDocked Finetune 2

CrossDocked does **not** use `.bin` files. Its preprocessing has two steps.

Either download the prepared assets from the training-data repository, or run
`notebooks/prott5_crossdocked_embeddings_en.ipynb` to generate:

```text
datasets/processed/crossdocked/per_residue_index.parquet
datasets/processed/crossdocked/per_residue_pack.npy
```

If the embeddings are stored in the legacy NPZ format, convert them once:

```bash
python scripts/convert_embedding_pack.py \
  --input datasets/processed/crossdocked/per_residue_pack.npz \
  --output datasets/processed/crossdocked/per_residue_pack.npy
```

Then add SELFIES and `token_ids` to the embedding index:

```bash
python scripts/tokenize_dataset.py \
  --config config/tokenization/crossdocked.yaml
```

Output:

```text
datasets/processed/crossdocked/crossdocked_clean_pocket_selfies_with_tokens.parquet
```

Finetune with pocket residue embeddings:

```bash
torchrun --standalone --nproc_per_node=4 scripts/train.py \
  --config config/training/finetune_2/crossdocked_sequence_add.yaml
```

Output checkpoint:

```text
checkpoints/ckpt_zinc20_chembl_crossdocked.pt
```

### 4) Load Pretrained Checkpoints

```bash
python -m huggingface_hub download pablovp8/PockLigGPT \
  --repo-type model \
  --local-dir checkpoints
```

---

### 5) Reinforcement Learning (RL)

Use `run_rl_pipeline.py` as the main entry point for RL:

```bash
python3 scripts/run_rl_pipeline.py \
  --pdbqt-path /path/to/receptor.pdbqt \
  --center "CENTER_X,CENTER_Y,CENTER_Z"
```

The pipeline:

* validates the receptor `.pdbqt` file and docking center
* updates the docking configuration
* creates the experiment output directories
* launches PPO training using `config/rl/sequence_add.yaml`

To use a different RL configuration:

```bash
python3 scripts/run_rl_pipeline.py \
  --pdbqt-path /path/to/receptor.pdbqt \
  --center "CENTER_X,CENTER_Y,CENTER_Z" \
  --config config/rl/sequence_add.yaml
```

Conditioning assets can be provided without editing the YAML:

```bash
python3 scripts/run_rl_pipeline.py \
  --pdbqt-path /path/to/receptor.pdbqt \
  --center "CENTER_X,CENTER_Y,CENTER_Z" \
  -- \
  --pocket-str-path conditioning/pocket_str.txt \
  --pocket-emb-path conditioning/pocket_emb.npy
```

---

## 🧬 Pocket Embeddings

Generate pocket sequences and ProtT5 embeddings:

```bash
notebooks/prott5_pocket_pipeline_simple_en.ipynb
```

Outputs:

* pocket amino-acid sequence
* ProtT5 residue embeddings (`.npy`)

For batch CrossDocked embedding extraction, use
`notebooks/prott5_crossdocked_embeddings_en.ipynb`. Keep the source CSV under
`datasets/raw/crossdocked/` and generated assets under
`datasets/processed/crossdocked/`.

---

## ⚙️ Docking setup

### Configure Meeko + Vina

Edit:

```bash
config/docking/vars_meeko.json
```

Set these fields for your environment:

* `num_processors`
* `vina_cpu_per_job`
* `exhaustiveness`
* `n_poses`
* `fallback_score`

The receptor path and docking center are set automatically from
`run_rl_pipeline.py`. Configure only the docking box dimensions manually:

* `size_x`, `size_y`, `size_z`

---

## ✅ Minimal checklist

Before running RL with docking reward:

* receptor `.pdbqt` file available
* RL SMILES dataset available
* tokenizer `.pkl` exists
* pocket string and residue embeddings `.npy` generated
* checkpoint path valid
* Meeko + Vina installed
* docking config correctly set

---

## ⚡ Compute Requirements

* Pretraining / finetuning: **4 GPUs**
* Reinforcement Learning: **2 GPUs**

---

## 📜 License

MIT License
