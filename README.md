# PockLigGPT

PockLigGPT is a pocket-conditioned molecular generation framework based on GPT architectures and reinforcement learning (RL) for structure-based drug design.

---

## 🌐 Online access

👉 https://pockliggpt.streamlit.app

Researchers and industry partners can submit target proteins (PDB) and request molecule generation or full computational studies.

---

## 🔗 Model Weights

👉 https://huggingface.co/pablovp8/PockLigGPT

Pretrained and fine-tuned checkpoints are available for direct use.

---

## 🚀 Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## ⚡ Quickstart (RL)

```bash
python3 scripts/run_rl_pipeline.py \
  --pdbqt-path ./pdbs/4yhj.pdbqt \
  --center "32.0,28.0,36.0"
```

---

## 🧭 Supported Workflows

PockLigGPT supports three main workflows:

1. **Full training pipeline**
   Pretraining → Finetuning → RL

2. **Use pretrained checkpoints + RL**

3. **Pocket-conditioned RL with real protein inputs**

---

## 📂 Workflow

### 1) Prepare datasets

Datasets are **not included**.

```bash
datasets/raw/
```

Typical datasets:

* ChEMBL
* ZINC20
* CrossDocked

---

### 2) Tokenization

```bash
python scripts/tokenize_dataset.py --config config/tokenization/chembl.yaml
python scripts/tokenize_dataset.py --config config/tokenization/zinc20.yaml
python scripts/tokenize_dataset.py --config config/tokenization/crossdocked.yaml
```

---

### 3) Training

```bash
python scripts/train.py --config config/training/finetune_2/crossdocked_sequence_add.yaml
```

---

### 4) Load pretrained checkpoints

```bash
python -m huggingface_hub download pablovp8/PockLigGPT \
  --repo-type model \
  --local-dir checkpoints/pockliggpt
```

---

### 5) Reinforcement Learning (RL)

Use `run_rl_pipeline.py` as the main entry point for RL:

```bash
python3 scripts/run_rl_pipeline.py \
  --pdbqt-path ./pdbs/4yhj.pdbqt \
  --center "32.0,28.0,36.0"
```

The pipeline:

* validates the receptor `.pdbqt` file and docking center
* updates the docking configuration
* creates the experiment output directories
* launches PPO training using `config/rl/sequence_add.yaml`

To use a different RL configuration:

```bash
python3 scripts/run_rl_pipeline.py \
  --pdbqt-path ./pdbs/4yhj.pdbqt \
  --center "32.0,28.0,36.0" \
  --config config/rl/sequence_add.yaml
```

Additional training options can be forwarded after `--`:

```bash
python3 scripts/run_rl_pipeline.py \
  --pdbqt-path ./pdbs/4yhj.pdbqt \
  --center "32.0,28.0,36.0" \
  -- --no-prompt
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
* datasets available
* tokenizer `.pkl` exists
* embeddings `.npy` generated
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
