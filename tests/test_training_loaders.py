import pickle

import numpy as np
import pandas as pd
import pytest

from pockliggpt.training.loaders import SequenceAddLoader, SequenceLoader


def test_sequence_loader_reads_flat_binary_datasets(tmp_path):
    block_size = 4
    train_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    meta_path = tmp_path / "meta.pkl"

    np.arange(12, dtype=np.uint16).tofile(train_path)
    np.arange(8, dtype=np.uint16).tofile(val_path)
    with meta_path.open("wb") as handle:
        pickle.dump({"vocab_size": 32, "stoi": {}}, handle)

    cfg = {
        "data": {
            "batch_size": 2,
            "block_size": block_size,
            "meta_path": str(meta_path),
            "train_bin": str(train_path),
            "val_bin": str(val_path),
            "pad_token_id": 0,
        }
    }

    loader = SequenceLoader(cfg, device="cpu", device_type="cpu")
    batch = loader.get_batch("train")

    assert loader.get_vocab_size() == 32
    assert batch["idx"].shape == (2, block_size)
    assert batch["targets"].shape == (2, block_size)
    assert batch["padding_mask"].shape == (2, block_size)


def _write_sequence_add_data(tmp_path, rows=None, emb_stack=None):
    parquet_path = tmp_path / "dataset.parquet"
    embeddings_path = tmp_path / "pockets.npy"
    meta_path = tmp_path / "meta.pkl"

    if rows is None:
        rows = [
            {
                "token_ids": [1, 2, 3, 4, 0, 0],
                "start": 0,
                "length": 3,
                "seq_pocket": "AC",
            },
            {
                "token_ids": [1, 2, 5, 6, 7, 0],
                "start": 3,
                "length": 4,
                "seq_pocket": "DEF",
            },
        ]
    if emb_stack is None:
        emb_stack = np.arange(28, dtype=np.float16).reshape(7, 4)

    pd.DataFrame(rows).to_parquet(parquet_path)
    np.save(embeddings_path, emb_stack)
    with meta_path.open("wb") as handle:
        pickle.dump({"vocab_size": 32}, handle)

    cfg = {
        "data": {
            "batch_size": 2,
            "block_size": 6,
            "meta_path": str(meta_path),
            "parquet_path": str(parquet_path),
            "pocket_embeddings_path": str(embeddings_path),
            "pad_token_id": 0,
            "aa_start": 2,
            "pocket_embedding_dim": 4,
            "val_fraction": 0.5,
            "split_seed": 42,
        }
    }
    return cfg


def test_sequence_add_loader_aligns_residue_embeddings(tmp_path):
    cfg = _write_sequence_add_data(tmp_path)
    loader = SequenceAddLoader(cfg, device="cpu", device_type="cpu")
    batch = loader.get_batch("train")

    assert batch["idx"].shape == (2, 6)
    assert batch["targets"].shape == (2, 6)
    assert batch["pocket_emb"].shape == (2, 6, 4)
    assert batch["pocket_emb"][:, :2].count_nonzero() == 0
    assert torch_all_last_targets_are_padding(batch["targets"])


def torch_all_last_targets_are_padding(targets):
    return bool((targets[:, -1] == 0).all())


def test_sequence_add_loader_rejects_misaligned_index(tmp_path):
    rows = [
        {
            "token_ids": [1, 2, 3, 4, 0, 0],
            "start": 0,
            "length": 4,
            "seq_pocket": "AC",
        },
        {
            "token_ids": [1, 2, 5, 6, 7, 0],
            "start": 2,
            "length": 3,
            "seq_pocket": "DEF",
        },
    ]
    cfg = _write_sequence_add_data(tmp_path, rows=rows)

    with pytest.raises(ValueError, match="length mismatch"):
        SequenceAddLoader(cfg, device="cpu", device_type="cpu")
