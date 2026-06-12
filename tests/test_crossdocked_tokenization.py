import pickle

import pandas as pd

from pockliggpt.data.tokenizers.preprocessing import smiles_to_selfies
from scripts.tokenize_dataset import tokenize_crossdocked_dataset


def _write_tokenizer_meta(path):
    tokens = [
        "<PAD>",
        "<SOS>",
        "<POCKET>",
        "</POCKET>",
        "<LIGAND>",
        "</LIGAND>",
        "<EOS>",
        "<UNK>",
        "<AA_ALA>",
        "<AA_CYS>",
        "[C]",
    ]
    stoi = {token: index for index, token in enumerate(tokens)}
    itos = {index: token for token, index in stoi.items()}
    with path.open("wb") as handle:
        pickle.dump({"vocab_size": len(tokens), "stoi": stoi, "itos": itos}, handle)


def test_smiles_to_selfies_preserves_source_indices():
    converted = smiles_to_selfies(["C", None, "invalid smiles", "CC"])

    assert converted.index.tolist() == [0, 3]
    assert converted["smiles"].tolist() == ["C", "CC"]


def test_crossdocked_tokenization_writes_metadata_parquet(tmp_path):
    index_path = tmp_path / "per_residue_index.parquet"
    meta_path = tmp_path / "meta.pkl"
    output_dir = tmp_path / "processed"

    pd.DataFrame(
        [
            {
                "pocket_id": "p_1",
                "start": 0,
                "length": 2,
                "seq_pocket": "AC",
                "smiles": "C",
            },
            {
                "pocket_id": "p_2",
                "start": 2,
                "length": 2,
                "seq_pocket": "AA",
                "smiles": "CC",
            },
        ]
    ).to_parquet(index_path, index=False)
    _write_tokenizer_meta(meta_path)

    config = {
        "dataset": {"type": "crossdocked", "file": str(index_path)},
        "tokenizer": {
            "type": "pocket_ligand",
            "max_length": 12,
            "meta_path": str(meta_path),
        },
        "output": {
            "dir": str(output_dir),
            "file": "crossdocked.parquet",
        },
    }

    tokenize_crossdocked_dataset(config)

    result = pd.read_parquet(output_dir / "crossdocked.parquet")
    assert result["pocket_id"].tolist() == ["p_1", "p_2"]
    assert result["start"].tolist() == [0, 2]
    assert result["seq_pocket"].tolist() == ["AC", "AA"]
    assert result["token_ids"].map(len).tolist() == [12, 12]
