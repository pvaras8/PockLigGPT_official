from pathlib import Path

import pytest

from pockliggpt.rl.conditioning import validate_conditioning_assets


def make_config(tmp_path: Path) -> dict:
    pocket_str_path = tmp_path / "pocket_str.txt"
    pocket_str_path.write_text("SER LEU ILE", encoding="utf-8")

    pocket_emb_path = tmp_path / "pocket_emb.npy"
    pocket_emb_path.write_bytes(b"test")

    return {
        "conditioning": {
            "pocket_str_path": str(pocket_str_path),
            "pocket_emb_path": str(pocket_emb_path),
        },
    }


def test_valid_conditioning_assets(tmp_path):
    validate_conditioning_assets(make_config(tmp_path))


def test_missing_pocket_string_configuration(tmp_path):
    cfg = make_config(tmp_path)
    cfg["conditioning"].pop("pocket_str_path")

    with pytest.raises(ValueError, match="pocket_str"):
        validate_conditioning_assets(cfg)


def test_missing_pocket_string_file(tmp_path):
    cfg = make_config(tmp_path)
    cfg["conditioning"]["pocket_str_path"] = str(tmp_path / "missing.txt")

    with pytest.raises(FileNotFoundError, match="archivo del pocket"):
        validate_conditioning_assets(cfg)


def test_empty_pocket_string_file(tmp_path):
    cfg = make_config(tmp_path)
    Path(cfg["conditioning"]["pocket_str_path"]).write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="vacío"):
        validate_conditioning_assets(cfg)


def test_missing_pocket_embedding_configuration(tmp_path):
    cfg = make_config(tmp_path)
    cfg["conditioning"].pop("pocket_emb_path")

    with pytest.raises(ValueError, match="pocket_emb_path"):
        validate_conditioning_assets(cfg)


def test_missing_pocket_embedding_file(tmp_path):
    cfg = make_config(tmp_path)
    cfg["conditioning"]["pocket_emb_path"] = str(tmp_path / "missing.npy")

    with pytest.raises(FileNotFoundError, match="embedding del pocket"):
        validate_conditioning_assets(cfg)


def test_conditioning_cannot_be_disabled(tmp_path):
    cfg = make_config(tmp_path)
    cfg["conditioning"] = {}

    with pytest.raises(ValueError, match="pocket_str"):
        validate_conditioning_assets(cfg)
