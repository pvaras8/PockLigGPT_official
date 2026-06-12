from pathlib import Path
from typing import Mapping


def _nonempty_text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def validate_conditioning_assets(cfg: Mapping) -> None:
    conditioning_cfg = cfg.get("conditioning", {})
    pocket_str = _nonempty_text(conditioning_cfg.get("pocket_str", ""))
    pocket_str_path = _nonempty_text(
        conditioning_cfg.get("pocket_str_path", "")
    )

    if not pocket_str and not pocket_str_path:
        raise ValueError(
            "Falta conditioning.pocket_str o conditioning.pocket_str_path"
        )

    if not pocket_str:
        path = Path(pocket_str_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"No existe el archivo del pocket: {pocket_str_path}"
            )
        if not path.read_text(encoding="utf-8").strip():
            raise ValueError(f"El archivo del pocket está vacío: {pocket_str_path}")

    pocket_emb_path = _nonempty_text(
        conditioning_cfg.get("pocket_emb_path", "")
    )
    if not pocket_emb_path:
        raise ValueError("Falta conditioning.pocket_emb_path")

    path = Path(pocket_emb_path)
    if path.suffix.lower() != ".npy":
        raise ValueError(
            f"El embedding del pocket debe ser un archivo .npy: {pocket_emb_path}"
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"No existe el embedding del pocket: {pocket_emb_path}"
        )
