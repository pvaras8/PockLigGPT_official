from pathlib import Path

import pandas as pd

from pockliggpt.data.tokenizers.preprocessing import split_pocket_smiles


def load_crossdocked(file):
    path = Path(file)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        required = {"pocket_id", "start", "length", "seq_pocket", "smiles"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"CrossDocked index is missing columns: {sorted(missing)}"
            )
        df = df.copy()
        df["pocket"] = df["seq_pocket"]
        return df

    df = pd.read_csv(path)
    if "pocket_smiles" not in df.columns:
        raise ValueError("CrossDocked CSV must contain 'pocket_smiles'")

    return split_pocket_smiles(df)
