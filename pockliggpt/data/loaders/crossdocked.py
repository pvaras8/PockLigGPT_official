import pandas as pd
from pockliggpt.data.tokenizers.preprocessing import split_pocket_smiles

def load_crossdocked(file):

    df = pd.read_csv(file)

    if "pocket_smiles" not in df.columns:
        raise ValueError("Missing pocket_smiles")

    return split_pocket_smiles(df)