import pandas as pd

def load_chembl_smiles(files):
    smiles = []

    for f in files:
        df = pd.read_csv(f)
        smiles.extend(df["__smiles_clean__"].dropna().tolist())

    return list(set(smiles))