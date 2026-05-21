import pandas as pd
import selfies as sf

def keep_before_dot(smiles: str) -> str:
    if not isinstance(smiles, str):
        return smiles
    return smiles.split('.', 1)[0].strip()

def smiles_to_selfies(smiles_list):
    valid_smiles = []
    selfies = []

    for smiles in smiles_list:
        try:
            if pd.notna(smiles):
                smi_main = keep_before_dot(smiles.strip())
                if not smi_main:
                    continue
                selfie = sf.encoder(smi_main)
                valid_smiles.append(smi_main)
                selfies.append(selfie)
        except Exception:
            continue

    return pd.DataFrame({'smiles': valid_smiles, 'selfies': selfies})

def split_pocket_smiles(df):
    pockets, smiles = [], []
    for entry in df["pocket_smiles"].dropna().tolist():
        if "_" in entry:
            p, s = entry.split("_", 1)
            pockets.append(p.strip())
            smiles.append(s.strip())
    return pd.DataFrame({"pocket": pockets, "smiles": smiles})