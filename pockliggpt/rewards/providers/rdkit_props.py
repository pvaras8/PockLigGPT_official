from typing import List

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from .base import RewardProvider


def calculate_logp(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return float(Descriptors.MolLogP(mol))
    return 0.0


def calculate_weight(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return float(Descriptors.MolWt(mol))
    return 0.0


class RDKitPropsProvider(RewardProvider):
    def compute(self, smiles_list: List[str], epoch: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "input_idx": np.arange(len(smiles_list)),
                "SMILES": smiles_list,
                "LogP": [calculate_logp(s) for s in smiles_list],
                "MW": [calculate_weight(s) for s in smiles_list],
            }
        )