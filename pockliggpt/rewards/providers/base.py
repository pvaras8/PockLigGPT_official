from typing import List
import pandas as pd


class RewardProvider:
    def __init__(self, provider_cfg=None):
        self.provider_cfg = provider_cfg or {}

    def compute(self, smiles_list: List[str], epoch: int) -> pd.DataFrame:
        raise NotImplementedError
