from typing import List, Optional

import pandas as pd
import selfies as sf

from .combiners import COMBINERS
from .providers.docking import DockingProvider
from .providers.rdkit_props import RDKitPropsProvider


PROVIDER_MAP = {
    "docking": DockingProvider,
    "rdkit_props": RDKitPropsProvider,
}


class RewardRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reward_cfg = cfg["reward"]

        self.combiner_name = self.reward_cfg["combiner"]
        if self.combiner_name not in COMBINERS:
            raise ValueError(f"Combiner desconocido: {self.combiner_name}")
        self.combiner = COMBINERS[self.combiner_name]

        self.output_prefix = self.reward_cfg.get("results_output_prefix", "reward_results")

        provider_names = self.reward_cfg.get("providers", [])
        if not provider_names:
            raise ValueError("reward.providers no puede estar vacío")

        self.providers = []
        for provider_name in provider_names:
            if provider_name not in PROVIDER_MAP:
                raise ValueError(f"Provider desconocido: {provider_name}")

            provider_cls = PROVIDER_MAP[provider_name]
            provider_cfg = self.reward_cfg.get(provider_name, {})
            self.providers.append(provider_cls(cfg, provider_cfg))

    def _decode_selfies(self, molecules_selfies: List[str]) -> List[Optional[str]]:
        molecules_smiles = []
        for selfie in molecules_selfies:
            try:
                smiles = sf.decoder(selfie)
                if smiles is None or not isinstance(smiles, str) or len(smiles.strip()) == 0:
                    smiles = None
                molecules_smiles.append(smiles)
            except Exception:
                molecules_smiles.append(None)
        return molecules_smiles

    def _merge_provider_outputs(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        if len(dfs) == 0:
            raise ValueError("No hay DataFrames de providers para mergear")

        merged = dfs[0].copy()
        for df in dfs[1:]:
            merged = merged.merge(df, on=["input_idx", "SMILES"], how="inner")
        return merged

    def _apply_combiner(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.combiner_name == "docking_only":
            if "Docking" not in df.columns:
                raise ValueError("Falta columna 'Docking'")
            df["Fitness"] = df["Docking"].apply(self.combiner)

        elif self.combiner_name == "docking_logp":
            if "Docking" not in df.columns or "LogP" not in df.columns:
                raise ValueError("Faltan columnas 'Docking' y/o 'LogP'")
            df["Fitness"] = df.apply(
                lambda row: self.combiner(row["Docking"], row["LogP"]),
                axis=1,
            )

        elif self.combiner_name == "mw_only":
            if "MW" not in df.columns:
                raise ValueError("Falta columna 'MW'")
            df["Fitness"] = df["MW"].apply(self.combiner)

        else:
            raise NotImplementedError(f"Combiner '{self.combiner_name}' no implementado")

        return df

    def _save_epoch_results(self, df_temp: pd.DataFrame, epoch: int) -> None:
        output_file = f"{self.output_prefix}_{epoch}.csv"
        df_temp.to_csv(output_file, index=False)

    def __call__(self, molecules_selfies: List[str], epoch: int) -> List[float]:
        molecules_smiles = self._decode_selfies(molecules_selfies)

        if len(molecules_smiles) == 0:
            return []

        valid_pairs = [
            (idx, smi) for idx, smi in enumerate(molecules_smiles) if smi is not None
        ]

        if len(valid_pairs) == 0:
            return [0.0] * len(molecules_selfies)

        valid_indices = [idx for idx, _ in valid_pairs]
        valid_smiles = [smi for _, smi in valid_pairs]

        try:
            provider_dfs = [provider.compute(valid_smiles, epoch) for provider in self.providers]
            df_temp = self._merge_provider_outputs(provider_dfs)
            df_temp = self._apply_combiner(df_temp)
            self._save_epoch_results(df_temp, epoch)

            valid_rewards = (
                df_temp.sort_values("input_idx")["Fitness"]
                .fillna(0.0)
                .astype(float)
                .tolist()
            )

            if len(valid_rewards) != len(valid_smiles):
                raise RuntimeError(
                    f"Longitud de rewards válidas incorrecta: {len(valid_rewards)} vs {len(valid_smiles)}"
                )

            rewards = [0.0] * len(molecules_selfies)
            for original_idx, reward in zip(valid_indices, valid_rewards):
                rewards[original_idx] = float(reward)

            return rewards

        except Exception as e:
            print(f"Error durante reward_fn: {e}")
            return [0.0] * len(molecules_selfies)