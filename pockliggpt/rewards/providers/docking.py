import os
import sys
import json
import subprocess
from typing import List

import numpy as np
import pandas as pd

from .base import RewardProvider


class DockingProvider(RewardProvider):
    """
    Wrapper simple para docking vía script externo.
    El script debe escribir un CSV temporal con una columna tipo Docking.
    """

    def __init__(self, cfg, provider_cfg=None):
        super().__init__(cfg, provider_cfg)

        self.script = os.path.abspath(self.provider_cfg["script"])
        self.vars_file = os.path.abspath(self.provider_cfg["vars_file"])
        self.smiles_output_file = os.path.abspath(self.provider_cfg["smiles_output_file"])
        self.output_prefix = self.provider_cfg.get("output_prefix", "docking_results")
        self.output_column = self.provider_cfg.get("output_column", "Docking")
        self.suffix = self.provider_cfg.get("suffix", None)

    def _write_smiles_file(self, smiles_list: List[str]) -> None:
        os.makedirs(os.path.dirname(self.smiles_output_file), exist_ok=True)
        with open(self.smiles_output_file, "w") as f:
            for smi in smiles_list:
                f.write(smi + "\n")

    def _get_output_file(self, epoch: int) -> str:
        with open(self.vars_file, "r") as f:
            vars_cfg = json.load(f)

        base_dir = os.path.abspath(
            vars_cfg.get("final_folder", os.path.dirname(self.smiles_output_file))
        )

        if self.suffix is None:
            return os.path.join(base_dir, f"{self.output_prefix}_{epoch}_temp.csv")
        return os.path.join(base_dir, f"{self.output_prefix}_{self.suffix}_{epoch}_temp.csv")

    def compute(self, smiles_list: List[str], epoch: int) -> pd.DataFrame:
        self._write_smiles_file(smiles_list)

        subprocess.run(
            [sys.executable, self.script, self.smiles_output_file, self.vars_file, str(epoch)],
            check=True,
        )

        output_file = self._get_output_file(epoch)

        if not os.path.exists(output_file):
            raise FileNotFoundError(f"No se encontró el archivo temporal: {output_file}")

        df = pd.read_csv(output_file)

        if self.output_column not in df.columns:
            raise ValueError(
                f"El CSV no contiene la columna '{self.output_column}'. "
                f"Columnas disponibles: {list(df.columns)}"
            )

        if len(df) != len(smiles_list):
            raise RuntimeError(
                f"Docking devolvió {len(df)} filas, esperaba {len(smiles_list)}"
            )

        if "SMILES" in df.columns:
            returned_smiles = df["SMILES"].astype(str).tolist()
            expected_smiles = [str(s) for s in smiles_list]
            if returned_smiles != expected_smiles:
                raise RuntimeError(
                    "El docking no devolvió los SMILES en el mismo orden que la entrada"
                )

        return pd.DataFrame(
            {
                "input_idx": np.arange(len(smiles_list)),
                "SMILES": smiles_list,
                self.output_column: df[self.output_column].astype(float).to_numpy(),
            }
        )