from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import selfies as sf
import torch


def split_selfies(selfies: str) -> List[str]:
    tokens = []
    token = ""
    for char in selfies:
        token += char
        if char == "]":
            tokens.append(token)
            token = ""
    return tokens


def pocket_tokens_from_string(pocket_str: str, stoi: Dict[str, int]) -> List[int]:
    aa_list = [aa.strip().upper() for aa in pocket_str.split()]
    return [stoi["<POCKET>"]] + [stoi[f"<AA_{aa}>"] for aa in aa_list] + [stoi["</POCKET>"]]


def strip_to_ligand(text: str) -> str:
    if "<POCKET>" in text and "</POCKET>" in text:
        text = text.split("</POCKET>", 1)[1]

    if "<LIGAND>" in text:
        text = text.split("<LIGAND>", 1)[1]
    if "</LIGAND>" in text:
        text = text.split("</LIGAND>", 1)[0]

    text = text.split("<EOS>")[0]

    for tok in ["<SOS>", "<POCKET>", "</POCKET>", "<LIGAND>", "</LIGAND>", "<EOS>"]:
        text = text.replace(tok, "")

    return text.strip()


class PromptBuilder:
    def __init__(self, cfg, stoi: Dict[str, int], adapter):
        self.cfg = cfg
        self.stoi = stoi
        self.adapter = adapter
        prefix_tokens = self.adapter.build_prompt_prefix(self.stoi)
        self.initial_ligand_tokens = int(cfg["prompt"].get("initial_ligand_tokens", 7))
        self.prompt_size = len(prefix_tokens) + 1 + self.initial_ligand_tokens
        
    def build_prompt_tokens(self, smiles: str) -> Optional[List[int]]:
        smiles = smiles.strip()
        if not smiles:
            return None

        selfie = sf.encoder(smiles)
        if not selfie:
            return None

        selfie_tokens = split_selfies(selfie)
        prefix_tokens = self.adapter.build_prompt_prefix(self.stoi)

        min_required = len(prefix_tokens) + 1  # +1 por <LIGAND>
        if self.prompt_size < min_required:
            raise ValueError(
                f"prompt_size={self.prompt_size} demasiado pequeño para el prefijo "
                f"(mínimo requerido: {min_required})"
            )

        if any(token not in self.stoi for token in selfie_tokens):
            return None

        ligand_tokens = [self.stoi["<LIGAND>"]]
        ligand_tokens.extend([self.stoi[token] for token in selfie_tokens])

        tokens = prefix_tokens + ligand_tokens

        if len(tokens) < self.prompt_size:
            return None

        if len(tokens) > self.prompt_size:
            tokens = tokens[: self.prompt_size]

        return tokens


class PromptDataset:
    def __init__(self, smiles_list: List[str], prompt_builder: PromptBuilder):
        self.prompts_input_ids: List[List[int]] = []
        self.prompt_builder = prompt_builder

        for smiles in smiles_list:
            try:
                if pd.notna(smiles):
                    tokens = self.prompt_builder.build_prompt_tokens(str(smiles))
                    if tokens is not None:
                        self.prompts_input_ids.append(tokens)
            except Exception as e:
                print(f"Error al convertir SMILES '{smiles}': {e}")
                continue

    def __getitem__(self, ix: int) -> List[int]:
        return self.prompts_input_ids[ix]

    def __len__(self) -> int:
        return len(self.prompts_input_ids)


class CustomPromptDataGenerator:
    def __init__(self, prompt_dataset: PromptDataset, prompt_batch_size: int):
        self.prompt_dataset = prompt_dataset
        self.prompt_batch_size = prompt_batch_size
        self.dataset_indices = np.arange(len(self.prompt_dataset))

    def __iter__(self):
        self.dataset_indices = np.arange(len(self.prompt_dataset))
        return self

    def __next__(self):
        if len(self.dataset_indices) < self.prompt_batch_size:
            raise StopIteration

        picked_indices = np.random.choice(
            self.dataset_indices,
            self.prompt_batch_size,
            replace=False,
        )
        samples = [self.prompt_dataset[i] for i in picked_indices]
        self.dataset_indices = np.setdiff1d(
            self.dataset_indices,
            picked_indices,
            assume_unique=True,
        )

        input_ids = torch.tensor(samples, dtype=torch.long)
        return {"input_ids": input_ids}