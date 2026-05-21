from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


@dataclass
class PPORLElement:
    query_tensor: torch.Tensor
    response_tensor: torch.Tensor
    logprobs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor


@dataclass
class PPORLBatch:
    query_tensors: torch.Tensor
    response_tensors: torch.Tensor
    logprobs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor


class PPORolloutStorage:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id
        self.history: List[PPORLElement] = []

    def push(self, exps: Iterable[PPORLElement]) -> None:
        self.history.extend(exps)

    def clear_history(self) -> None:
        self.history = []

    def __getitem__(self, index: int) -> PPORLElement:
        return self.history[index]

    def __len__(self) -> int:
        return len(self.history)

    def create_loader(self, mini_batch_size: int, shuffle: bool) -> DataLoader:
        def collate_fn(elems: Iterable[PPORLElement]) -> PPORLBatch:
            elems = list(elems)
            return PPORLBatch(
                query_tensors=pad_sequence(
                    [elem.query_tensor for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ),
                response_tensors=pad_sequence(
                    [elem.response_tensor for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ),
                logprobs=pad_sequence(
                    [elem.logprobs for elem in elems],
                    padding_value=float("nan"),
                    batch_first=True,
                ),
                values=pad_sequence(
                    [elem.values for elem in elems],
                    padding_value=float("nan"),
                    batch_first=True,
                ),
                rewards=pad_sequence(
                    [elem.rewards for elem in elems],
                    padding_value=float("nan"),
                    batch_first=True,
                ),
            )

        return DataLoader(
            self,
            batch_size=mini_batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )