import pickle
import numpy as np
import pandas as pd
import torch


class SequenceAddLoader:
    def __init__(self, cfg: dict, device: str, device_type: str):
        self.cfg = cfg
        self.device = device
        self.device_type = device_type

        data_cfg = cfg["data"]
        self.batch_size = data_cfg["batch_size"]
        self.block_size = data_cfg["block_size"]
        self.meta_path = data_cfg["meta_path"]
        self.parquet_path = data_cfg["parquet_path"]
        self.pocket_npz_path = data_cfg["pocket_npz_path"]
        self.pad_token_id = data_cfg.get("pad_token_id", 0)
        self.aa_start = data_cfg.get("aa_start", 2)

        self.dataset_df = pd.read_parquet(self.parquet_path)

        lengths = self.dataset_df["token_ids"].apply(len)
        if lengths.nunique() != 1:
            raise ValueError(
                f"Inconsistent token_ids lengths in dataset: {sorted(lengths.unique())}"
            )

        dataset_block_size = int(lengths.iloc[0])
        if dataset_block_size != self.block_size:
            raise ValueError(
                f"block_size mismatch: YAML says {self.block_size}, "
                f"but dataset token_ids have length {dataset_block_size}"
            )

        pack = np.load(self.pocket_npz_path, mmap_mode="r")
        self.emb_stack = pack["emb_stack"]
        self.d_prot = self.emb_stack.shape[1]

        all_indices = np.arange(len(self.dataset_df))
        np.random.shuffle(all_indices)

        split_idx = int(0.9 * len(all_indices))
        self.train_idx = all_indices[:split_idx]
        self.val_idx = all_indices[split_idx:]

    def get_vocab_size(self):
        if self.meta_path is None:
            return None
        with open(self.meta_path, "rb") as f:
            meta = pickle.load(f)
        return meta.get("vocab_size")

    def get_batch(self, split: str):
        idx_pool = self.train_idx if split == "train" else self.val_idx

        if len(idx_pool) == 0:
            raise ValueError(f"No hay ejemplos en split='{split}'")

        replace = len(idx_pool) < self.batch_size
        chosen = np.random.choice(idx_pool, size=self.batch_size, replace=replace)

        x_list = []
        y_list = []

        pocket_emb_batch = torch.zeros(
            (self.batch_size, self.block_size, self.d_prot),
            dtype=torch.float32
        )

        for j, row_idx in enumerate(chosen):
            row = self.dataset_df.iloc[row_idx]

            tokens = np.array(row["token_ids"], dtype=np.int64)
            if len(tokens) != self.block_size:
                raise ValueError(
                    f"Row {row_idx} has len {len(tokens)} != block_size {self.block_size}"
                )

            x_seq = torch.from_numpy(tokens)

            y_tokens = np.empty_like(tokens)
            y_tokens[:-1] = tokens[1:]
            y_tokens[-1] = self.pad_token_id
            y_seq = torch.from_numpy(y_tokens)

            x_list.append(x_seq)
            y_list.append(y_seq)

            start = int(row["start"])
            l_seq = len(row["seq_pocket"])
            l = min(l_seq, self.block_size - self.aa_start)

            emb = self.emb_stack[start:start + l]
            emb = np.asarray(emb, dtype=np.float32)

            pocket_emb_batch[j, self.aa_start:self.aa_start + l, :] = torch.from_numpy(emb)

        x = torch.stack(x_list, dim=0)
        y = torch.stack(y_list, dim=0)
        padding_mask = (x == self.pad_token_id)

        if self.device_type == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
            padding_mask = padding_mask.pin_memory().to(self.device, non_blocking=True)
            pocket_emb_batch = pocket_emb_batch.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
            padding_mask = padding_mask.to(self.device)
            pocket_emb_batch = pocket_emb_batch.to(self.device)

        return {
            "idx": x,
            "targets": y,
            "padding_mask": padding_mask,
            "pocket_emb": pocket_emb_batch,
        }