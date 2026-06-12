import pickle
import numpy as np
import pandas as pd
import torch


class SequenceLoader:
    def __init__(self, cfg: dict, device: str, device_type: str):
        data_cfg = cfg["data"]
        self.device = device
        self.device_type = device_type
        self.batch_size = int(data_cfg["batch_size"])
        self.block_size = int(data_cfg["block_size"])
        self.meta_path = data_cfg["meta_path"]
        self.pad_token_id = int(data_cfg.get("pad_token_id", 0))

        self.datasets = {
            "train": self._load_bin(data_cfg["train_bin"]),
            "val": self._load_bin(data_cfg["val_bin"]),
        }

    def _load_bin(self, path: str) -> np.memmap:
        data = np.memmap(path, dtype=np.uint16, mode="r")
        if data.size % self.block_size != 0:
            raise ValueError(
                f"{path} contiene {data.size} tokens, no divisible por "
                f"block_size={self.block_size}"
            )
        if data.size == 0:
            raise ValueError(f"Dataset vacío: {path}")
        return data.reshape(-1, self.block_size)

    def get_vocab_size(self):
        with open(self.meta_path, "rb") as f:
            meta = pickle.load(f)
        return meta.get("vocab_size", len(meta["stoi"]))

    def get_batch(self, split: str):
        if split not in self.datasets:
            raise ValueError(f"Split desconocido: {split}")

        dataset = self.datasets[split]
        row_indices = np.random.randint(0, len(dataset), size=self.batch_size)
        tokens = np.asarray(dataset[row_indices], dtype=np.int64)

        targets = np.empty_like(tokens)
        targets[:, :-1] = tokens[:, 1:]
        targets[:, -1] = self.pad_token_id

        x = torch.from_numpy(tokens)
        y = torch.from_numpy(targets)
        padding_mask = x.eq(self.pad_token_id)

        if self.device_type == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
            padding_mask = padding_mask.pin_memory().to(
                self.device, non_blocking=True
            )
        else:
            x = x.to(self.device)
            y = y.to(self.device)
            padding_mask = padding_mask.to(self.device)

        return {
            "idx": x,
            "targets": y,
            "padding_mask": padding_mask,
        }


class SequenceAddLoader:
    REQUIRED_COLUMNS = {"token_ids", "start", "seq_pocket"}

    def __init__(self, cfg: dict, device: str, device_type: str):
        self.device = device
        self.device_type = device_type

        data_cfg = cfg["data"]
        self.batch_size = int(data_cfg["batch_size"])
        self.block_size = int(data_cfg["block_size"])
        self.meta_path = data_cfg["meta_path"]
        self.parquet_path = data_cfg["parquet_path"]
        self.pocket_embeddings_path = data_cfg.get(
            "pocket_embeddings_path",
            data_cfg.get("pocket_npz_path"),
        )
        if not self.pocket_embeddings_path:
            raise ValueError("data.pocket_embeddings_path is required")
        self.pad_token_id = int(data_cfg.get("pad_token_id", 0))
        self.aa_start = int(data_cfg.get("aa_start", 2))
        self.val_fraction = float(data_cfg.get("val_fraction", 0.1))
        self.split_seed = int(data_cfg.get("split_seed", 42))

        if not 0 <= self.aa_start < self.block_size:
            raise ValueError(
                f"aa_start={self.aa_start} must be in [0, {self.block_size})"
            )
        if not 0 < self.val_fraction < 1:
            raise ValueError("val_fraction must be between 0 and 1")

        self.dataset_df = pd.read_parquet(self.parquet_path)
        if self.dataset_df.empty:
            raise ValueError(f"Empty dataset: {self.parquet_path}")
        if len(self.dataset_df) < 2:
            raise ValueError("SequenceAddLoader requires at least two rows")

        missing_columns = self.REQUIRED_COLUMNS - set(self.dataset_df.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required parquet columns: {sorted(missing_columns)}"
            )

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

        pack = np.load(self.pocket_embeddings_path, mmap_mode="r")
        if isinstance(pack, np.ndarray):
            self.emb_stack = pack
        else:
            if "emb_stack" not in pack.files:
                raise ValueError(
                    f"{self.pocket_embeddings_path} must contain an "
                    "'emb_stack' array"
                )
            self.emb_stack = pack["emb_stack"]
        if self.emb_stack.ndim != 2:
            raise ValueError(
                f"emb_stack must be 2D, got shape {self.emb_stack.shape}"
            )
        self.d_prot = self.emb_stack.shape[1]
        expected_d_prot = data_cfg.get("pocket_embedding_dim")
        if expected_d_prot is not None and self.d_prot != int(expected_d_prot):
            raise ValueError(
                f"Pocket embedding dimension mismatch: expected "
                f"{expected_d_prot}, got {self.d_prot}"
            )

        self._validate_embedding_index()

        all_indices = np.arange(len(self.dataset_df))
        rng = np.random.default_rng(self.split_seed)
        rng.shuffle(all_indices)

        split_idx = int((1.0 - self.val_fraction) * len(all_indices))
        split_idx = min(max(split_idx, 1), len(all_indices) - 1)
        self.train_idx = all_indices[:split_idx]
        self.val_idx = all_indices[split_idx:]

    def _validate_embedding_index(self):
        stack_length = len(self.emb_stack)
        sequences = self.dataset_df["seq_pocket"]
        valid_sequences = sequences.map(
            lambda seq: isinstance(seq, str) and bool(seq)
        )
        if not valid_sequences.all():
            row_idx = valid_sequences[~valid_sequences].index[0]
            raise ValueError(
                f"Row {row_idx} has an empty or invalid seq_pocket"
            )

        sequence_lengths = sequences.str.len().astype(np.int64)
        starts = self.dataset_df["start"].astype(np.int64)
        stored_lengths = sequence_lengths

        if "length" in self.dataset_df.columns:
            indexed_lengths = self.dataset_df["length"].astype(np.int64)
            valid_lengths = indexed_lengths.eq(sequence_lengths) | indexed_lengths.eq(
                sequence_lengths + 1
            )
            mismatches = ~valid_lengths
            if mismatches.any():
                row_idx = mismatches[mismatches].index[0]
                raise ValueError(
                    f"Row {row_idx} length mismatch: "
                    f"length={indexed_lengths.loc[row_idx]}, "
                    f"len(seq_pocket)={sequence_lengths.loc[row_idx]}; "
                    "expected one residue embedding per amino acid and "
                    "optionally one final ProtT5 special-token embedding"
                )
            stored_lengths = indexed_lengths

        invalid_ranges = starts.lt(0) | (starts + stored_lengths).gt(
            stack_length
        )
        if invalid_ranges.any():
            row_idx = invalid_ranges[invalid_ranges].index[0]
            start = starts.loc[row_idx]
            end = start + stored_lengths.loc[row_idx]
            raise ValueError(
                f"Row {row_idx} embedding range [{start}, {end}) is outside "
                f"emb_stack length {stack_length}"
            )

    def get_vocab_size(self):
        if self.meta_path is None:
            return None
        with open(self.meta_path, "rb") as f:
            meta = pickle.load(f)
        return meta.get("vocab_size")

    def get_batch(self, split: str):
        if split == "train":
            idx_pool = self.train_idx
        elif split == "val":
            idx_pool = self.val_idx
        else:
            raise ValueError(f"Unknown split: {split}")

        if len(idx_pool) == 0:
            raise ValueError(f"No examples available for split='{split}'")

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
