import os
import pickle
import re


class SelfiesTokenizer:
    def __init__(self, meta_path):
        if not meta_path:
            raise ValueError("Debes pasar meta_path obligatoriamente.")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"No existe el vocabulario: {meta_path}")

        self.load_meta(meta_path)
        self.token2id = {v: k for k, v in self.id2token.items()}

    def split_selfies(self, selfies_str):
        if not isinstance(selfies_str, str):
            return []
        return re.findall(r"\[[^\]]+\]", selfies_str)

    def tokenize(self, selfies_str, max_length=156):
        toks = self.split_selfies(selfies_str)

        enc = [
            self.token2id["<SOS>"],
            self.token2id["<LIGAND>"],
        ]
        enc += [self.token2id.get(t, self.token2id["<UNK>"]) for t in toks]
        enc += [
            self.token2id["</LIGAND>"],
            self.token2id["<EOS>"],
        ]

        if len(enc) > max_length:
            raise ValueError(f"Sequence length {len(enc)} exceeds max_length={max_length}")

        if len(enc) < max_length:
            enc += [self.token2id["<PAD>"]] * (max_length - len(enc))

        return enc

    def decode(self, token_ids):
        skip = {"<PAD>", "<SOS>", "<EOS>", "<LIGAND>", "</LIGAND>", "<UNK>"}
        return "".join([self.id2token[i] for i in token_ids if self.id2token[i] not in skip])

    def filter_valid_selfies(self, df, max_len):
        return df[df["selfies"].apply(lambda x: len(self.split_selfies(x)) <= max_len - 4)]

    def load_meta(self, path):
        with open(path, "rb") as f:
            meta = pickle.load(f)
        self.id2token = meta["itos"]
        self.token2id = meta["stoi"]

    @property
    def vocab_size(self):
        return len(self.id2token)