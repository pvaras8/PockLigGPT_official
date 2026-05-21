import os
import pickle
import re


AA_ONE_TO_TOKEN = {
    "A": "<AA_ALA>", "C": "<AA_CYS>", "D": "<AA_ASP>", "E": "<AA_GLU>",
    "F": "<AA_PHE>", "G": "<AA_GLY>", "H": "<AA_HIS>", "I": "<AA_ILE>",
    "K": "<AA_LYS>", "L": "<AA_LEU>", "M": "<AA_MET>", "N": "<AA_ASN>",
    "P": "<AA_PRO>", "Q": "<AA_GLN>", "R": "<AA_ARG>", "S": "<AA_SER>",
    "T": "<AA_THR>", "V": "<AA_VAL>", "W": "<AA_TRP>", "Y": "<AA_TYR>",
}


class PocketLigandTokenizer:
    def __init__(self, meta_path):
        if not meta_path:
            raise ValueError("Debes pasar meta_path obligatoriamente.")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"No existe el vocabulario: {meta_path}")

        self.load_meta(meta_path)
        self.token2id = {v: k for k, v in self.id2token.items()}

    def split_selfies(self, s):
        if not isinstance(s, str):
            return []
        return re.findall(r"\[[^\]]+\]", s)

    def core_length(self, pocket_str, selfies_str):
        return len(pocket_str) + len(self.split_selfies(selfies_str))

    def filter_valid_pairs(self, df_pairs, max_sequence_length):
        max_core = max_sequence_length - 6
        keep = []
        for _, row in df_pairs.iterrows():
            keep.append(self.core_length(row["pocket"], row["selfies"]) <= max_core)
        return df_pairs[keep]

    def tokenize_pair(self, pocket, selfies, max_len=156):
        seq = ["<SOS>", "<POCKET>"]
        seq += [AA_ONE_TO_TOKEN.get(a, "<UNK>") for a in pocket]
        seq += ["</POCKET>", "<LIGAND>"]
        seq += self.split_selfies(selfies)
        seq += ["</LIGAND>", "<EOS>"]

        ids = [self.token2id.get(t, self.token2id["<UNK>"]) for t in seq]

        if len(ids) > max_len:
            raise ValueError(f"Sequence length {len(ids)} exceeds max_len={max_len}")

        if len(ids) < max_len:
            ids += [self.token2id["<PAD>"]] * (max_len - len(ids))

        return ids

    def load_meta(self, path):
        with open(path, "rb") as f:
            meta = pickle.load(f)
        self.id2token = meta["itos"]
        self.token2id = meta["stoi"]

    @property
    def vocab_size(self):
        return len(self.id2token)