import numpy as np
import pickle

def save_to_bin_file(data, filename):
    data = np.array(data, dtype=np.uint16)
    with open(filename, 'ab') as f:
        data.tofile(f)

def save_meta(tokenizer, meta_path):
    meta = {
        'vocab_size': tokenizer.vocab_size,
        'itos': tokenizer.id2token,
        'stoi': tokenizer.token2id,
    }
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)