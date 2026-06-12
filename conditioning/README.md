# Conditioning files

Put your pocket conditioning assets here.

- `pocket_str.txt`: pocket amino-acid sequence separated by spaces.
- `pocket_emb.npy`: ProtT5 embedding array aligned with `pocket_str`.

Expected usage in RL config (`config/rl/sequence_add.yaml`):

```yaml
conditioning:
  pocket_str_path: "conditioning/pocket_str.txt"
  pocket_emb_path: "conditioning/pocket_emb.npy"
  pocket_emb_aa_start: 2
```
