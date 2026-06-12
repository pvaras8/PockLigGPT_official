import torch

from pockliggpt.models.model_sequence import GPT, GPTConfig


def test_generate_restores_training_mode(tmp_path):
    config = GPTConfig(
        block_size=8,
        vocab_size=16,
        n_layer=1,
        n_head=1,
        n_embd=8,
        dropout=0.1,
        bias=True,
    )
    config.attention_output_dir = str(tmp_path)
    model = GPT(config)
    model.train()

    model.generate(
        idx=torch.tensor([[1, 2]], dtype=torch.long),
        seq_length=4,
        max_new_tokens=2,
        temperature=1.0,
        top_k=4,
        eos_token=15,
        pad_token=0,
        epoch=0,
    )

    assert model.training
    assert (tmp_path / "attention_mean_0.txt").is_file()
