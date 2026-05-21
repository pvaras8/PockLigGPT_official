"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x, padding_mask=None, epoch=None):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if padding_mask is not None:
            padding_mask = padding_mask.to(torch.bool)
            key_mask = padding_mask[:, None, None, :]   # (B, 1, 1, T)
            att = att.masked_fill(key_mask, float("-inf"))

        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = torch.nan_to_num(att, nan=0.0)

        att = self.attn_dropout(att)

        if epoch is not None:
            att_file = f"attention_mean_{epoch}.txt"
            att_mean_batch = att.mean(1).detach().to(torch.float32).cpu().numpy()
            with open(att_file, "a") as f:
                f.write(
                    f"=== Nueva atención media (batch_size={att_mean_batch.shape[0]}, "
                    f"T={att_mean_batch.shape[1]}) ===\n"
                )
                for bi in range(att_mean_batch.shape[0]):
                    f.write(f"-- sample {bi} --\n")
                    np.savetxt(f, att_mean_batch[bi], fmt="%.4f")
                    f.write("\n")

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        if padding_mask is not None:
            y = y.masked_fill(padding_mask.unsqueeze(-1), 0)

        y = self.resid_dropout(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, padding_mask=None, epoch=None):
        x = x + self.attn(self.ln_1(x), padding_mask=padding_mask, epoch=epoch)
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _resolve_padding_mask(self, padding_mask=None, attention_mask=None):
        if padding_mask is not None and attention_mask is not None:
            if not torch.equal(padding_mask, attention_mask):
                raise ValueError("Se han pasado padding_mask y attention_mask con valores distintos")
        mask = padding_mask if padding_mask is not None else attention_mask
        if mask is not None:
            mask = mask.to(torch.bool)
        return mask

    def _crop_pocket_emb(self, pocket_emb, target_len: int):
        if pocket_emb is None:
            return None
        if pocket_emb.dim() == 2:
            pocket_emb = pocket_emb.unsqueeze(0)
        if pocket_emb.size(1) >= target_len:
            return pocket_emb[:, :target_len, :]
        return pocket_emb

    def forward(
        self,
        idx,
        targets=None,
        padding_mask=None,
        attention_mask=None,  # alias por compatibilidad
        pocket_emb=None,
        return_hidden_states=False,
        epoch=None,
    ):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )

        padding_mask = self._resolve_padding_mask(
            padding_mask=padding_mask,
            attention_mask=attention_mask,
        )

        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)

        if pocket_emb is not None:
            pocket_emb = self._crop_pocket_emb(pocket_emb, t)
            pocket_emb = pocket_emb.to(tok_emb.dtype)
            pocket_emb_ln = self.transformer.ln_f(pocket_emb)
            tok_emb = tok_emb + pocket_emb_ln

        x = self.transformer.drop(tok_emb + pos_emb)

        for i, block in enumerate(self.transformer.h):
            ep = epoch if i == len(self.transformer.h) - 1 else None
            x = block(x, padding_mask=padding_mask, epoch=ep)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        if return_hidden_states:
            return logits, x

        if targets is not None:
            ignore_ids = torch.tensor(
                [0, 4, 6, 7] + list(range(8, 28)),
                device=device,
                dtype=torch.long,
            )

            V = logits.size(-1)
            loss_per_token = F.cross_entropy(
                logits.reshape(-1, V),
                targets.reshape(-1),
                reduction="none",
            )

            mask = ~torch.isin(targets.reshape(-1), ignore_ids)
            denom = mask.float().sum().clamp_min(1.0)
            loss = (loss_per_token * mask.float()).sum() / denom

            return logits, loss

        return logits, None

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}
        assert all(k == "dropout" for k in override_args)

        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        config_args["bias"] = True

        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]

        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        assert len(sd_keys_hf) == len(sd_keys), (
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        )

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(
            f"num decayed parameter tensors: {len(decay_params)}, "
            f"with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, "
            f"with {num_nodecay_params:,} parameters"
        )

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            **extra_args,
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(
        self,
        idx,
        seq_length,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        eos_token=2,
        pad_token=0,
        epoch=None,
        pocket_emb=None,
    ):
        batch_size = idx.size(0)
        eos_generated = torch.zeros(batch_size, dtype=torch.bool, device=idx.device)

        for _ in range(max_new_tokens):
            if idx.size(1) > self.config.block_size:
                idx_cond = idx[:, -self.config.block_size:]
            else:
                idx_cond = idx

            T_cond = idx_cond.size(1)

            pocket_emb_cond = None
            if pocket_emb is not None:
                if pocket_emb.dim() == 2:
                    pocket_emb = pocket_emb.unsqueeze(0)

                if idx.size(1) > self.config.block_size:
                    pocket_emb_cond = pocket_emb[:, -self.config.block_size:, :]
                else:
                    pocket_emb_cond = pocket_emb[:, :T_cond, :]

            logits, _ = self(
                idx_cond,
                epoch=None,
                pocket_emb=pocket_emb_cond,
            )

            logits = logits[:, -1, :] / temperature
            logits[:, pad_token] = -float("inf")

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx_next = torch.where(
                eos_generated.unsqueeze(-1),
                torch.full_like(idx_next, pad_token),
                idx_next,
            )
            idx = torch.cat((idx, idx_next), dim=1)

            eos_generated |= (idx_next.squeeze(-1) == eos_token)

            if eos_generated.all():
                break

        if epoch is not None:
            idx_final = idx[:, -self.config.block_size:] if idx.size(1) > self.config.block_size else idx
            T_final = idx_final.size(1)

            pocket_emb_final = None
            if pocket_emb is not None:
                if pocket_emb.dim() == 2:
                    pocket_emb = pocket_emb.unsqueeze(0)

                if idx.size(1) > self.config.block_size:
                    pocket_emb_final = pocket_emb[:, -self.config.block_size:, :]
                else:
                    pocket_emb_final = pocket_emb[:, :T_final, :]

            was_training = self.training
            self.eval()
            _ = self(
                idx_final,
                epoch=epoch,
                pocket_emb=pocket_emb_final,
            )
            if was_training:
                self.train()

        pad_length = seq_length - idx.size(1)
        if pad_length > 0:
            idx = F.pad(idx, (0, pad_length), value=pad_token)

        return idx