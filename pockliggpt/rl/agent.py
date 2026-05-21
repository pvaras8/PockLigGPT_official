from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class PPOAgent(nn.Module):
    def __init__(
        self,
        cfg,
        adapter,
        stoi,
        model_checkpoint_path: str,
        trainable: bool,
        device: str,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.cfg = cfg
        self.adapter = adapter
        self.stoi = stoi
        self.trainable = trainable
        self.device_name = device
        self.dtype = dtype

        generation_cfg = cfg["generation"]

        prefix_tokens = self.adapter.build_prompt_prefix(stoi)
        initial_ligand_tokens = int(cfg["prompt"].get("initial_ligand_tokens", 7))

        prompt_size = len(prefix_tokens) + 1 + initial_ligand_tokens
        seq_length = int(generation_cfg["seq_length"])
        max_new_tokens = seq_length - prompt_size

        if max_new_tokens <= 0:
            raise ValueError(
                f"seq_length={seq_length} demasiado pequeño para prompt_size={prompt_size}"
            )

        self.generate_kwargs = {
            "seq_length": seq_length,
            "max_new_tokens": max_new_tokens,
            "top_k": generation_cfg["top_k"],
            "temperature": generation_cfg["temperature"],
            "eos_token": stoi["<EOS>"],
            "pad_token": stoi["<PAD>"],
        }

        GPTConfig, GPT = self.adapter.load_model_classes()

        checkpoint = torch.load(model_checkpoint_path, map_location=device)
        self._validate_checkpoint(checkpoint, model_checkpoint_path)

        gptconf = GPTConfig(**checkpoint["model_args"])
        self.model = GPT(gptconf).to(device=device, dtype=dtype)

        cleaned_state_dict = self._clean_state_dict(checkpoint["model"])
        self.model.load_state_dict(cleaned_state_dict, strict=False)

        self.logit_head = self.model.lm_head
        self.block_size = self.model.config.block_size
        self.n_embd = self.model.config.n_embd

        if not self.trainable:
            self.model.eval()
            self.model.requires_grad_(False)
            self.value_head = None
        else:
            self.model.train()
            self.model.requires_grad_(True)
            self.adapter.maybe_freeze(self.model)
            self.value_head = self._build_value_head(self.n_embd, device, dtype)

    @staticmethod
    def _validate_checkpoint(checkpoint: dict, checkpoint_path: str) -> None:
        required_keys = ["model_args", "model"]
        for key in required_keys:
            if key not in checkpoint:
                raise KeyError(
                    f"El checkpoint '{checkpoint_path}' no contiene la clave requerida '{key}'"
                )

    @staticmethod
    def _clean_state_dict(state_dict: dict) -> dict:
        unwanted_prefix = "_orig_mod."
        cleaned_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith(unwanted_prefix):
                key = key[len(unwanted_prefix) :]
            cleaned_state_dict[key] = value

        return cleaned_state_dict

    @staticmethod
    def _build_value_head(n_embd: int, device: str, dtype: torch.dtype) -> nn.Module:
        return nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, 1),
        ).to(device=device, dtype=dtype)

    def _extra_model_kwargs(self, input_ids: torch.Tensor) -> dict:
        return self.adapter.build_model_kwargs(
            input_ids=input_ids,
            device=self.device_name,
            dtype=self.dtype,
            block_size=self.block_size,
            n_embd=self.n_embd,
        )

    def generate(self, input_ids: torch.Tensor, epoch: int) -> torch.Tensor:
        extra_kwargs = self._extra_model_kwargs(input_ids)
        return self.model.generate(
            idx=input_ids,
            epoch=epoch,
            **extra_kwargs,
            **self.generate_kwargs,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        extra_kwargs = self._extra_model_kwargs(input_ids)

        lm_logits, last_hidden_state = self.model(
            input_ids,
            padding_mask=padding_mask,
            return_hidden_states=True,
            **extra_kwargs,
        )

        if not self.trainable:
            return lm_logits

        if self.value_head is None:
            raise RuntimeError("value_head no está inicializado en modo trainable")

        last_hidden_state = last_hidden_state.to(self.dtype)
        value = self.value_head(last_hidden_state).squeeze(-1)
        return lm_logits, value