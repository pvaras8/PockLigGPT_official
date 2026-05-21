import os
import pickle
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

from .agent import PPOAgent
from .losses import gae, logprobs_from_logits, normalize_advantages, ppo_loss
from .model_adapters import get_torch_dtype
from .prompts import CustomPromptDataGenerator, PromptBuilder, PromptDataset, strip_to_ligand
from pockliggpt.rewards import RewardRunner
from .storage import PPORLElement, PPORolloutStorage


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tokenizer_meta_from_checkpoint(cfg) -> Tuple[Dict[str, int], Dict[int, str]]:
    meta_path = cfg.tokenizer.meta_path

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No existe meta_path: {meta_path}")

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    return meta["stoi"], meta["itos"]


class RolloutCreator:
    def __init__(
        self,
        cfg,
        prompt_dataset: PromptDataset,
        stoi: Dict[str, int],
        decode_fn,
        reward_runner: RewardRunner,
        ref_model: PPOAgent,
    ):
        self.cfg = cfg
        self.stoi = stoi
        self.decode = decode_fn
        self.reward_runner = reward_runner
        self.ref_model = ref_model

        self.train_device = cfg["system"]["train_device"]
        self.ref_device = cfg["system"]["ref_device"]
        self.prompt_batch_size = int(cfg["rl"]["prompt_batch_size"])
        self.kl_coef = float(cfg["rl"]["kl_coef"])
        self.save_trajectories = bool(cfg["output"].get("save_trajectories", False))

        self.trajectories_dir = cfg["output"].get("trajectories_dir", None)
        if self.trajectories_dir is None:
            ppo_ckpt_path = cfg["output"]["ppo_ckpt_path"]
            self.trajectories_dir = os.path.join(os.path.dirname(ppo_ckpt_path), "trajectories")

        os.makedirs(self.trajectories_dir, exist_ok=True)

        self.prompt_dataset = prompt_dataset
        self.prompt_generator = CustomPromptDataGenerator(prompt_dataset, self.prompt_batch_size)
        self.prompt_iterator = iter(self.prompt_generator)

    def _save_trajectories(self, trajectories: torch.Tensor, epoch: int):
        if not self.save_trajectories:
            return

        output_file = os.path.join(self.trajectories_dir, f"trajectories_{epoch}.txt")
        with open(output_file, "w") as f:
            for traj in trajectories:
                f.write(" ".join(map(str, traj.tolist())) + "\n")

    def make_experience(self, model: PPOAgent, epoch: int, num_rollouts: int):
        all_rollouts = []
        all_scores = []

        while len(all_rollouts) < num_rollouts:
            try:
                batch = next(self.prompt_iterator)
            except StopIteration:
                self.prompt_generator = CustomPromptDataGenerator(
                    self.prompt_dataset,
                    self.prompt_batch_size,
                )
                self.prompt_iterator = iter(self.prompt_generator)
                batch = next(self.prompt_iterator)

            query_tensors = batch["input_ids"].to(self.train_device)
            trajectories = model.generate(query_tensors, epoch=epoch)
            self._save_trajectories(trajectories, epoch)

            padding_mask = trajectories.eq(self.stoi["<PAD>"]).to(self.train_device)

            with torch.no_grad():
                logits, values = model(trajectories, padding_mask=padding_mask)

                trajectories_ref = trajectories.to(self.ref_device)
                padding_mask_ref = padding_mask.to(self.ref_device)

                ref_logits = self.ref_model(
                    trajectories_ref,
                    padding_mask=padding_mask_ref,
                )
                ref_logits = ref_logits.to(self.train_device)

            logprobs = logprobs_from_logits(logits[:, :-1, :], trajectories[:, 1:])
            ref_logprobs = logprobs_from_logits(ref_logits[:, :-1, :], trajectories[:, 1:])

            n_trajectories = trajectories.shape[0]
            values = values[:, :-1]

            start = batch["input_ids"].shape[1] - 1
            valid_mask = ~padding_mask[:, 1:]
            valid_counts = valid_mask[:, start:].sum(1)
            ends = start + valid_counts

            texts = [self.decode(seq.tolist()) for seq in trajectories]
            texts = [strip_to_ligand(t) for t in texts]
            scores = self.reward_runner(texts, epoch)

            if len(scores) != n_trajectories:
                raise RuntimeError(
                    f"RewardRunner devolvió {len(scores)} scores pero esperaba {n_trajectories}"
                )

            rewards = -self.kl_coef * (logprobs - ref_logprobs)

            valid_indices = []
            truncated_responses = []
            truncated_values = []
            truncated_logprobs = []
            all_rewards = []

            for i in range(n_trajectories):
                seq_end = int(ends[i].item())
                seq_len = seq_end - start

                if seq_len <= 0:
                    continue

                valid_indices.append(i)

                traj_response = trajectories[i, start + 1 : seq_end + 1].clone()
                traj_values = values[i, start:seq_end].clone()
                traj_logprobs = logprobs[i, start:seq_end].clone()
                traj_rewards = rewards[i, start:seq_end].clone()

                traj_rewards[-1] += scores[i]

                truncated_responses.append(traj_response)
                truncated_values.append(traj_values)
                truncated_logprobs.append(traj_logprobs)
                all_rewards.append(traj_rewards)
                all_scores.append(scores[i])

            if len(valid_indices) == 0:
                continue

            truncated_responses_padded = pad_sequence(
                truncated_responses,
                batch_first=True,
                padding_value=self.stoi["<PAD>"],
            )
            truncated_values_padded = pad_sequence(
                truncated_values,
                batch_first=True,
                padding_value=float("nan"),
            )
            truncated_logprobs_padded = pad_sequence(
                truncated_logprobs,
                batch_first=True,
                padding_value=float("nan"),
            )
            all_rewards_padded = pad_sequence(
                all_rewards,
                batch_first=True,
                padding_value=float("nan"),
            )

            new_rollouts = [
                PPORLElement(
                    query_tensor=query_tensors[i].detach().cpu(),
                    response_tensor=truncated_responses_padded[j].detach().cpu(),
                    logprobs=truncated_logprobs_padded[j].detach().cpu(),
                    values=truncated_values_padded[j].detach().cpu(),
                    rewards=all_rewards_padded[j].detach().cpu(),
                )
                for j, i in enumerate(valid_indices)
            ]
            all_rollouts.extend(new_rollouts)

        rollouts_out = all_rollouts[:num_rollouts]
        scores_out = all_scores[: len(rollouts_out)]
        mean_score = float(np.mean(scores_out)) if len(scores_out) > 0 else 0.0

        return rollouts_out, mean_score


def compute_loss(cfg, batch, model: PPOAgent, stoi: Dict[str, int], device: str):
    query_tensors = batch.query_tensors.to(device)
    response_tensors = batch.response_tensors.to(device)
    old_logprobs = batch.logprobs.to(device)
    old_values = batch.values.to(device)
    old_rewards = batch.rewards.to(device)

    response_length = old_rewards.shape[1]
    mask = ~torch.isnan(old_rewards)
    mask = mask.float()

    old_values_masked = torch.where(mask == 1, old_values, torch.zeros_like(old_values))
    old_rewards_masked = torch.where(mask == 1, old_rewards, torch.zeros_like(old_rewards))
    old_logprobs_masked = torch.where(mask == 1, old_logprobs, torch.zeros_like(old_logprobs))

    advantages, returns = gae(
        values=old_values_masked,
        rewards=old_rewards_masked,
        mask=mask,
        gamma=float(cfg["rl"]["gamma"]),
        lam=float(cfg["rl"]["lam"]),
    )
    advantages = normalize_advantages(advantages, mask)

    trajectories = torch.hstack([query_tensors, response_tensors])
    padding_mask = trajectories.eq(stoi["<PAD>"])

    logits, values_pred = model(trajectories, padding_mask=padding_mask)
    values_pred = values_pred[:, :-1]
    logprobs = logprobs_from_logits(logits[:, :-1, :], trajectories[:, 1:])

    start = query_tensors.shape[1] - 1
    end = start + response_length

    logprobs = logprobs[:, start:end]
    values_pred = values_pred[:, start:end]

    loss, stats = ppo_loss(
        logprobs=logprobs,
        values=values_pred,
        old_logprobs=old_logprobs_masked,
        old_values=old_values_masked,
        advantages=advantages,
        returns=returns,
        mask=mask,
        cliprange=float(cfg["rl"]["cliprange"]),
        cliprange_value=float(cfg["rl"]["cliprange_value"]),
        vf_coef=float(cfg["rl"]["vf_coef"]),
    )

    lengths = mask.sum(dim=1).long().clamp_min(1)
    last_idx = (lengths - 1).unsqueeze(1)
    last_rewards = old_rewards_masked.gather(1, last_idx).squeeze(1)
    mean_last_reward = last_rewards.mean().item()

    return loss, stats, mean_last_reward


def save_loss_stats(loss_history_file: str, stats: dict):
    with open(loss_history_file, "a") as f:
        f.write(
            f"{stats['loss']},{stats['pg_loss']},{stats['vf_loss']},{stats['pg_clipfrac']}\n"
        )


def run_ppo_training(cfg, adapter):
    if "seed" in cfg:
        set_seed(int(cfg["seed"]))

    train_device = cfg["system"]["train_device"]
    ref_device = cfg["system"]["ref_device"]
    dtype_name = cfg["system"]["dtype"]
    dtype = get_torch_dtype(dtype_name)

    ckpt_path = cfg["model"]["checkpoint_path"]
    stoi, itos = load_tokenizer_meta_from_checkpoint(cfg)

    decode = lambda ids: "".join([itos.get(i, "") for i in ids if i in itos])

    smiles_df = pd.read_csv(cfg["data"]["smiles_csv"])
    smiles_list = smiles_df[cfg["data"]["smiles_column"]].astype(str).str.strip().tolist()

    prompt_builder = PromptBuilder(cfg, stoi, adapter)
    prompt_dataset = PromptDataset(smiles_list, prompt_builder)

    ref_model = PPOAgent(
        cfg=cfg,
        adapter=adapter,
        stoi=stoi,
        model_checkpoint_path=ckpt_path,
        trainable=False,
        device=ref_device,
        dtype=dtype,
    )

    model = PPOAgent(
        cfg=cfg,
        adapter=adapter,
        stoi=stoi,
        model_checkpoint_path=ckpt_path,
        trainable=True,
        device=train_device,
        dtype=dtype,
    )

    reward_runner = RewardRunner(cfg)
    rollout_creator = RolloutCreator(
        cfg=cfg,
        prompt_dataset=prompt_dataset,
        stoi=stoi,
        decode_fn=decode,
        reward_runner=reward_runner,
        ref_model=ref_model,
    )

    store = PPORolloutStorage(pad_token_id=stoi["<PAD>"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["rl"]["lr"]))

    batch_size = int(cfg["rl"]["batch_size"])
    ppo_epochs = int(cfg["rl"]["ppo_epochs"])
    num_rollouts = int(cfg["rl"]["num_rollouts"])
    epochs = int(cfg["rl"]["epochs"])

    total_steps = (num_rollouts // batch_size) * ppo_epochs * epochs
    tbar = tqdm(initial=0, total=total_steps)

    ckpt_ppo_path = cfg["output"]["ppo_ckpt_path"]
    loss_history_file = cfg["output"]["loss_history_file"]

    ckpt_dir = os.path.dirname(ckpt_ppo_path)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    loss_dir = os.path.dirname(loss_history_file)
    if loss_dir:
        os.makedirs(loss_dir, exist_ok=True)

    all_scores = []
    best_score = -float("inf")

    for i in range(epochs):
        store.clear_history()
        rollouts, score = rollout_creator.make_experience(model, i, num_rollouts)
        store.push(rollouts)

        train_dataloader = store.create_loader(batch_size, shuffle=True)

        for batch in train_dataloader:
            for _ in range(ppo_epochs):
                loss, stats, reward = compute_loss(
                    cfg=cfg,
                    batch=batch,
                    model=model,
                    stoi=stoi,
                    device=train_device,
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                save_loss_stats(loss_history_file, stats)
                tbar.update()

        all_scores.append(score)
        print("all_scores", all_scores)
        tbar.set_description(f"| Época: {i+1} | Recompensa Promedio: {score:.3f} |")

        if score > best_score:
            best_score = score
            checkpoint_ppo = {
                "epoch": i + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "all_scores": all_scores,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            torch.save(checkpoint_ppo, ckpt_ppo_path)
            print(
                f"✅ Nuevo mejor modelo guardado en {ckpt_ppo_path} "
                f"(Época {i+1}, Recompensa: {best_score:.3f})"
            )