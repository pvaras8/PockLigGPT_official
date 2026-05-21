from typing import Tuple

import torch
import torch.nn.functional as F


def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logprobs = F.log_softmax(logits, dim=-1)
    gathered = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1))
    return gathered.squeeze(-1)


def gae(
    values: torch.Tensor,
    rewards: torch.Tensor,
    mask: torch.Tensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards, device=rewards.device)
    last_advantage = torch.zeros(
        rewards.shape[0],
        device=rewards.device,
        dtype=rewards.dtype,
    )
    last_value = torch.zeros(
        values.shape[0],
        device=values.device,
        dtype=values.dtype,
    )

    with torch.no_grad():
        for t in reversed(range(rewards.shape[1])):
            delta = rewards[:, t] + gamma * last_value * mask[:, t] - values[:, t]
            last_advantage = delta + gamma * lam * last_advantage * mask[:, t]
            advantages[:, t] = last_advantage * mask[:, t]
            last_value = values[:, t] * mask[:, t]

        returns = advantages + values

    return advantages, returns


def normalize_advantages(advantages: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid_mask = mask.bool()
    valid = advantages[valid_mask]

    if valid.numel() == 0:
        return torch.zeros_like(advantages)

    mean = valid.mean()
    std = valid.std(unbiased=False).clamp_min(1e-8)
    normalized = (advantages - mean) / std

    return torch.where(valid_mask, normalized, torch.zeros_like(advantages))


def ppo_loss(
    logprobs: torch.Tensor,
    values: torch.Tensor,
    old_logprobs: torch.Tensor,
    old_values: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    cliprange: float,
    cliprange_value: float,
    vf_coef: float,
) -> Tuple[torch.Tensor, dict]:
    values_clipped = torch.clamp(
        values,
        old_values - cliprange_value,
        old_values + cliprange_value,
    )

    n = mask.sum().clamp_min(1.0)

    vf_loss1 = (values - returns) ** 2
    vf_loss2 = (values_clipped - returns) ** 2
    vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask) / n

    log_ratio = (logprobs - old_logprobs) * mask
    ratio = torch.exp(log_ratio)

    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / n

    pg_clipfrac = torch.sum((pg_loss2 > pg_loss1).float() * mask) / n

    loss = pg_loss + vf_coef * vf_loss

    stats = {
        "loss": float(loss.item()),
        "pg_loss": float(pg_loss.item()),
        "vf_loss": float(vf_loss.item()),
        "pg_clipfrac": float(pg_clipfrac.item()),
    }
    return loss, stats