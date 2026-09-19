#!/usr/bin/env python
"""Small, dependency-free building blocks for CycleManip-style history."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def cycle_image_delta_indices(history_size: int = 32, num_frames: int = 6) -> list[int]:
    """Return sparse visual offsets with a long temporal horizon."""
    if history_size < 2:
        raise ValueError("history_size must be at least 2")
    if num_frames < 2:
        raise ValueError("num_frames must be at least 2")
    offsets = [-(history_size - 1)]
    for divisor in (2, 4):
        offsets.append(-max(1, history_size // divisor))
    offsets.extend([-2, -1, 0])
    result = sorted(set(offsets))
    if len(result) < num_frames:
        result = list(range(-(num_frames - 1), 1))
    if len(result) > num_frames:
        result = result[: num_frames - 1] + [0]
    return result


def cycle_state_delta_indices(history_size: int = 32) -> list[int]:
    """Return dense state offsets for the low-cost proprioceptive history."""
    if history_size < 1:
        raise ValueError("history_size must be positive")
    return list(range(-(history_size - 1), 1))


def cycle_progress_targets(
    frame_index: Tensor,
    episode_index: Tensor,
    episode_lengths: dict[str, int] | None,
    num_bins: int = 10,
    fallback_length: int = 1,
) -> Tensor:
    """Convert episode-relative frame positions into discretized progress labels."""
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2")
    frame_index = frame_index.reshape(-1).to(dtype=torch.float32)
    episode_index = episode_index.reshape(-1).to(dtype=torch.long)
    lengths = torch.full_like(frame_index, max(1, fallback_length))
    if episode_lengths:
        for episode_id, length in episode_lengths.items():
            mask = episode_index == int(episode_id)
            lengths = torch.where(mask, torch.as_tensor(max(1, int(length)), device=lengths.device), lengths)
    progress = frame_index / (lengths - 1).clamp_min(1)
    return torch.floor(progress.clamp(0, 1) * num_bins).long().clamp_max(num_bins - 1)


class CycleHistoryEncoder(nn.Module):
    """Encode dense state history into one VLM prefix token."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        output_dim: int,
        max_history: int,
        num_layers: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.max_history = max_history
        self.input_proj = nn.Linear(state_dim, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_history, hidden_dim))
        nn.init.normal_(self.position_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, state_history: Tensor, is_pad: Tensor | None = None) -> Tensor:
        if state_history.ndim != 3:
            raise ValueError(f"Expected state history shaped (B,T,D), got {tuple(state_history.shape)}")
        if state_history.shape[1] > self.max_history:
            state_history = state_history[:, -self.max_history :]
        batch_size, length, _ = state_history.shape
        if is_pad is None:
            is_pad = torch.zeros(batch_size, length, dtype=torch.bool, device=state_history.device)
        else:
            is_pad = is_pad.to(device=state_history.device, dtype=torch.bool)
            if is_pad.ndim == 1:
                is_pad = is_pad[:, None]
            is_pad = is_pad[:, -length:]

        x = self.input_proj(state_history) + self.position_embedding[:, -length:]
        encoded = self.encoder(x, src_key_padding_mask=is_pad)
        valid = (~is_pad).to(dtype=encoded.dtype)
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_feature = (encoded * valid.unsqueeze(-1)).sum(dim=1) / denom
        positions = torch.arange(length, device=encoded.device).expand(batch_size, -1)
        last_index = positions.masked_fill(is_pad, -1).amax(dim=1).clamp_min(0)
        last_feature = encoded[torch.arange(batch_size, device=encoded.device), last_index]
        return self.fusion(torch.cat([mean_feature, last_feature], dim=-1))
