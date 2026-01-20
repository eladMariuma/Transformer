"""Shared layers and masking utilities."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network."""
        return self.net(x)


class AddNorm(nn.Module):
    """Residual connection followed by layer normalization."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, sublayer: torch.Tensor) -> torch.Tensor:
        """Add residual connection and apply layer normalization."""
        return self.norm(x + self.dropout(sublayer))


def create_padding_mask(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Create padding mask for attention.

    Returns a boolean mask with shape (batch, 1, 1, seq_len).
    True values indicate positions that should be masked.
    """
    return (tokens == pad_id).unsqueeze(1).unsqueeze(2)


def create_causal_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Create a causal mask for decoder self-attention.

    Returns a boolean mask with shape (1, 1, seq_len, seq_len).
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(0)
