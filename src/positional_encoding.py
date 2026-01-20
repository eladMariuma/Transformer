"""Sinusoidal positional encoding implementation."""
from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Add sinusoidal positional encodings to token embeddings."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        positions = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to the input embeddings."""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]
