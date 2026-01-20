"""Transformer encoder implementation."""
from __future__ import annotations

from typing import List

import torch
from torch import nn

from .attention import MultiHeadAttention
from .layers import AddNorm, FeedForward


class EncoderLayer(nn.Module):
    """Single Transformer encoder layer."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """Apply self-attention and feed-forward sublayers."""
        attn_output = self.self_attn(x, x, x, mask=src_mask)
        x = self.add_norm1(x, attn_output)
        ffn_output = self.ffn(x)
        x = self.add_norm2(x, ffn_output)
        return x


class Encoder(nn.Module):
    """Transformer encoder stack."""

    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.layers: List[EncoderLayer] = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """Run input through the encoder stack."""
        for layer in self.layers:
            x = layer(x, src_mask)
        return x
