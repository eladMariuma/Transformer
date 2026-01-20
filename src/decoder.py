"""Transformer decoder implementation."""
from __future__ import annotations

from typing import List

import torch
from torch import nn

from .attention import MultiHeadAttention
from .layers import AddNorm, FeedForward


class DecoderLayer(nn.Module):
    """Single Transformer decoder layer."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.add_norm3 = AddNorm(d_model, dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply masked self-attention, cross-attention, and feed-forward layers."""
        attn_output = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.add_norm1(x, attn_output)
        cross_output = self.cross_attn(x, memory, memory, mask=memory_mask)
        x = self.add_norm2(x, cross_output)
        ffn_output = self.ffn(x)
        x = self.add_norm3(x, ffn_output)
        return x


class Decoder(nn.Module):
    """Transformer decoder stack."""

    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.layers: List[DecoderLayer] = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run input through the decoder stack."""
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return x
