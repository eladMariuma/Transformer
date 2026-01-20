"""Attention building blocks for the Transformer."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot-product attention.

    Args:
        query: Tensor of shape (batch, heads, seq_q, d_k)
        key: Tensor of shape (batch, heads, seq_k, d_k)
        value: Tensor of shape (batch, heads, seq_k, d_k)
        mask: Boolean mask where True indicates positions to mask.
        dropout: Optional dropout layer applied to attention weights.
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    output = torch.matmul(weights, value)
    return output, weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention module."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply multi-head attention."""
        batch_size = query.size(0)

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attn_output, _ = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        return self.out_proj(attn_output)
