"""
Large Transformer Regression Model for Sprint Sequences
Multi-head self-attention with positional encoding and attention extraction.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import transformer_cfg
from models.common import (PositionalEncoding, InputProjection,
                            RegressionHead, create_padding_mask)


class SprintTransformerLayer(nn.Module):
    """Single Transformer encoder layer with extractable attention."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Store attention weights for explainability
        self.attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None) \
            -> torch.Tensor:
        # Self-attention with residual
        attn_out, self.attn_weights = self.self_attn(
            x, x, x,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual
        x = self.norm2(x + self.ff(x))
        return x


class SprintTransformer(nn.Module):
    """
    Large Transformer encoder for regression on sprint sequences.

    Architecture:
        Input → Projection → Positional Encoding →
        N × TransformerEncoderLayer → CLS pooling → Regression Head

    Supports attention weight extraction for explainability.
    """

    def __init__(self, input_dim: int = None,
                 d_model: int = None,
                 nhead: int = None,
                 num_layers: int = None,
                 dim_feedforward: int = None,
                 dropout: float = None,
                 max_seq_len: int = None):
        super().__init__()
        cfg = transformer_cfg
        self.d_model = d_model or cfg.d_model
        self.nhead = nhead or cfg.nhead
        self.num_layers = num_layers or cfg.num_layers
        input_dim = input_dim or cfg.input_dim
        dim_ff = dim_feedforward or cfg.dim_feedforward
        drop = dropout or cfg.dropout
        max_len = max_seq_len or cfg.max_seq_len

        # CLS token (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)

        # Input projection
        self.input_proj = InputProjection(input_dim, self.d_model, drop)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            self.d_model, max_len=max_len + 1, dropout=drop)

        # Transformer layers (custom for attention extraction)
        self.layers = nn.ModuleList([
            SprintTransformerLayer(self.d_model, self.nhead, dim_ff, drop)
            for _ in range(self.num_layers)
        ])

        # Regression head
        self.regressor = RegressionHead(
            self.d_model, self.d_model // 2, drop)

    def get_attention_weights(self):
        """Extract attention weights from all layers."""
        return [layer.attn_weights for layer in self.layers
                if layer.attn_weights is not None]

    def forward(self, x: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
            lengths: (batch,) actual sequence lengths

        Returns:
            (batch, 1) predicted effort
        """
        batch_size, seq_len, _ = x.shape

        # Project input features
        x = self.input_proj(x)  # (B, L, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, L+1, d_model)

        # Positional encoding
        x = self.pos_encoding(x)

        # Create padding mask (account for CLS token at position 0)
        mask = None
        if lengths is not None:
            # Shift lengths by 1 for CLS token (CLS is always valid)
            mask = create_padding_mask(lengths + 1, seq_len + 1)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=mask)

        # CLS token representation
        cls_repr = x[:, 0, :]  # (B, d_model)

        # Regression
        return self.regressor(cls_repr)
