"""
Hybrid CNN + Transformer Model
1D CNN extracts local sprint patterns → Transformer captures long-range
dependencies → Regression head predicts effort.
"""

import torch
import torch.nn as nn
from typing import Optional, List

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import hybrid_cfg
from models.common import (PositionalEncoding, RegressionHead,
                            create_padding_mask)


class Conv1DBlock(nn.Module):
    """1D convolutional block with batch norm and residual connection."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Residual projection if dimensions change
        self.residual = (nn.Conv1d(in_channels, out_channels, 1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, seq_len)"""
        residual = self.residual(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        return x + residual


class CNNFeatureExtractor(nn.Module):
    """Multi-layer 1D CNN for local sprint pattern extraction."""

    def __init__(self, input_dim: int,
                 channels: List[int] = None,
                 kernel_size: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        channels = channels or [64, 128, 256]

        layers = []
        in_ch = input_dim
        for out_ch in channels:
            layers.append(Conv1DBlock(in_ch, out_ch, kernel_size, dropout))
            in_ch = out_ch

        self.cnn = nn.Sequential(*layers)
        self.output_dim = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch, seq_len, output_dim)
        """
        x = x.transpose(1, 2)   # (B, C, L)
        x = self.cnn(x)
        x = x.transpose(1, 2)   # (B, L, C)
        return x


class CNNTransformerHybrid(nn.Module):
    """
    Hybrid architecture:
        1. 1D CNN extracts local sprint-level patterns
        2. Transformer encoder captures long-range temporal dependencies
        3. CLS-pooled output → regression head

    CNN handles micro-patterns (adjacent sprint features),
    Transformer handles macro-patterns (project-level trends).
    """

    def __init__(self, input_dim: int = None,
                 cnn_channels: List[int] = None,
                 cnn_kernel_size: int = None,
                 d_model: int = None,
                 nhead: int = None,
                 num_transformer_layers: int = None,
                 dim_feedforward: int = None,
                 dropout: float = None,
                 max_seq_len: int = None):
        super().__init__()
        cfg = hybrid_cfg
        input_dim = input_dim or cfg.input_dim
        cnn_channels = cnn_channels or cfg.cnn_channels
        cnn_kernel = cnn_kernel_size or cfg.cnn_kernel_size
        self.d_model = d_model or cfg.d_model
        nhead = nhead or cfg.nhead
        n_trans = num_transformer_layers or cfg.num_transformer_layers
        dim_ff = dim_feedforward or cfg.dim_feedforward
        drop = dropout or cfg.dropout
        max_len = max_seq_len or cfg.max_seq_len

        # CNN feature extractor
        self.cnn = CNNFeatureExtractor(
            input_dim, cnn_channels, cnn_kernel, drop)

        # Project CNN output to transformer dimension
        self.cnn_to_transformer = nn.Sequential(
            nn.Linear(self.cnn.output_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(drop),
        )

        # CLS token
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, self.d_model) * 0.02)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            self.d_model, max_len=max_len + 1, dropout=drop)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=drop,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_trans)

        # Regression head
        self.regressor = RegressionHead(
            self.d_model, self.d_model // 2, drop)

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

        # CNN feature extraction
        cnn_out = self.cnn(x)                    # (B, L, cnn_out_dim)
        x = self.cnn_to_transformer(cnn_out)     # (B, L, d_model)

        # Prepend CLS
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)           # (B, L+1, d_model)

        # Positional encoding
        x = self.pos_encoding(x)

        # Padding mask
        mask = None
        if lengths is not None:
            mask = create_padding_mask(lengths + 1, seq_len + 1)

        # Transformer
        x = self.transformer(x, src_key_padding_mask=mask)

        # CLS pooling
        cls_repr = x[:, 0, :]

        # Regression
        return self.regressor(cls_repr)
