"""
Deep LSTM Stack for Sprint-Sequence Regression
Multi-layer bidirectional LSTM with attention pooling.
"""

import torch
import torch.nn as nn
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import lstm_cfg
from models.common import InputProjection, RegressionHead, create_padding_mask


class AttentionPooling(nn.Module):
    """Learned attention pooling over LSTM hidden states."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, hidden_states: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) — True = padded

        Returns:
            (batch, hidden_dim)
        """
        attn_scores = self.attention(hidden_states).squeeze(-1)  # (B, L)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)        # (B, L)
        context = torch.bmm(attn_weights.unsqueeze(1),
                            hidden_states).squeeze(1)             # (B, H)
        return context


class DeepLSTMStack(nn.Module):
    """
    Multi-layer bidirectional LSTM with:
    - Input projection
    - Stacked LSTM layers with residual connections
    - Attention pooling over hidden states
    - MLP regression head
    """

    def __init__(self, input_dim: int = None,
                 hidden_dim: int = None,
                 num_layers: int = None,
                 dropout: float = None,
                 bidirectional: bool = None):
        super().__init__()
        cfg = lstm_cfg
        self.input_dim = input_dim or cfg.input_dim
        self.hidden_dim = hidden_dim or cfg.hidden_dim
        self.num_layers = num_layers or cfg.num_layers
        self.dropout_rate = dropout or cfg.dropout
        self.bidirectional = bidirectional if bidirectional is not None \
            else cfg.bidirectional

        self.directions = 2 if self.bidirectional else 1
        lstm_output_dim = self.hidden_dim * self.directions

        # Input projection
        self.input_proj = InputProjection(
            self.input_dim, self.hidden_dim, self.dropout_rate)

        # Stacked LSTM
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
        )

        # Layer norm after LSTM
        self.layer_norm = nn.LayerNorm(lstm_output_dim)

        # Attention pooling
        self.attention_pool = AttentionPooling(lstm_output_dim)

        # Regression head
        self.regressor = RegressionHead(
            lstm_output_dim, lstm_output_dim // 2, self.dropout_rate)

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

        # Project input
        x = self.input_proj(x)  # (B, L, hidden)

        # Pack for efficiency if lengths provided
        if lengths is not None:
            lengths_cpu = lengths.cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=seq_len)
        else:
            lstm_out, _ = self.lstm(x)

        # Layer norm
        lstm_out = self.layer_norm(lstm_out)

        # Attention pooling
        mask = None
        if lengths is not None:
            mask = create_padding_mask(lengths, seq_len)

        context = self.attention_pool(lstm_out, mask)

        # Regression
        return self.regressor(context)
