"""
Adaptive Trading Network — the base learner for MAML.

Architecture: Input projection → LSTM → Classification head.

Kept deliberately small (~30-50K params) because:
  - MAML inner-loop adaptation works best with compact models
  - Per-ticker training sets are only ~1500 days
  - Smaller models generalize better in low-data regimes
"""

import torch
import torch.nn as nn


class AdaptiveTradingNetwork(nn.Module):
    """
    LSTM-based classifier for trading signals.

    Input:  (batch, sequence_length, input_dim)
    Output: (batch, output_dim) — logits for signal classes
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        output_dim: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # ── Input Projection ────────────────────────────────────────
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # ── LSTM Encoder ────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # ── Classification Head ─────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        lstm_out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        return self.head(last_hidden)
