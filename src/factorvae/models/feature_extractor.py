"""
Feature Extractor (phi_feat).

Section 2.1 of the scaffolding / Eq. 4 of the paper.

Architecture:
  1. Per-timestep linear projection with LeakyReLU: (C -> H)
  2. GRU over T timesteps: hidden size H
  3. Output: last hidden state -> (N, H)

Weights are shared across all N tickers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class FeatureExtractor(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int, leaky_slope: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(num_features, hidden_dim)
        self.act = nn.LeakyReLU(negative_slope=leaky_slope)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (N, T, C)
        Returns:
            e: (N, H)
        """
        # Project each timestep: (N, T, C) -> (N, T, H)
        h_proj = self.act(self.proj(x))
        # GRU: (N, T, H) -> output (N, T, H), h_n (1, N, H)
        _, h_n = self.gru(h_proj)
        return h_n.squeeze(0)  # (N, H)
