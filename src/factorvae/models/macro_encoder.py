"""
MacroEncoder — GRU-based encoder for macro context windows.

Receives a (T, M) window of T days of M macro features and produces
a single conditioning vector h_macro ∈ R^(H_macro).

Architecture (macro_encoder_plano_v2.md, Parte 3):
    Linear(M → H_macro) + LeakyReLU
    GRU(1 layer, unidirectional, batch_first)
    Dropout + LayerNorm on the last hidden state h_T

Justification: FactorVAE original (Duan et al. 2022) uses GRU as its
FeatureExtractor. Transformers overfit on limited financial time-series
data (MDPI 2025). Keeping GRU ensures architectural parity and defends
the design choice in front of a committee.

Parameter count (n_macro=8, hidden_dim=16):
    proj   :  8×16 + 16 =   144
    GRU    : 3×(16×16 + 16×16 + 16) = 1 584
    LayerNorm: 32
    Total  : ~1 760  (vs. ~20 000 for a 2-layer Transformer)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class MacroEncoder(nn.Module):
    """
    GRU encoder for a T-day macro window.

    Args:
        n_macro    : number of macro features M (raw input width)
        hidden_dim : GRU hidden size = output dimension H_macro
        dropout    : dropout applied to the final hidden state
        leaky_slope: negative slope for LeakyReLU in the input projection
    """

    def __init__(
        self,
        n_macro: int,
        hidden_dim: int = 16,
        dropout: float = 0.1,
        leaky_slope: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(n_macro, hidden_dim)
        self.act  = nn.LeakyReLU(leaky_slope)
        self.gru  = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.ln   = nn.LayerNorm(hidden_dim)

    def forward(self, m: Tensor) -> Tensor:
        """
        Args:
            m: (T, M)  macro window — T timesteps, M features
        Returns:
            h_macro: (H_macro,)  — single conditioning vector
        """
        x = self.act(self.proj(m))       # (T, H_macro)
        x = x.unsqueeze(0)               # (1, T, H_macro)  — batch dim for GRU
        _, h_T = self.gru(x)             # h_T: (1, 1, H_macro)
        h = h_T.squeeze(0).squeeze(0)   # (H_macro,)
        return self.ln(self.drop(h))
