"""
Factor Encoder (phi_enc).

Section 2.2 of the scaffolding / Eq. 6-8 of the paper.

Two submodules:
  - PortfolioLayer: builds M dynamic portfolios from embeddings e, weighted by softmax
    over stocks (dim=0). Returns portfolio returns y_p of shape (M,).
  - MappingLayer: maps y_p -> (mu_post, sigma_post), both shape (K,).

CRITICAL: softmax in PortfolioLayer must normalize over stocks (dim=0), NOT over
portfolios (dim=1). The test test_softmax_dim_correctness enforces this.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PortfolioLayer(nn.Module):
    """
    Constructs M dynamically-weighted portfolios from stock embeddings and returns.

    Portfolio weights a_p^(i,j) = softmax over i (stocks) of (w_p @ e^(i) + b_p)^(j).
    Portfolio returns y_p^(j) = sum_i y^(i) * a_p^(i,j).
    """

    def __init__(self, hidden_dim: int, num_portfolios: int):
        super().__init__()
        # Projects each stock embedding (H,) to portfolio scores (M,)
        self.linear = nn.Linear(hidden_dim, num_portfolios)

    def forward(self, y: Tensor, e: Tensor) -> Tensor:
        """
        Args:
            y: (N,)   stock returns
            e: (N, H) stock embeddings
        Returns:
            y_p: (M,) portfolio returns
        """
        # scores: (N, M)
        scores = self.linear(e)
        # softmax over stocks (dim=0) so weights sum to 1 for each portfolio
        weights = F.softmax(scores, dim=0)  # (N, M)
        # weighted sum of returns: y^T @ weights -> (M,)
        y_p = y @ weights  # (M,)
        return y_p


class MappingLayer(nn.Module):
    """
    Maps portfolio returns y_p (M,) to posterior factor distribution (K,).
    """

    def __init__(self, num_portfolios: int, num_factors: int):
        super().__init__()
        self.mu_head = nn.Linear(num_portfolios, num_factors)
        self.sigma_head = nn.Linear(num_portfolios, num_factors)

    def forward(self, y_p: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            y_p: (M,)
        Returns:
            mu_post:    (K,)
            sigma_post: (K,)  strictly positive via Softplus
        """
        mu = self.mu_head(y_p)                           # (K,)
        sigma = F.softplus(self.sigma_head(y_p))         # (K,)
        return mu, sigma


class FactorEncoder(nn.Module):
    """
    Factor Encoder phi_enc.

    Combines PortfolioLayer and MappingLayer to map (y, e) -> (mu_post, sigma_post).
    Only used during training.
    """

    def __init__(self, hidden_dim: int, num_portfolios: int, num_factors: int):
        super().__init__()
        self.portfolio = PortfolioLayer(hidden_dim, num_portfolios)
        self.mapping = MappingLayer(num_portfolios, num_factors)

    def forward(self, y: Tensor, e: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            y: (N,)   future stock returns (oracle signal, train only)
            e: (N, H) stock embeddings from feature extractor
        Returns:
            mu_post:    (K,)
            sigma_post: (K,)
        """
        y_p = self.portfolio(y, e)
        return self.mapping(y_p)
