"""
Factor Predictor (phi_pred).

Section 2.3 of the scaffolding / Eq. 13-16 of the paper.

Architecture:
  - K independent SingleHeadAttention heads, each with a learnable query q.
    Attention uses cosine similarity with ReLU (not softmax).
  - One shared DistributionNetwork applied head-wise to produce
    (mu_prior^(k), sigma_prior^(k)) per head.
  - Outputs: (mu_prior, sigma_prior) both shape (K,).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SingleHeadAttention(nn.Module):
    """
    Single attention head using cosine similarity with ReLU gate and
    a learnable query vector q.

    k^(i) = w_key  @ e^(i)
    v^(i) = w_value @ e^(i)
    a^(i) = max(0, cos_sim(q, k^(i))) / sum_j max(0, cos_sim(q, k^(j)))
    h_att = sum_i a^(i) * v^(i)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.q = nn.Parameter(torch.randn(hidden_dim) * 0.1)

    def forward(self, e: Tensor) -> Tensor:
        """
        Args:
            e: (N, H)
        Returns:
            h_att: (H,)  global market representation for this head
        """
        k = self.key(e)    # (N, H)
        v = self.value(e)  # (N, H)

        # Cosine similarity: q·k^(i) / (||q||_2 * ||k^(i)||_2)
        q_norm = F.normalize(self.q.unsqueeze(0), dim=-1)  # (1, H)
        k_norm = F.normalize(k, dim=-1)                     # (N, H)
        scores = (q_norm * k_norm).sum(dim=-1)              # (N,)

        # ReLU gate: set negative scores to zero
        scores = scores.clamp(min=0.0)

        # Normalize (safe: if all scores are zero, fall back to uniform)
        score_sum = scores.sum()
        if score_sum < 1e-9:
            weights = torch.ones_like(scores) / scores.shape[0]
        else:
            weights = scores / score_sum                    # (N,)

        h_att = (weights.unsqueeze(-1) * v).sum(dim=0)     # (H,)
        return h_att


class DistributionNetwork(nn.Module):
    """
    Shared distribution network applied to each of the K attention heads.

    h_prior^(k) = LeakyReLU(w_pri @ h_muti^(k) + b_pri)
    mu_prior^(k)    = w_mu  @ h_prior^(k) + b_mu
    sigma_prior^(k) = Softplus(w_sigma @ h_prior^(k) + b_sigma)

    Applied to input of shape (K, H); produces (K,) outputs.
    Weights are shared across all K heads.
    """

    def __init__(self, hidden_dim: int, leaky_slope: float = 0.1):
        super().__init__()
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.LeakyReLU(negative_slope=leaky_slope)
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.sigma_head = nn.Linear(hidden_dim, 1)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            h: (K, H)  stacked head representations
        Returns:
            mu:    (K,)
            sigma: (K,)  strictly positive via Softplus
        """
        h_hidden = self.act(self.hidden(h))            # (K, H)
        mu = self.mu_head(h_hidden).squeeze(-1)        # (K,)
        sigma = F.softplus(self.sigma_head(h_hidden)).squeeze(-1)  # (K,)
        return mu, sigma


class FactorPredictor(nn.Module):
    """
    Factor Predictor phi_pred.

    K independent attention heads produce h_muti in (K, H).
    Shared DistributionNetwork maps each head to (mu_k, sigma_k).
    """

    def __init__(self, hidden_dim: int, num_factors: int, leaky_slope: float = 0.1):
        super().__init__()
        self.heads = nn.ModuleList(
            [SingleHeadAttention(hidden_dim) for _ in range(num_factors)]
        )
        self.dist_net = DistributionNetwork(hidden_dim, leaky_slope)

    def forward(self, e: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            e: (N, H) stock embeddings
        Returns:
            mu_prior:    (K,)
            sigma_prior: (K,)
        """
        h_muti = torch.stack([head(e) for head in self.heads], dim=0)  # (K, H)
        return self.dist_net(h_muti)
