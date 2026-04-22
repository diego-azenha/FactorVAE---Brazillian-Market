"""
Factor Decoder (phi_dec).

Section 2.4 of the scaffolding / Eq. 9-12 of the paper.

Two submodules:
  - AlphaLayer: e -> (mu_alpha, sigma_alpha), both shape (N,)
  - BetaLayer:  e -> beta, shape (N, K)

Composition is ANALYTIC (no reparameterization trick, no Monte Carlo).
See compose_return in distributions.py for the closed-form formulas.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from factorvae.models.distributions import compose_return


class AlphaLayer(nn.Module):
    """
    Produces idiosyncratic return distribution: alpha ~ N(mu_alpha, sigma_alpha^2).

    h_alpha = LeakyReLU(w_alpha @ e + b_alpha)           (N, H)
    mu_alpha    = w_mu @ h_alpha + b_mu                  (N,)
    sigma_alpha = Softplus(w_sigma @ h_alpha + b_sigma)  (N,)
    """

    def __init__(self, hidden_dim: int, leaky_slope: float = 0.1):
        super().__init__()
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.LeakyReLU(negative_slope=leaky_slope)
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.sigma_head = nn.Linear(hidden_dim, 1)

    def forward(self, e: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            e: (N, H)
        Returns:
            mu_alpha:    (N,)
            sigma_alpha: (N,)  strictly positive
        """
        h = self.act(self.hidden(e))                     # (N, H)
        mu = self.mu_head(h).squeeze(-1)                 # (N,)
        sigma = F.softplus(self.sigma_head(h)).squeeze(-1)  # (N,)
        return mu, sigma


class BetaLayer(nn.Module):
    """
    Produces factor exposure matrix: beta = w_beta @ e + b_beta.
    Linear, no activation.
    """

    def __init__(self, hidden_dim: int, num_factors: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_factors)

    def forward(self, e: Tensor) -> Tensor:
        """
        Args:
            e: (N, H)
        Returns:
            beta: (N, K)
        """
        return self.linear(e)


class FactorDecoder(nn.Module):
    """
    Factor Decoder phi_dec.

    Converts (mu_z, sigma_z, e) into (mu_y, sigma_y) analytically.
    Never samples — uses the closed-form Gaussian composition (Eq. 12).
    """

    def __init__(self, hidden_dim: int, num_factors: int, leaky_slope: float = 0.1):
        super().__init__()
        self.alpha = AlphaLayer(hidden_dim, leaky_slope)
        self.beta = BetaLayer(hidden_dim, num_factors)

    def forward(
        self,
        mu_z: Tensor,
        sigma_z: Tensor,
        e: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            mu_z:    (K,)
            sigma_z: (K,)
            e:       (N, H)
        Returns:
            mu_y:    (N,)
            sigma_y: (N,)
        """
        mu_alpha, sigma_alpha = self.alpha(e)
        beta = self.beta(e)                              # (N, K)
        return compose_return(mu_alpha, sigma_alpha, beta, mu_z, sigma_z)
