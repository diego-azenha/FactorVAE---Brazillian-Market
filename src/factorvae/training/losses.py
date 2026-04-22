"""
Loss functions for FactorVAE training.

Thin wrappers over the pure functions in distributions.py.
Separate logging of reconstruction and KL terms is critical for diagnosis.
"""

from __future__ import annotations

from torch import Tensor

from factorvae.models.distributions import gaussian_nll, kl_gaussian_diagonal


def reconstruction_loss(
    y: Tensor,
    mu_y_rec: Tensor,
    sigma_y_rec: Tensor,
    floor: float = 1e-6,
) -> Tensor:
    """
    Gaussian NLL of observed returns under the reconstructed distribution.
    Averaged over the N stocks in the cross-section.
    """
    return gaussian_nll(y, mu_y_rec, sigma_y_rec, floor)


def kl_loss(
    mu_post: Tensor,
    sigma_post: Tensor,
    mu_prior: Tensor,
    sigma_prior: Tensor,
    floor: float = 1e-6,
) -> Tensor:
    """
    KL(posterior || prior) for the factor distribution.
    q = posterior (encoder), p = prior (predictor). Do not swap.
    Summed over K factors.
    """
    return kl_gaussian_diagonal(mu_post, sigma_post, mu_prior, sigma_prior, floor)
