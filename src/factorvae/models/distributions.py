"""
Pure mathematical functions used throughout FactorVAE.
No nn.Module — fully testable in isolation.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def compose_return(
    mu_alpha: Tensor,
    sigma_alpha: Tensor,
    beta: Tensor,
    mu_z: Tensor,
    sigma_z: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Analytic composition of Gaussian alpha + beta @ z.

    Eq. 12 from the paper: because alpha ~ N(mu_alpha, sigma_alpha^2) and
    z ~ N(mu_z, sigma_z^2) are independent Gaussians and beta is deterministic,
    y^(i) = alpha^(i) + sum_k beta^(i,k) * z^(k) is also Gaussian with:
        mu_y^(i)    = mu_alpha^(i) + beta^(i,:) @ mu_z
        sigma_y^(i) = sqrt(sigma_alpha^(i)^2 + sum_k (beta^(i,k))^2 * sigma_z^(k)^2)

    Args:
        mu_alpha:    (N,)
        sigma_alpha: (N,)  strictly positive
        beta:        (N, K)
        mu_z:        (K,)
        sigma_z:     (K,)  strictly positive

    Returns:
        mu_y:    (N,)
        sigma_y: (N,)
    """
    mu_y = mu_alpha + beta @ mu_z                        # (N,)
    var_factor = (beta ** 2) @ (sigma_z ** 2)            # (N,)
    sigma_y = torch.sqrt(sigma_alpha ** 2 + var_factor)  # (N,)
    return mu_y, sigma_y


def gaussian_nll(
    y: Tensor,
    mu: Tensor,
    sigma: Tensor,
    floor: float = 1e-6,
) -> Tensor:
    """
    Gaussian negative log-likelihood, averaged over elements.

    -log P(y | mu, sigma) = 0.5 * [log(2*pi*sigma^2) + (y-mu)^2/sigma^2]

    Args:
        y:     (N,)
        mu:    (N,)
        sigma: (N,)  clamped to floor before use
        floor: minimum sigma value for numerical stability

    Returns:
        scalar Tensor
    """
    sigma = sigma.clamp(min=floor)
    log_term = torch.log(2.0 * math.pi * sigma ** 2)
    sq_term = (y - mu) ** 2 / (sigma ** 2)
    return 0.5 * (log_term + sq_term).mean()


def kl_gaussian_diagonal(
    mu_q: Tensor,
    sigma_q: Tensor,
    mu_p: Tensor,
    sigma_p: Tensor,
    floor: float = 1e-6,
) -> Tensor:
    """
    KL(N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2)), summed over all dimensions.

    Closed-form for diagonal Gaussians:
        KL = log(sigma_p/sigma_q) + (sigma_q^2 + (mu_q-mu_p)^2)/(2*sigma_p^2) - 0.5

    IMPORTANT: q is the posterior (from encoder), p is the learned prior (from predictor).
    Swapping arguments inverts the gradient of the predictor.

    Args:
        mu_q, sigma_q: posterior parameters, shape (K,)
        mu_p, sigma_p: prior parameters,     shape (K,)
        floor: minimum sigma value

    Returns:
        scalar Tensor (summed, not averaged, over K)
    """
    sigma_q = sigma_q.clamp(min=floor)
    sigma_p = sigma_p.clamp(min=floor)
    term1 = torch.log(sigma_p / sigma_q)
    term2 = (sigma_q ** 2 + (mu_q - mu_p) ** 2) / (2.0 * sigma_p ** 2)
    return (term1 + term2 - 0.5).sum()
