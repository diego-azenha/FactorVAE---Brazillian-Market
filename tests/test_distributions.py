"""
Tests for src/factorvae/models/distributions.py.

Covers: compose_return (shapes + analytic moments vs Monte Carlo),
        gaussian_nll (vs scipy), kl_gaussian_diagonal (closed-form properties).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.stats
import torch

from factorvae.models.distributions import (
    compose_return,
    gaussian_nll,
    kl_gaussian_diagonal,
)


# ─────────────────────────────────────────────
# compose_return
# ─────────────────────────────────────────────

def test_compose_return_shapes(dims):
    N, K = dims["N"], dims["K"]
    torch.manual_seed(0)
    mu_alpha = torch.randn(N)
    sigma_alpha = torch.rand(N) + 0.01
    beta = torch.randn(N, K)
    mu_z = torch.randn(K)
    sigma_z = torch.rand(K) + 0.01

    mu_y, sigma_y = compose_return(mu_alpha, sigma_alpha, beta, mu_z, sigma_z)

    assert mu_y.shape == (N,), f"Expected (N,), got {mu_y.shape}"
    assert sigma_y.shape == (N,), f"Expected (N,), got {sigma_y.shape}"


def test_compose_return_matches_sampled_moments(dims):
    """
    Analytic moments should match Monte Carlo moments within tolerance.
    """
    N, K = dims["N"], dims["K"]
    torch.manual_seed(5)
    mu_alpha = torch.randn(N)
    sigma_alpha = torch.rand(N) * 0.5 + 0.1
    beta = torch.randn(N, K) * 0.5
    mu_z = torch.randn(K)
    sigma_z = torch.rand(K) * 0.5 + 0.1

    mu_y, sigma_y = compose_return(mu_alpha, sigma_alpha, beta, mu_z, sigma_z)

    # Monte Carlo with 100_000 samples
    n_samples = 100_000
    alpha_samples = mu_alpha.unsqueeze(0) + sigma_alpha.unsqueeze(0) * torch.randn(n_samples, N)
    z_samples = mu_z.unsqueeze(0) + sigma_z.unsqueeze(0) * torch.randn(n_samples, K)
    y_samples = alpha_samples + z_samples @ beta.T   # (n_samples, N)

    mc_mu = y_samples.mean(dim=0)
    mc_std = y_samples.std(dim=0)

    assert torch.allclose(mu_y, mc_mu, atol=1e-2), f"Max mean err: {(mu_y - mc_mu).abs().max():.4f}"
    assert torch.allclose(sigma_y, mc_std, atol=1e-2), f"Max std err: {(sigma_y - mc_std).abs().max():.4f}"


def test_compose_return_variance_uses_beta_squared(dims):
    """
    Verify that sigma_y^2 = sigma_alpha^2 + sum_k beta^(i,k)^2 * sigma_z^(k)^2,
    NOT |beta| * sigma_z (a common bug).
    """
    N, K = 3, 2
    mu_alpha = torch.zeros(N)
    sigma_alpha = torch.ones(N)
    beta = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0]])
    mu_z = torch.zeros(K)
    sigma_z = torch.ones(K)

    _, sigma_y = compose_return(mu_alpha, sigma_alpha, beta, mu_z, sigma_z)
    expected_var = sigma_alpha ** 2 + (beta ** 2) @ (sigma_z ** 2)
    expected_sigma = expected_var.sqrt()

    assert torch.allclose(sigma_y, expected_sigma, atol=1e-6)


# ─────────────────────────────────────────────
# gaussian_nll
# ─────────────────────────────────────────────

def test_gaussian_nll_matches_scipy():
    torch.manual_seed(7)
    N = 50
    y = torch.randn(N)
    mu = torch.randn(N)
    sigma = torch.rand(N) * 2.0 + 0.1

    nll_ours = gaussian_nll(y, mu, sigma).item()

    # scipy reference: sum of -logpdf, averaged
    nll_scipy = -np.mean(
        scipy.stats.norm.logpdf(y.numpy(), loc=mu.numpy(), scale=sigma.numpy())
    )

    assert abs(nll_ours - nll_scipy) < 1e-5, f"NLL mismatch: ours={nll_ours:.6f}  scipy={nll_scipy:.6f}"


def test_gaussian_nll_positive():
    y = torch.zeros(10)
    mu = torch.zeros(10)
    sigma = torch.ones(10)
    # NLL at sigma=1, y=mu: 0.5*log(2*pi) > 0
    assert gaussian_nll(y, mu, sigma).item() > 0


def test_gaussian_nll_floor_prevents_nan():
    y = torch.randn(5)
    mu = torch.randn(5)
    sigma = torch.zeros(5)  # would cause division by zero without floor
    result = gaussian_nll(y, mu, sigma)
    assert not torch.isnan(result), "NLL should not be NaN with zero sigma due to floor"


# ─────────────────────────────────────────────
# kl_gaussian_diagonal
# ─────────────────────────────────────────────

def test_kl_zero_for_identical_distributions():
    torch.manual_seed(3)
    K = 8
    mu = torch.randn(K)
    sigma = torch.rand(K) + 0.1
    kl = kl_gaussian_diagonal(mu, sigma, mu, sigma)
    assert kl.item() < 1e-5, f"KL(p, p) should be ~0, got {kl.item()}"


def test_kl_nonnegative():
    torch.manual_seed(4)
    K = 8
    for _ in range(10):
        mu_q = torch.randn(K)
        sigma_q = torch.rand(K) + 0.1
        mu_p = torch.randn(K)
        sigma_p = torch.rand(K) + 0.1
        kl = kl_gaussian_diagonal(mu_q, sigma_q, mu_p, sigma_p)
        assert kl.item() >= -1e-6, f"KL must be non-negative, got {kl.item()}"


def test_kl_matches_monte_carlo():
    """
    KL should match Monte Carlo estimate (100k samples) within 1e-2.
    """
    torch.manual_seed(9)
    K = 4
    mu_q = torch.tensor([1.0, -0.5, 0.2, 0.8])
    sigma_q = torch.tensor([0.5, 1.2, 0.3, 0.9])
    mu_p = torch.tensor([0.0, 0.0, 0.0, 0.0])
    sigma_p = torch.tensor([1.0, 1.0, 1.0, 1.0])

    kl_analytic = kl_gaussian_diagonal(mu_q, sigma_q, mu_p, sigma_p).item()

    # MC estimate: E_q[log q(z) - log p(z)]
    n = 100_000
    z = mu_q + sigma_q * torch.randn(n, K)
    log_q = scipy.stats.norm.logpdf(z.numpy(), loc=mu_q.numpy(), scale=sigma_q.numpy()).sum(axis=1)
    log_p = scipy.stats.norm.logpdf(z.numpy(), loc=mu_p.numpy(), scale=sigma_p.numpy()).sum(axis=1)
    kl_mc = float(np.mean(log_q - log_p))

    assert abs(kl_analytic - kl_mc) < 1e-2, f"KL analytic={kl_analytic:.4f}  MC={kl_mc:.4f}"


def test_kl_argument_order():
    """
    KL(q||p) != KL(p||q) in general. Confirm order matters.
    """
    torch.manual_seed(11)
    K = 4
    mu_q = torch.randn(K)
    sigma_q = torch.rand(K) + 0.1
    mu_p = torch.zeros(K)
    sigma_p = torch.ones(K)

    kl_qp = kl_gaussian_diagonal(mu_q, sigma_q, mu_p, sigma_p).item()
    kl_pq = kl_gaussian_diagonal(mu_p, sigma_p, mu_q, sigma_q).item()

    assert abs(kl_qp - kl_pq) > 1e-4, "KL(q||p) should differ from KL(p||q)"
