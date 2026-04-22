"""Tests for FactorDecoder."""

from __future__ import annotations

import torch
from factorvae.models.factor_decoder import FactorDecoder


def test_decoder_output_shapes(dims):
    N, H, K = dims["N"], dims["H"], dims["K"]
    decoder = FactorDecoder(H, K)
    mu_z = torch.randn(K)
    sigma_z = torch.rand(K) + 0.1
    e = torch.randn(N, H)
    mu_y, sigma_y = decoder(mu_z, sigma_z, e)
    assert mu_y.shape == (N,), f"Expected ({N},), got {mu_y.shape}"
    assert sigma_y.shape == (N,), f"Expected ({N},), got {sigma_y.shape}"


def test_decoder_sigma_positive(dims):
    N, H, K = dims["N"], dims["H"], dims["K"]
    decoder = FactorDecoder(H, K)
    mu_z = torch.randn(K)
    sigma_z = torch.rand(K) + 0.1
    e = torch.randn(N, H)
    _, sigma_y = decoder(mu_z, sigma_z, e)
    assert (sigma_y > 0).all(), "sigma_y must be strictly positive"


def test_decoder_matches_sampled_moments(dims):
    """
    For known (mu_alpha, sigma_alpha, beta, mu_z, sigma_z),
    analytic (mu_y, sigma_y) should match Monte Carlo moments within 1e-2.
    """
    N, H, K = 20, dims["H"], dims["K"]
    torch.manual_seed(42)
    decoder = FactorDecoder(H, K)
    decoder.eval()

    mu_z = torch.randn(K)
    sigma_z = torch.rand(K) * 0.5 + 0.1
    e = torch.randn(N, H)

    with torch.no_grad():
        mu_y, sigma_y = decoder(mu_z, sigma_z, e)
        # Extract alpha and beta directly for MC sampling
        mu_alpha, sigma_alpha = decoder.alpha(e)
        beta = decoder.beta(e)

    n_samples = 100_000
    alpha_samples = mu_alpha + sigma_alpha * torch.randn(n_samples, N)
    z_samples = mu_z + sigma_z * torch.randn(n_samples, K)
    y_samples = alpha_samples + z_samples @ beta.T  # (n_samples, N)

    mc_mu = y_samples.mean(dim=0)
    mc_std = y_samples.std(dim=0)

    assert torch.allclose(mu_y, mc_mu, atol=1e-2), f"mu_y mismatch max: {(mu_y - mc_mu).abs().max():.4f}"
    assert torch.allclose(sigma_y, mc_std, atol=1e-2), f"sigma_y mismatch max: {(sigma_y - mc_std).abs().max():.4f}"
