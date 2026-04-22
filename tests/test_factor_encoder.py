"""Tests for FactorEncoder."""

from __future__ import annotations

import torch
from factorvae.models.factor_encoder import FactorEncoder


def test_encoder_output_shapes(dims):
    N, H, K, M = dims["N"], dims["H"], dims["K"], dims["M"]
    encoder = FactorEncoder(H, M, K)
    y = torch.randn(N)
    e = torch.randn(N, H)
    mu, sigma = encoder(y, e)
    assert mu.shape == (K,), f"Expected ({K},), got {mu.shape}"
    assert sigma.shape == (K,), f"Expected ({K},), got {sigma.shape}"


def test_encoder_sigma_positive(dims):
    N, H, K, M = dims["N"], dims["H"], dims["K"], dims["M"]
    encoder = FactorEncoder(H, M, K)
    y = torch.randn(N)
    e = torch.randn(N, H)
    _, sigma = encoder(y, e)
    assert (sigma > 0).all(), "sigma_post must be strictly positive"


def test_encoder_n_invariance(dims):
    H, K, M = dims["H"], dims["K"], dims["M"]
    encoder = FactorEncoder(H, M, K)
    encoder.eval()
    for N in [50, 500]:
        y = torch.randn(N)
        e = torch.randn(N, H)
        with torch.no_grad():
            mu, sigma = encoder(y, e)
        assert mu.shape == (K,)
        assert sigma.shape == (K,)
