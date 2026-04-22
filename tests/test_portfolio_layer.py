"""Tests for PortfolioLayer — the most sensitive component in FactorVAE."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from factorvae.models.factor_encoder import PortfolioLayer


def test_portfolio_weights_sum_to_one(dims):
    N, H, M = dims["N"], dims["H"], dims["M"]
    layer = PortfolioLayer(H, M)
    e = torch.randn(N, H)
    # Expose the raw weights directly
    scores = layer.linear(e)               # (N, M)
    weights = F.softmax(scores, dim=0)    # (N, M) — softmax over stocks
    col_sums = weights.sum(dim=0)         # (M,)
    assert torch.allclose(col_sums, torch.ones(M), atol=1e-5), (
        f"Weights don't sum to 1 per portfolio. Max error: {(col_sums - 1).abs().max():.2e}"
    )


def test_portfolio_weights_nonnegative(dims):
    N, H, M = dims["N"], dims["H"], dims["M"]
    layer = PortfolioLayer(H, M)
    e = torch.randn(N, H)
    scores = layer.linear(e)
    weights = F.softmax(scores, dim=0)
    assert (weights >= 0).all(), "Portfolio weights must be non-negative"


def test_portfolio_output_shape_invariant_to_n(dims):
    H, M = dims["H"], dims["M"]
    layer = PortfolioLayer(H, M)
    for N in [50, 100, 500]:
        y = torch.randn(N)
        e = torch.randn(N, H)
        y_p = layer(y, e)
        assert y_p.shape == (M,), f"Expected ({M},), got {y_p.shape} for N={N}"


def test_softmax_dim_correctness(dims):
    """
    Patholological test: if softmax were applied over portfolios (dim=1) instead
    of stocks (dim=0), the output would be very different from the correct one.
    We construct an asymmetric case and verify the correct behavior.
    """
    H, M = dims["H"], dims["M"]
    N = 10  # small N so the difference is clear
    layer = PortfolioLayer(H, M)
    layer.eval()

    y = torch.ones(N)
    e = torch.randn(N, H)

    with torch.no_grad():
        scores = layer.linear(e)  # (N, M)

        # Correct: softmax over dim=0 (stocks), then y @ weights
        weights_correct = F.softmax(scores, dim=0)  # (N, M)
        y_p_correct = y @ weights_correct            # (M,)

        # Wrong: softmax over dim=1 (portfolios) — each stock's scores normalized
        weights_wrong = F.softmax(scores, dim=1)    # (N, M) — wrong normalization
        y_p_wrong = y @ weights_wrong               # (M,)

    # Forward pass should match correct version
    y_p_model = layer(y, e)
    assert torch.allclose(y_p_model, y_p_correct, atol=1e-5), (
        "PortfolioLayer output does not match softmax over stocks (dim=0)"
    )

    # The two versions must differ (otherwise the test is trivial)
    assert not torch.allclose(y_p_correct, y_p_wrong, atol=1e-5), (
        "Correct and wrong softmax dim produce same result — test is degenerate"
    )
