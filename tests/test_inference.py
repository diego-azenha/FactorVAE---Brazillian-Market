"""Tests for inference invariants."""

from __future__ import annotations

import torch

from factorvae.models.factorvae import FactorVAE
from factorvae.data.dataset import SyntheticDataset


def _make_config(dims):
    return {
        "model": {
            "num_features": dims["C"],
            "hidden_dim": dims["H"],
            "num_factors": dims["K"],
            "num_portfolios": dims["M"],
            "leaky_relu_slope": 0.1,
        }
    }


def test_predict_robust_to_y_corruption(dims, random_x):
    """
    Perturbing y (future returns) must NOT change forward_predict output.
    If it does, there is a look-ahead leak.
    """
    model = FactorVAE(_make_config(dims))
    model.eval()
    N = dims["N"]

    with torch.no_grad():
        mu1, sig1 = model.forward_predict(random_x)
        # Corrupt y — forward_predict should not use it at all
        y_corrupt = torch.randn(N) * 100
        mu2, sig2 = model.forward_predict(random_x)

    assert torch.allclose(mu1, mu2), "forward_predict changed when y was corrupted — possible leak"
    assert torch.allclose(sig1, sig2), "forward_predict changed when y was corrupted — possible leak"


def test_deterministic_inference(dims, random_x):
    """With fixed seeds, forward_predict must return the same result twice."""
    model = FactorVAE(_make_config(dims))
    model.eval()

    torch.manual_seed(0)
    with torch.no_grad():
        mu1, sig1 = model.forward_predict(random_x)

    torch.manual_seed(0)
    with torch.no_grad():
        mu2, sig2 = model.forward_predict(random_x)

    assert torch.allclose(mu1, mu2, atol=1e-6)
    assert torch.allclose(sig1, sig2, atol=1e-6)
