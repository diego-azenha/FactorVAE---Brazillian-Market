"""
Tests for losses.py.

Includes the critical overfit gate: the model must reduce loss by ≥50%
in 200 gradient steps on a single synthetic batch.
"""

from __future__ import annotations

import torch
import torch.optim as optim

from factorvae.models.factorvae import FactorVAE
from factorvae.training.losses import kl_loss, reconstruction_loss


def _make_config(dims):
    return {
        "model": {
            "num_features": dims["C"],
            "hidden_dim": dims["H"],
            "num_factors": dims["K"],
            "num_portfolios": dims["M"],
            "leaky_relu_slope": 0.1,
        },
        "training": {"gamma": 1.0, "sigma_floor": 1e-6},
    }


def test_reconstruction_loss_finite(dims):
    N = dims["N"]
    y = torch.randn(N)
    mu = torch.randn(N)
    sigma = torch.rand(N) + 0.1
    loss = reconstruction_loss(y, mu, sigma)
    assert torch.isfinite(loss), f"Reconstruction loss is not finite: {loss}"


def test_kl_loss_finite(dims):
    K = dims["K"]
    mu_post = torch.randn(K)
    sigma_post = torch.rand(K) + 0.1
    mu_prior = torch.randn(K)
    sigma_prior = torch.rand(K) + 0.1
    loss = kl_loss(mu_post, sigma_post, mu_prior, sigma_prior)
    assert torch.isfinite(loss), f"KL loss is not finite: {loss}"


def test_overfit_single_batch(dims, synthetic_factor_batch):
    """
    GATE: Training on a single synthetic batch for 200 steps must reduce
    total loss by at least 50%. Failure means the model cannot learn at all.
    """
    x, y = synthetic_factor_batch
    config = _make_config(dims)
    gamma = config["training"]["gamma"]
    floor = config["training"]["sigma_floor"]

    torch.manual_seed(0)
    model = FactorVAE(config)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for step in range(200):
        optimizer.zero_grad()
        out = model.forward_train(x, y)
        loss_r = reconstruction_loss(y, out["mu_y_rec"], out["sigma_y_rec"], floor)
        loss_k = kl_loss(out["mu_post"], out["sigma_post"], out["mu_prior"], out["sigma_prior"], floor)
        loss = loss_r + gamma * loss_k
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    initial_loss = losses[0]
    final_loss = losses[-1]
    reduction = (initial_loss - final_loss) / abs(initial_loss)

    assert reduction >= 0.50, (
        f"OVERFIT GATE FAILED: loss reduced only {reduction*100:.1f}% "
        f"(from {initial_loss:.4f} to {final_loss:.4f}). "
        "Model cannot learn from synthetic data — check architecture."
    )
