"""Integration tests for FactorVAE."""

from __future__ import annotations

import torch
import yaml
from pathlib import Path

from factorvae.models.factorvae import FactorVAE


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


def test_forward_train_output_keys(dims, random_x):
    N, K = dims["N"], dims["K"]
    model = FactorVAE(_make_config(dims))
    y = torch.randn(N)
    out = model.forward_train(random_x, y)

    expected_keys = {"e", "mu_post", "sigma_post", "mu_prior", "sigma_prior", "mu_y_rec", "sigma_y_rec"}
    assert set(out.keys()) == expected_keys


def test_forward_train_output_shapes(dims, random_x):
    N, H, K = dims["N"], dims["H"], dims["K"]
    model = FactorVAE(_make_config(dims))
    y = torch.randn(N)
    out = model.forward_train(random_x, y)

    assert out["e"].shape == (N, H)
    for key in ("mu_post", "sigma_post", "mu_prior", "sigma_prior"):
        assert out[key].shape == (K,), f"{key} shape wrong: {out[key].shape}"
    for key in ("mu_y_rec", "sigma_y_rec"):
        assert out[key].shape == (N,), f"{key} shape wrong: {out[key].shape}"


def test_forward_train_grad_flows_all_modules(dims, random_x):
    """Gradient must reach parameters in all four modules after a train step."""
    N = dims["N"]
    model = FactorVAE(_make_config(dims))
    y = torch.randn(N)
    out = model.forward_train(random_x, y)
    # Include all outputs so every module's parameters receive gradient
    loss = (out["mu_y_rec"].sum() + out["sigma_y_rec"].sum()
            + out["mu_prior"].sum() + out["sigma_prior"].sum())
    loss.backward()

    modules_to_check = {
        "feature_extractor": model.feature_extractor,
        "encoder": model.encoder,
        "predictor": model.predictor,
        "decoder": model.decoder,
    }
    for module_name, module in modules_to_check.items():
        for param_name, param in module.named_parameters():
            assert param.grad is not None, f"No gradient: {module_name}.{param_name}"


def test_forward_predict_output_shapes(dims, random_x):
    N = dims["N"]
    model = FactorVAE(_make_config(dims))
    mu_pred, sigma_pred = model.forward_predict(random_x)
    assert mu_pred.shape == (N,)
    assert sigma_pred.shape == (N,)


def test_encoder_unused_in_predict(dims, random_x):
    """
    After forward_predict, encoder parameters must have NO gradient.
    This ensures there is no information leakage through the encoder at inference.
    """
    model = FactorVAE(_make_config(dims))
    mu_pred, sigma_pred = model.forward_predict(random_x)
    loss = (mu_pred + sigma_pred).sum()
    loss.backward()

    for name, param in model.encoder.named_parameters():
        grad_sum = 0.0 if param.grad is None else param.grad.abs().sum().item()
        assert grad_sum == 0.0, (
            f"Encoder parameter '{name}' has non-zero gradient during predict: {grad_sum}"
        )
