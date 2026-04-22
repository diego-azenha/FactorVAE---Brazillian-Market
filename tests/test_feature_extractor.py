"""Tests for FeatureExtractor."""

from __future__ import annotations

import torch
from factorvae.models.feature_extractor import FeatureExtractor


def test_feature_extractor_output_shape(dims):
    N, T, C, H = dims["N"], dims["T"], dims["C"], dims["H"]
    model = FeatureExtractor(C, H)
    x = torch.randn(N, T, C)
    e = model(x)
    assert e.shape == (N, H), f"Expected ({N},{H}), got {e.shape}"


def test_feature_extractor_grad_flows(dims):
    N, T, C, H = dims["N"], dims["T"], dims["C"], dims["H"]
    model = FeatureExtractor(C, H)
    x = torch.randn(N, T, C)
    e = model(x)
    loss = e.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.abs().sum().item() > 0, f"Zero gradient for {name}"


def test_feature_extractor_n_invariance(dims):
    T, C, H = dims["T"], dims["C"], dims["H"]
    model = FeatureExtractor(C, H)
    model.eval()
    with torch.no_grad():
        e_small = model(torch.randn(50, T, C))
        e_large = model(torch.randn(500, T, C))
    assert e_small.shape == (50, H)
    assert e_large.shape == (500, H)
