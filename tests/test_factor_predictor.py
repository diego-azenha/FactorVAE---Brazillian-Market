"""Tests for FactorPredictor."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from factorvae.models.factor_predictor import FactorPredictor, SingleHeadAttention


def test_predictor_output_shapes(dims):
    N, H, K = dims["N"], dims["H"], dims["K"]
    predictor = FactorPredictor(H, K)
    e = torch.randn(N, H)
    mu, sigma = predictor(e)
    assert mu.shape == (K,), f"Expected ({K},), got {mu.shape}"
    assert sigma.shape == (K,), f"Expected ({K},), got {sigma.shape}"


def test_predictor_sigma_positive(dims):
    N, H, K = dims["N"], dims["H"], dims["K"]
    predictor = FactorPredictor(H, K)
    e = torch.randn(N, H)
    _, sigma = predictor(e)
    assert (sigma > 0).all(), "sigma_prior must be strictly positive"


def test_attention_weights_sum_to_one(dims):
    N, H = dims["N"], dims["H"]
    head = SingleHeadAttention(H)
    head.eval()
    e = torch.randn(N, H)

    with torch.no_grad():
        k = head.key(e)     # (N, H)
        q_norm = F.normalize(head.q.unsqueeze(0), dim=-1)
        k_norm = F.normalize(k, dim=-1)
        scores = (q_norm * k_norm).sum(dim=-1).clamp(min=0)
        score_sum = scores.sum()

    if score_sum > 1e-9:
        weights = scores / score_sum
        assert abs(weights.sum().item() - 1.0) < 1e-5, (
            f"Attention weights must sum to 1, got {weights.sum().item()}"
        )


def test_predictor_query_receives_gradient(dims):
    N, H, K = dims["N"], dims["H"], dims["K"]
    predictor = FactorPredictor(H, K)
    e = torch.randn(N, H)
    mu, sigma = predictor(e)
    loss = mu.sum() + sigma.sum()
    loss.backward()

    for k_idx, head in enumerate(predictor.heads):
        assert head.q.grad is not None, f"Head {k_idx} query q has no gradient"
        assert head.q.grad.abs().sum().item() > 0, f"Head {k_idx} query q has zero gradient"
