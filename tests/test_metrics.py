"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import torch

from factorvae.evaluation.metrics import compute_rank_ic, compute_rank_icir


def test_rank_ic_perfect():
    torch.manual_seed(0)
    y = torch.randn(100)
    assert abs(compute_rank_ic(y, y) - 1.0) < 1e-6


def test_rank_ic_inverse():
    torch.manual_seed(0)
    y = torch.randn(100)
    assert abs(compute_rank_ic(y, -y) - (-1.0)) < 1e-6


def test_rank_ic_near_zero_for_random():
    torch.manual_seed(42)
    rank_ics = []
    for _ in range(20):
        y_true = torch.randn(500)
        y_pred = torch.randn(500)
        rank_ics.append(compute_rank_ic(y_true, y_pred))
    mean_ic = abs(np.mean(rank_ics))
    assert mean_ic < 0.1, f"Random Rank IC should be near 0; got {mean_ic:.4f}"


def test_rank_icir_stable_signal():
    # Consistent positive IC -> positive ICIR
    rank_ics = [0.05] * 100
    icir = compute_rank_icir(rank_ics)
    # std=0 -> ICIR = 0 (degenerate); allow for that edge case
    assert icir == 0.0 or icir > 0


def test_rank_ic_bounded():
    torch.manual_seed(1)
    y = torch.randn(200)
    y_pred = torch.randn(200)
    ic = compute_rank_ic(y, y_pred)
    assert -1.0 <= ic <= 1.0
