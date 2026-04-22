"""
Evaluation metrics for FactorVAE.

Rank IC (Spearman correlation between predicted and true ranks in the cross-section).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def compute_rank_ic(y_true: Tensor, y_pred: Tensor) -> float:
    """
    Rank IC: Spearman correlation between y_true and y_pred.

    Computed as Pearson correlation of their ranks (both ranked independently).

    Args:
        y_true: (N,)
        y_pred: (N,)

    Returns:
        Rank IC as a Python float in [-1, 1].
    """
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()

    rank_true = _rank(y_true_np)
    rank_pred = _rank(y_pred_np)

    return float(_pearson(rank_true, rank_pred))


def compute_rank_icir(rank_ics: list[float]) -> float:
    """
    Rank ICIR: mean / std of a series of Rank IC values.
    Measures consistency of the signal.
    """
    arr = np.array(rank_ics)
    std = arr.std()
    if std < 1e-9:
        return 0.0
    return float(arr.mean() / std)


def _rank(x: np.ndarray) -> np.ndarray:
    """Convert array to ranks (1-based, average for ties)."""
    idx = np.argsort(x)
    ranks = np.empty_like(idx, dtype=float)
    ranks[idx] = np.arange(1, len(x) + 1)
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a_dm = a - a.mean()
    b_dm = b - b.mean()
    denom = np.sqrt((a_dm ** 2).sum() * (b_dm ** 2).sum())
    if denom < 1e-12:
        return 0.0
    return float((a_dm * b_dm).sum() / denom)
