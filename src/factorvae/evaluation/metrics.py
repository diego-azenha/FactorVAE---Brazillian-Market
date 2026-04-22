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


def rolling_rank_ic(
    predictions: "pd.DataFrame",
    window: int = 60,
) -> "pd.Series":
    """
    Rolling mean of per-date Rank IC over a window of trading days.

    Args:
        predictions: DataFrame with columns [date, ticker, mu_pred, y_true]
        window:      rolling window in trading days (60 ≈ 3 months)

    Returns:
        pd.Series indexed by date with rolling mean Rank IC.
    """
    import pandas as pd
    import torch

    predictions = predictions.copy()
    predictions["date"] = pd.to_datetime(predictions["date"])

    ics: list[float] = []
    dates: list = []
    for date, grp in predictions.groupby("date"):
        valid = grp.dropna(subset=["y_true", "mu_pred"])
        if len(valid) < 5:
            continue
        y_true = torch.tensor(valid["y_true"].values, dtype=torch.float32)
        mu     = torch.tensor(valid["mu_pred"].values, dtype=torch.float32)
        ics.append(compute_rank_ic(y_true, mu))
        dates.append(date)

    series = pd.Series(ics, index=dates).sort_index()
    return series.rolling(window, min_periods=window // 2).mean()
