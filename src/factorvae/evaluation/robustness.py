"""
Moderate robustness test for the Brazilian equity universe.

The original paper (China A-shares, ~3500 tickers) randomly removes 50–200 stocks
and checks that Rank IC degrades gracefully. That methodology does not translate
to a ~130-stock Brazilian universe: removing 50 stocks destroys the cross-section.

This module uses a fractional drop instead:
  - Drop `drop_frac` of each date's available stocks (default 15% ≈ 20 stocks).
  - Repeat for `n_trials` independent random seeds.
  - Report mean and std of Rank IC across trials, plus the full-universe baseline.

A well-calibrated model should show:
  - IC_mean_drop ≈ IC_full  (small degradation — signal is spread across the universe)
  - IC_std_drop  small      (stability — not driven by a handful of stocks)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from factorvae.evaluation.metrics import compute_rank_ic


def robustness_drop_test(
    predictions: pd.DataFrame,
    drop_frac: float = 0.15,
    n_trials: int = 5,
    seed: int = 42,
) -> dict:
    """
    Assess how much Rank IC degrades when a fraction of stocks is randomly removed.

    Args:
        predictions: DataFrame with columns [date, ticker, mu_pred, y_true].
                     Rows with NaN y_true are excluded before dropping.
        drop_frac:   Fraction of stocks to drop per date per trial (0 < drop_frac < 1).
                     Default 0.15 — drops ~20 of ~130 stocks, leaving ~110.
        n_trials:    Number of independent drop trials. Default 5.
        seed:        Base random seed; trial i uses seed + i.

    Returns:
        dict with keys:
            rank_ic_full   : Rank IC on the complete universe (baseline)
            rank_ic_mean   : Mean Rank IC across all (date, trial) pairs
            rank_ic_std    : Std Rank IC across trials (per-trial means)
            drop_frac      : drop_frac used
            n_trials       : n_trials used
            avg_n_full     : Average number of stocks per date (full universe)
            avg_n_dropped  : Average number of stocks per date after dropping
    """
    predictions = predictions.copy()
    predictions["date"] = pd.to_datetime(predictions["date"])

    # ── Full-universe baseline ────────────────────────────────────────────
    full_ics: list[float] = []
    dates = sorted(predictions["date"].unique())

    for date in dates:
        grp = predictions[predictions["date"] == date].dropna(subset=["y_true"])
        if len(grp) < 5:
            continue
        y_true = torch.tensor(grp["y_true"].values, dtype=torch.float32)
        mu     = torch.tensor(grp["mu_pred"].values, dtype=torch.float32)
        full_ics.append(compute_rank_ic(y_true, mu))

    rank_ic_full = float(np.mean(full_ics)) if full_ics else float("nan")

    # ── Drop trials ───────────────────────────────────────────────────────
    trial_means: list[float] = []
    n_full_per_date: list[int]    = []
    n_dropped_per_date: list[int] = []

    for trial in range(n_trials):
        rng = np.random.default_rng(seed + trial)
        trial_ics: list[float] = []

        for date in dates:
            grp = predictions[predictions["date"] == date].dropna(subset=["y_true"])
            n = len(grp)
            if n < 5:
                continue

            n_drop = max(1, int(np.round(n * drop_frac)))
            n_keep = n - n_drop
            if n_keep < 5:
                # Guarantee a minimum cross-section of 5 regardless of universe size
                n_keep = 5
            idx = rng.choice(n, size=n_keep, replace=False)
            sub = grp.iloc[idx]

            y_true = torch.tensor(sub["y_true"].values, dtype=torch.float32)
            mu     = torch.tensor(sub["mu_pred"].values, dtype=torch.float32)
            trial_ics.append(compute_rank_ic(y_true, mu))

            if trial == 0:  # collect sizes once
                n_full_per_date.append(n)
                n_dropped_per_date.append(n_keep)

        if trial_ics:
            trial_means.append(float(np.mean(trial_ics)))

    rank_ic_mean = float(np.mean(trial_means)) if trial_means else float("nan")
    rank_ic_std  = float(np.std(trial_means, ddof=1)) if len(trial_means) > 1 else 0.0
    avg_n_full    = float(np.mean(n_full_per_date))    if n_full_per_date    else float("nan")
    avg_n_dropped = float(np.mean(n_dropped_per_date)) if n_dropped_per_date else float("nan")

    return {
        "rank_ic_full":   rank_ic_full,
        "rank_ic_mean":   rank_ic_mean,
        "rank_ic_std":    rank_ic_std,
        "drop_frac":      drop_frac,
        "n_trials":       n_trials,
        "avg_n_full":     avg_n_full,
        "avg_n_dropped":  avg_n_dropped,
    }
