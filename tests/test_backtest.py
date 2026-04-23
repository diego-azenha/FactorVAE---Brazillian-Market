"""
Tests for evaluation/backtest.py.

Covers topk_drop_strategy and compute_performance_metrics with
synthetic predictions DataFrames to verify correctness of portfolio
construction, turnover mechanics, and performance metric calculations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorvae.evaluation.backtest import compute_performance_metrics, topk_drop_strategy


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _make_predictions(
    n_dates: int = 20,
    n_stocks: int = 50,
    seed: int = 0,
    signal_strength: float = 0.0,
    stable_ranking: bool = False,
) -> pd.DataFrame:
    """
    Build a synthetic predictions DataFrame.

    If signal_strength > 0, mu_pred is positively correlated with y_true
    (perfect predictor uses signal_strength=1.0).

    If stable_ranking=True, each stock gets a large fixed base score that
    dominates daily noise, so the top-k+n stocks are consistent across
    dates.  Required for testing the n/k turnover bound.
    """
    rng = np.random.default_rng(seed)
    records = []
    dates = [f"2020-01-{d+1:02d}" for d in range(n_dates)]
    tickers = [f"T{i:03d}" for i in range(n_stocks)]
    # Fixed per-stock base score (descending): T000 has highest score
    base_scores = np.linspace(1.0, 0.0, n_stocks)  # range 1.0 → 0.0

    for date in dates:
        y_true = rng.normal(0.001, 0.02, size=n_stocks)
        noise = rng.normal(0, 0.02, size=n_stocks)
        mu_pred = signal_strength * y_true + (1 - signal_strength) * noise
        if stable_ranking:
            # Add large fixed offset so rankings never shuffle
            mu_pred = base_scores + rng.normal(0, 0.001, size=n_stocks)
        sigma_pred = np.abs(rng.normal(0.02, 0.005, size=n_stocks))
        for i, ticker in enumerate(tickers):
            records.append({
                "date": date,
                "ticker": ticker,
                "mu_pred": float(mu_pred[i]),
                "sigma_pred": float(sigma_pred[i]),
                "y_true": float(y_true[i]),
            })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# topk_drop_strategy tests
# ─────────────────────────────────────────────────────────────

def test_topk_drop_output_columns():
    """Output DataFrame must have exactly these three columns."""
    preds = _make_predictions()
    result = topk_drop_strategy(preds, k=10, n=2)
    assert set(result.columns) == {"date", "portfolio_return", "turnover"}


def test_topk_drop_row_count():
    """One row per trading date."""
    n_dates = 15
    preds = _make_predictions(n_dates=n_dates)
    result = topk_drop_strategy(preds, k=10, n=2)
    assert len(result) == n_dates


def test_topk_drop_first_turnover_is_one():
    """On the very first date, turnover must be 1.0 (portfolio built from scratch)."""
    preds = _make_predictions()
    result = topk_drop_strategy(preds, k=10, n=2)
    assert result.iloc[0]["turnover"] == 1.0


def test_topk_drop_subsequent_turnover_bounded():
    """
    After the first day, turnover must not exceed n/k per day
    (at most n stocks are replaced).

    Requires a stable ranking signal so that the top-(k+n) window
    contains consistent stocks across dates.  With fully random scores
    the window can shift entirely, causing complete portfolio replacement.
    """
    k, n = 10, 3
    preds = _make_predictions(n_dates=30, stable_ranking=True)
    result = topk_drop_strategy(preds, k=k, n=n)
    max_subsequent = result.iloc[1:]["turnover"].max()
    assert max_subsequent <= n / k + 1e-9, (
        f"Turnover {max_subsequent:.3f} exceeds max allowed {n/k:.3f}"
    )


def test_topk_drop_tdrisk_uses_sigma():
    """
    With eta > 0, TDrisk scores = mu - eta*sigma.
    Ranking must differ from pure-mu ranking at least some of the time
    when sigma is non-constant.
    """
    preds = _make_predictions(n_dates=10, seed=99)
    result_mu = topk_drop_strategy(preds, k=10, n=5, eta=0.0)
    result_td = topk_drop_strategy(preds, k=10, n=5, eta=5.0)
    # With non-constant sigma and eta=5, the two strategies should differ
    # at least once over 10 dates (not identical portfolio returns every day)
    identical_days = (result_mu["portfolio_return"].values == result_td["portfolio_return"].values).all()
    assert not identical_days, (
        "TDrisk (eta=5) produced identical returns to pure-mu — sigma is not influencing selection"
    )


def test_topk_drop_portfolio_return_is_finite():
    """No NaN or Inf in portfolio_return."""
    preds = _make_predictions(n_dates=20, n_stocks=100)
    result = topk_drop_strategy(preds, k=10, n=3)
    assert result["portfolio_return"].notna().all()
    assert np.isfinite(result["portfolio_return"].values).all()


# ─────────────────────────────────────────────────────────────
# compute_performance_metrics tests
# ─────────────────────────────────────────────────────────────

def test_performance_metrics_keys():
    """Output dict must contain all expected metric keys."""
    rets = pd.Series([0.001] * 252)
    bench = pd.Series([0.0] * 252)
    m = compute_performance_metrics(rets, bench)
    required = {
        "annualized_return", "annualized_excess", "volatility",
        "sharpe", "information_ratio", "max_drawdown", "calmar", "hit_rate",
    }
    assert required.issubset(set(m.keys()))


def test_performance_metrics_positive_constant_series():
    """
    For a constant positive excess return, AR > 0, MDD ≈ 0.
    (No drawdown on a monotonically growing cumulative curve.)
    """
    rets = pd.Series([0.001] * 252)
    bench = pd.Series([0.0] * 252)
    m = compute_performance_metrics(rets, bench)
    assert m["annualized_return"] > 0
    assert m["max_drawdown"] < 1e-6


def test_performance_metrics_zero_series():
    """Zero excess returns → AR = 0, Sharpe = 0."""
    rets = pd.Series([0.0] * 252)
    bench = pd.Series([0.0] * 252)
    m = compute_performance_metrics(rets, bench)
    assert abs(m["annualized_return"]) < 1e-9
    assert m["sharpe"] == 0.0


def test_performance_metrics_max_drawdown_detection():
    """
    Series that goes up then drops 50% should report MDD ≈ 0.5.
    """
    # Cumulative: 1 → 2 → 1  (50% drawdown)
    # Daily excess: +100% then -50%  i.e. +1.0 then -0.5
    rets = pd.Series([1.0] + [-0.5] + [0.0] * 250)
    bench = pd.Series([0.0] * 252)
    m = compute_performance_metrics(rets, bench)
    assert abs(m["max_drawdown"] - 0.5) < 1e-3
