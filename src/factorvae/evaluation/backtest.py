"""
Portfolio backtest for FactorVAE predictions.

Implements TopK-Drop strategy and its risk-adjusted variant (TDrisk).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def topk_drop_strategy(
    predictions: pd.DataFrame,
    k: int,
    n: int,
    eta: float = 0.0,
    fee_rate: float = 0.001,
) -> pd.DataFrame:
    """
    TopK-Drop strategy.

    On each trading date, hold k stocks. Allow at most n stocks to be dropped/added
    per day (turnover constraint: |P_t ∩ P_{t-1}| >= k - n).

    Args:
        predictions: DataFrame with columns [date, ticker, mu_pred, sigma_pred, y_true]
        k:           portfolio size
        n:           max stocks replaced per day
        eta:         risk aversion weight; score = mu - eta*sigma (0 = pure alpha)
        fee_rate:    one-way transaction cost

    Returns:
        DataFrame with columns [date, portfolio_return, turnover]
    """
    predictions = predictions.sort_values("date")
    dates = sorted(predictions["date"].unique())

    current_portfolio: set[str] = set()
    records = []

    for date in dates:
        day = predictions[predictions["date"] == date].copy()

        if eta > 0.0:
            day["score"] = day["mu_pred"] - eta * day["sigma_pred"]
        else:
            day["score"] = day["mu_pred"]

        # Rank by score descending
        day = day.sort_values("score", ascending=False)

        # TopK-Drop: keep existing holdings that are still in top-(k + n) candidates,
        # then fill to k with the highest-scoring newcomers.
        top_candidates = set(day.head(k + n)["ticker"])
        retained = current_portfolio & top_candidates
        needed = k - len(retained)
        candidates_sorted = day[~day["ticker"].isin(retained)]["ticker"].tolist()
        new_stocks = set(candidates_sorted[:needed])
        new_portfolio = retained | new_stocks

        # Turnover: fraction of portfolio changed
        if current_portfolio:
            turnover = len(new_portfolio - current_portfolio) / k
        else:
            turnover = 1.0

        # Equal-weight return
        held = day[day["ticker"].isin(new_portfolio)]
        gross_return = held["y_true"].mean() if len(held) > 0 else 0.0
        net_return = gross_return - fee_rate * turnover

        records.append({"date": date, "portfolio_return": net_return, "turnover": turnover})
        current_portfolio = new_portfolio

    return pd.DataFrame(records)


def compute_performance_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> dict:
    """
    Compute annualized excess return, Sharpe ratio, and max drawdown
    of portfolio relative to benchmark.

    Args:
        portfolio_returns: daily portfolio returns (net of fees)
        benchmark_returns: daily benchmark returns

    Returns:
        dict with keys: annualized_return, sharpe, max_drawdown
    """
    excess = portfolio_returns.values - benchmark_returns.reindex(portfolio_returns.index).fillna(0).values
    trading_days = 252

    annualized_return = float(np.mean(excess) * trading_days)
    annualized_vol = float(np.std(excess, ddof=1) * np.sqrt(trading_days))
    sharpe = annualized_return / annualized_vol if annualized_vol > 1e-9 else 0.0

    # Max drawdown on cumulative excess
    cum = np.cumprod(1 + excess)
    running_max = np.maximum.accumulate(cum)
    drawdown = (running_max - cum) / running_max
    max_drawdown = float(drawdown.max())

    return {
        "annualized_return": annualized_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }
