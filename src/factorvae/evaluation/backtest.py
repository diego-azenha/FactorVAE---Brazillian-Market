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
    turnover: "pd.Series | None" = None,
) -> dict:
    """
    Extended performance metrics.

    Args:
        portfolio_returns: daily net returns of the strategy
        benchmark_returns: daily returns of the reference index
        turnover:          daily turnover series from topk_drop_strategy (optional)

    Returns dict with:
        annualized_return  : strategy AR (absolute)
        annualized_excess  : AR over benchmark
        volatility         : annualized std of strategy returns
        sharpe             : SR on excess returns
        information_ratio  : excess AR / tracking error (identical to sharpe here)
        max_drawdown       : max peak-to-trough on cumulative excess returns
        calmar             : annualized_excess / max_drawdown
        hit_rate           : fraction of days where excess return > 0
        avg_turnover       : mean daily turnover (only if turnover is provided)
    """
    bench = benchmark_returns.reindex(portfolio_returns.index).fillna(0.0).values
    port  = portfolio_returns.values
    excess = port - bench

    days = 252
    ann_return = float(np.mean(port) * days)
    ann_excess = float(np.mean(excess) * days)
    vol = float(np.std(port, ddof=1) * np.sqrt(days))

    excess_vol = float(np.std(excess, ddof=1) * np.sqrt(days))
    sharpe = ann_excess / excess_vol if excess_vol > 1e-9 else 0.0
    info_ratio = sharpe  # identical under this convention; both reported for clarity

    cum_excess  = np.cumprod(1.0 + excess)
    running_max = np.maximum.accumulate(cum_excess)
    drawdown    = (running_max - cum_excess) / running_max
    mdd = float(drawdown.max()) if len(drawdown) > 0 else 0.0

    calmar   = ann_excess / mdd if mdd > 1e-9 else 0.0
    hit_rate = float(np.mean(excess > 0))

    out = {
        "annualized_return":  ann_return,
        "annualized_excess":  ann_excess,
        "volatility":         vol,
        "sharpe":             sharpe,
        "information_ratio":  info_ratio,
        "max_drawdown":       mdd,
        "calmar":             calmar,
        "hit_rate":           hit_rate,
    }
    if turnover is not None:
        out["avg_turnover"] = float(turnover.reindex(portfolio_returns.index).mean())
    return out
