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
    fee_rate: float = 0.0025,
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

        # TopK-Drop: retain existing holdings that still rank in the universe,
        # but cap additions at n per day (hard turnover constraint).
        # Step 1: force-retain the best k-n holdings from the current portfolio
        #         (those ranked highest among current stocks today).
        if current_portfolio:
            held_today = day[day["ticker"].isin(current_portfolio)]
            force_keep = set(held_today.head(k - n)["ticker"])  # top k-n existing
        else:
            force_keep = set()
        # Step 2: from remaining universe (excluding forced keeps), pick top n newcomers
        remaining = day[~day["ticker"].isin(force_keep)]
        new_picks = set(remaining.head(n)["ticker"])
        new_portfolio = force_keep | new_picks

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
    risk_free_rate: float = 0.10,
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
        sharpe             : (annualized_return - risk_free_rate) / volatility
        risk_free_rate     : annualised risk-free rate used in Sharpe (default 10%)
        information_ratio  : excess AR / tracking error
        max_drawdown       : max peak-to-trough on cumulative portfolio returns
        calmar             : annualized_return / max_drawdown
        hit_rate           : fraction of days where portfolio return > 0
        avg_turnover       : mean daily turnover (only if turnover is provided)
    """
    bench = benchmark_returns.reindex(portfolio_returns.index).fillna(0.0).values
    port  = portfolio_returns.values
    excess = port - bench

    # Use nan-safe operations so that dates with no valid y_true (e.g. last 2 days
    # of the dataset where forward return is undefined) don't propagate NaN.
    days = 252

    # CAGR (geometric annualised return) — consistent with cumulative wealth plots.
    # Arithmetic mean × 252 overstates performance for high-volatility strategies.
    valid_port  = port[~np.isnan(port)]
    n_port      = len(valid_port)
    cum_wealth  = float(np.prod(1.0 + valid_port))
    ann_return  = float(cum_wealth ** (days / n_port) - 1.0) if n_port > 0 else 0.0

    valid_bench = bench[~np.isnan(bench)]
    n_bench     = len(valid_bench)
    cum_bm      = float(np.prod(1.0 + valid_bench))
    ann_bm      = float(cum_bm ** (days / n_bench) - 1.0) if n_bench > 0 else 0.0
    ann_excess  = ann_return - ann_bm

    vol = float(np.nanstd(port, ddof=1) * np.sqrt(days))

    excess_vol = float(np.nanstd(excess, ddof=1) * np.sqrt(days))
    sharpe     = (ann_return - risk_free_rate) / vol if vol > 1e-9 else 0.0
    info_ratio = ann_excess / excess_vol if excess_vol > 1e-9 else 0.0

    cum_port    = np.cumprod(1.0 + valid_port)
    running_max = np.maximum.accumulate(cum_port)
    drawdown    = (running_max - cum_port) / running_max
    mdd = float(drawdown.max()) if len(drawdown) > 0 else 0.0

    calmar   = ann_return / mdd if mdd > 1e-9 else 0.0
    hit_rate = float(np.nanmean(port > 0))

    out = {
        "annualized_return":   ann_return,
        "cumulative_return":   float(cum_wealth - 1.0),
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
