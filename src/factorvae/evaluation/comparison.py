"""
Comparison framework: load all prediction sources, compute metrics, print table.

Used by both scripts/evaluate.py (inline) and scripts/backtest.py (standalone).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from factorvae.evaluation.backtest import compute_performance_metrics, topk_drop_strategy
from factorvae.evaluation.metrics import compute_rank_ic, compute_rank_icir


# ── Prediction loading ────────────────────────────────────────────────────────

def load_all_predictions(root: Path) -> dict[str, pd.DataFrame]:
    """
    Load FactorVAE and benchmark prediction parquets that exist on disk.

    Returns a dict mapping model name → DataFrame with columns
    [date, ticker, mu_pred, sigma_pred, y_true].
    Missing files are silently skipped.
    """
    sources = {
        "FactorVAE":      root / "results" / "predictions" / "predictions.parquet",
        "Momentum":       root / "benchmarks" / "predictions" / "momentum_predictions.parquet",
        "Linear (Ridge)": root / "benchmarks" / "predictions" / "linear_predictions.parquet",
    }
    out: dict[str, pd.DataFrame] = {}
    for name, path in sources.items():
        if path.exists():
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["date"])
            out[name] = df
    return out


# ── IC summary ────────────────────────────────────────────────────────────────

def compute_ic_summary(predictions: pd.DataFrame) -> dict:
    """
    Compute Rank IC and Rank ICIR aggregated over all dates.

    Dates with fewer than 5 valid rows are skipped.
    """
    ics: list[float] = []
    for _, grp in predictions.groupby("date"):
        valid = grp.dropna(subset=["y_true"])
        if len(valid) < 5:
            continue
        y_true = torch.tensor(valid["y_true"].values, dtype=torch.float32)
        mu     = torch.tensor(valid["mu_pred"].values,  dtype=torch.float32)
        ics.append(compute_rank_ic(y_true, mu))

    if not ics:
        return {"rank_ic": float("nan"), "rank_icir": float("nan")}
    return {
        "rank_ic":    sum(ics) / len(ics),
        "rank_icir":  compute_rank_icir(ics),
    }


# ── Benchmark loading helper ──────────────────────────────────────────────────

def load_benchmark(path: Path, predictions: pd.DataFrame) -> pd.Series:
    """
    Load benchmark return series.

    If *path* exists, reads a parquet with columns [date, return].
    Otherwise falls back to equal-weight cross-section average of y_true
    (labeled "EW Market" in output).
    """
    if path.exists():
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")["return"]
    predictions = predictions.copy()
    predictions["date"] = pd.to_datetime(predictions["date"])
    return predictions.groupby("date")["y_true"].mean().rename("EW Market")


# ── Full comparison table ─────────────────────────────────────────────────────

def build_comparison_table(
    root: Path,
    benchmark_returns: pd.Series,
    k: int = 50,
    n: int = 5,
    eta: float = 0.0,
) -> pd.DataFrame:
    """
    Run TopK-Drop for each prediction source and build a comparative metrics table.

    Args:
        root:              workspace root path
        benchmark_returns: daily benchmark return series (indexed by date)
        k:                 portfolio size
        n:                 max stocks replaced per day
        eta:               risk-aversion weight (0 = pure alpha ranking)

    Returns:
        pd.DataFrame indexed by model name, columns = all metrics.
    """
    preds_by_model = load_all_predictions(root)
    rows: list[dict] = []

    for name, preds in preds_by_model.items():
        ic = compute_ic_summary(preds)
        port      = topk_drop_strategy(preds, k=k, n=n, eta=eta)
        port_ret  = port.set_index("date")["portfolio_return"]
        turnover  = port.set_index("date")["turnover"]
        perf = compute_performance_metrics(port_ret, benchmark_returns, turnover=turnover)
        rows.append({"model": name, **ic, **perf})

    return pd.DataFrame(rows).set_index("model")


# ── Formatted printer ─────────────────────────────────────────────────────────

def print_comparison(df: pd.DataFrame) -> None:
    """
    Print comparison table with % formatting for return-like columns.
    """
    pct_cols = [
        "annualized_return", "annualized_excess", "volatility",
        "max_drawdown", "hit_rate", "avg_turnover",
    ]
    flt_cols = ["rank_ic", "rank_icir", "sharpe", "information_ratio", "calmar"]

    formatted = df.copy().astype(object)
    for c in pct_cols:
        if c in formatted.columns:
            formatted[c] = formatted[c].map(
                lambda v: f"{v * 100:+.2f}%" if not np.isnan(float(v)) else "N/A"
            )
    for c in flt_cols:
        if c in formatted.columns:
            formatted[c] = formatted[c].map(
                lambda v: f"{v:+.3f}" if not np.isnan(float(v)) else "N/A"
            )
    print(formatted.to_string())
