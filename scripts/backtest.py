"""
Portfolio backtest from saved predictions.

Reads results/predictions/predictions.parquet, applies TopK-Drop (eta=0)
and TDrisk (eta>0), computes performance metrics, saves tables and equity curve.

Usage:
    python scripts/backtest.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from factorvae.evaluation.backtest import compute_performance_metrics, topk_drop_strategy

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument(
        "--predictions",
        default=str(ROOT / "results" / "predictions" / "predictions.parquet"),
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    k = config["evaluation"]["top_k"]
    n = config["evaluation"]["drop_n"]
    eta = config["evaluation"]["risk_aversion_eta"]

    predictions = pd.read_parquet(args.predictions)

    # Benchmark: equal-weight of all stocks each day
    benchmark = predictions.groupby("date")["y_true"].mean().rename("benchmark")

    # Strategy 1: TopK-Drop (mu only)
    port_mu = topk_drop_strategy(predictions, k=k, n=n, eta=0.0)
    port_mu = port_mu.set_index("date")["portfolio_return"]

    # Strategy 2: TDrisk (mu - eta*sigma)
    port_td = topk_drop_strategy(predictions, k=k, n=n, eta=eta)
    port_td = port_td.set_index("date")["portfolio_return"]

    # Metrics
    metrics_mu = compute_performance_metrics(port_mu, benchmark)
    metrics_td = compute_performance_metrics(port_td, benchmark)

    print("\n=== TopK-Drop (mu only) ===")
    for k_, v in metrics_mu.items():
        print(f"  {k_:25s}: {v:.4f}")

    print("\n=== TDrisk (mu - eta*sigma) ===")
    for k_, v in metrics_td.items():
        print(f"  {k_:25s}: {v:.4f}")

    # Save figures
    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    cum_mu = (1 + port_mu).cumprod()
    cum_td = (1 + port_td).cumprod()
    cum_bm = (1 + benchmark.reindex(port_mu.index).fillna(0)).cumprod()
    ax.plot(cum_mu.values, label="TopK-Drop")
    ax.plot(cum_td.values, label=f"TDrisk (η={eta})")
    ax.plot(cum_bm.values, label="Benchmark", linestyle="--")
    ax.set_title("Cumulative Return")
    ax.legend()
    ax.set_xlabel("Trading days")
    ax.set_ylabel("Cumulative return (1 = start)")
    fig.savefig(fig_dir / "cumulative_return.png", dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {fig_dir}/cumulative_return.png")


if __name__ == "__main__":
    main()
