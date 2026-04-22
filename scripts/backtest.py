"""
Portfolio backtest from saved predictions.

Reads results/predictions/predictions.parquet plus any available benchmark
model predictions, applies TopK-Drop strategy, computes extended performance
metrics, prints a comparison table, and saves three figures to results/figures/.

The core logic is exposed as `run_backtest_from_predictions()` so that
scripts/evaluate.py can call it inline (no subprocess needed).

Usage:
    python scripts/backtest.py
    python scripts/backtest.py --benchmark data/processed/benchmark.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from factorvae.evaluation.backtest  import compute_performance_metrics, topk_drop_strategy
from factorvae.evaluation.comparison import (
    build_comparison_table,
    load_all_predictions,
    load_benchmark,
    print_comparison,
)
from factorvae.evaluation.metrics import rolling_rank_ic

ROOT = Path(__file__).resolve().parents[1]


# ── Figure helpers ────────────────────────────────────────────────────────────

def _apply_date_fmt(ax: "plt.Axes", fig: "plt.Figure") -> None:
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()


# ── Core backtest logic (importable by evaluate.py) ───────────────────────────

def run_backtest_from_predictions(
    factorvaepreds: pd.DataFrame,
    config: dict,
    root: Path,
    benchmark_path: "Path | None" = None,
) -> None:
    """
    Full backtest: metrics table + three figures.

    Args:
        factorvaepreds: FactorVAE predictions (already loaded and date-parsed).
        config:         parsed config.yaml dict.
        root:           workspace root for locating benchmark parquets + output dirs.
        benchmark_path: path to benchmark return parquet; falls back to EW market.
    """
    k   = config["evaluation"]["top_k"]
    n   = config["evaluation"]["drop_n"]
    eta = config["evaluation"]["risk_aversion_eta"]

    if benchmark_path is None:
        benchmark_path = root / "data" / "processed" / "benchmark.parquet"

    benchmark = load_benchmark(benchmark_path, factorvaepreds)

    # ── Load all prediction sources (FactorVAE always present; benchmarks optional)
    all_preds = load_all_predictions(root)
    if "FactorVAE" not in all_preds:
        # Use the passed-in predictions directly (evaluate.py path before file is saved)
        all_preds["FactorVAE"] = factorvaepreds

    # ── Comparison table ──────────────────────────────────────────────────────
    table = build_comparison_table(root, benchmark, k=k, n=n, eta=0.0)

    fig_dir = root / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    csv_path = fig_dir / "comparison_table.csv"
    table.to_csv(csv_path)
    print(f"\n── Comparison: FactorVAE vs Benchmarks {'─' * 20}")
    print_comparison(table)
    print(f"\nTable saved → {csv_path.relative_to(root)}")

    # ── Warn if FactorVAE is dominated on Rank IC ─────────────────────────────
    if "rank_ic" in table.columns and "FactorVAE" in table.index:
        fv_ic = table.loc["FactorVAE", "rank_ic"]
        for other in table.index:
            if other != "FactorVAE":
                other_ic = table.loc[other, "rank_ic"]
                try:
                    if float(other_ic) > float(fv_ic):
                        print(
                            f"  WARNING: {other} has higher Rank IC "
                            f"({float(other_ic):.4f}) than FactorVAE ({float(fv_ic):.4f})"
                        )
                except (TypeError, ValueError):
                    pass

    # ── Build per-model portfolio return series ───────────────────────────────
    port_series: dict[str, pd.Series] = {}
    for name, preds in all_preds.items():
        port = topk_drop_strategy(preds, k=k, n=n, eta=0.0).set_index("date")
        port_series[name] = port["portfolio_return"]

    # ── Figure 1: Cumulative absolute return ──────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(13, 5))
    for name, ret in port_series.items():
        cum = (1.0 + ret).cumprod()
        ax1.plot(cum.index, cum.values, label=name)
    all_dates = sorted({d for r in port_series.values() for d in r.index})
    bm_aligned = benchmark.reindex(all_dates).fillna(0.0)
    cum_bm = (1.0 + bm_aligned).cumprod()
    label_bm = benchmark.name if hasattr(benchmark, "name") and benchmark.name else "Benchmark"
    ax1.plot(cum_bm.index, cum_bm.values, linestyle="--", color="grey", label=label_bm)
    ax1.set_title("Cumulative Return (TopK-Drop, net of fees)")
    ax1.set_ylabel("Cumulative return (1 = start)")
    ax1.legend()
    _apply_date_fmt(ax1, fig1)
    fig1.savefig(fig_dir / "cumulative_return.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("Figure saved → results/figures/cumulative_return.png")

    # ── Figure 2: Cumulative excess return (%) ────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(13, 5))
    for name, ret in port_series.items():
        excess = ret - benchmark.reindex(ret.index).fillna(0.0)
        cum_ex = (1.0 + excess).cumprod() - 1.0
        ax2.plot(cum_ex.index, cum_ex.values * 100.0, label=name)
    ax2.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax2.set_title("Cumulative Excess Return vs Benchmark (TopK-Drop)")
    ax2.set_ylabel("Cumulative excess return (%)")
    ax2.legend()
    _apply_date_fmt(ax2, fig2)
    fig2.savefig(fig_dir / "cumulative_excess_return.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("Figure saved → results/figures/cumulative_excess_return.png")

    # ── Figure 3: Rolling 60-day Rank IC ──────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(13, 4))
    for name, preds in all_preds.items():
        r = rolling_rank_ic(preds, window=60)
        ax3.plot(r.index, r.values, label=name)
    ax3.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax3.set_title("60-day Rolling Rank IC")
    ax3.set_ylabel("Rank IC (60-day rolling mean)")
    ax3.legend()
    _apply_date_fmt(ax3, fig3)
    fig3.savefig(fig_dir / "rolling_rank_ic.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("Figure saved → results/figures/rolling_rank_ic.png")


# ── Standalone entry point ────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run full backtest and comparison.")
    parser.add_argument("--config",      default=str(ROOT / "config.yaml"))
    parser.add_argument(
        "--predictions",
        default=str(ROOT / "results" / "predictions" / "predictions.parquet"),
    )
    parser.add_argument(
        "--benchmark",
        default=str(ROOT / "data" / "processed" / "benchmark.parquet"),
        help=(
            "Parquet with columns [date, return] for the index benchmark. "
            "Falls back to equal-weight market if the file does not exist."
        ),
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    preds = pd.read_parquet(args.predictions)
    preds["date"] = pd.to_datetime(preds["date"])

    run_backtest_from_predictions(
        factorvaepreds=preds,
        config=config,
        root=ROOT,
        benchmark_path=Path(args.benchmark),
    )


if __name__ == "__main__":
    main()
