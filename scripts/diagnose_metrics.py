"""
Day 0 diagnostic — Verify performance metric arithmetic.

Recomputes all metrics from saved predictions + benchmark from scratch,
prints a full breakdown showing intermediate values (daily means, vol, etc.)
to confirm the reported Sharpe/IR is internally consistent.

Also flags the labelling convention: the reported "Sharpe" is actually
ann_excess_return / tracking_error_vol (i.e., IR), not ann_return / vol.

Usage:
    python scripts/diagnose_metrics.py
    python scripts/diagnose_metrics.py --predictions results/predictions/predictions.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import yaml

from factorvae.evaluation.backtest import compute_performance_metrics, topk_drop_strategy
from factorvae.evaluation.comparison import load_benchmark

ROOT = Path(__file__).resolve().parents[1]
DAYS = 252


def _arithmetic_breakdown(label: str, port: pd.Series, bench: pd.Series) -> dict:
    bench_aligned = bench.reindex(port.index).fillna(0.0)
    excess = port - bench_aligned

    daily_mean_port   = port.mean()
    daily_mean_bench  = bench_aligned.mean()
    daily_mean_excess = excess.mean()

    ann_return  = daily_mean_port   * DAYS
    ann_bench   = daily_mean_bench  * DAYS
    ann_excess  = daily_mean_excess * DAYS

    vol_port    = port.std(ddof=1)   * np.sqrt(DAYS)
    vol_bench   = bench_aligned.std(ddof=1) * np.sqrt(DAYS)
    vol_excess  = excess.std(ddof=1) * np.sqrt(DAYS)   # tracking error

    sharpe_traditional = ann_return  / vol_port   if vol_port   > 1e-9 else np.nan
    ir_vs_bench        = ann_excess  / vol_excess if vol_excess > 1e-9 else np.nan

    cum_excess  = np.cumprod(1.0 + excess.values)
    running_max = np.maximum.accumulate(cum_excess)
    drawdown    = (running_max - cum_excess) / running_max
    mdd = drawdown.max()

    n_days = len(port)
    n_pos  = (excess > 0).sum()

    return {
        "label":                label,
        "n_days":               n_days,
        "daily_mean_port (bps)":daily_mean_port * 1e4,
        "daily_mean_bench(bps)":daily_mean_bench * 1e4,
        "daily_mean_excess(bps)":daily_mean_excess * 1e4,
        "ann_return   (%)":     ann_return  * 100,
        "ann_bench    (%)":     ann_bench   * 100,
        "ann_excess   (%)":     ann_excess  * 100,
        "vol_port     (%)":     vol_port    * 100,
        "vol_bench    (%)":     vol_bench   * 100,
        "tracking_err (%)":     vol_excess  * 100,
        "Sharpe (ann/vol_port)":sharpe_traditional,
        "IR    (exc/te)  REPORTED": ir_vs_bench,
        "max_drawdown (%)":     mdd * 100,
        "hit_rate (%)":         n_pos / n_days * 100,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default=str(ROOT / "config.yaml"))
    parser.add_argument("--predictions", default=str(ROOT / "results" / "predictions" / "predictions.parquet"))
    parser.add_argument(
        "--benchmark",
        default=str(ROOT / "data" / "processed" / "benchmark.parquet"),
    )
    parser.add_argument("--top-k",  type=int, default=None)
    parser.add_argument("--drop-n", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    top_k  = args.top_k  or config["evaluation"]["top_k"]
    drop_n = args.drop_n or config["evaluation"]["drop_n"]

    preds = pd.read_parquet(args.predictions)
    preds["date"] = pd.to_datetime(preds["date"])

    benchmark = load_benchmark(Path(args.benchmark), preds)

    result_df = topk_drop_strategy(preds, k=top_k, n=drop_n)
    port_returns = result_df.set_index("date")["portfolio_return"]
    turnover = result_df.set_index("date")["turnover"]

    print("\n" + "=" * 70)
    print("METRIC ARITHMETIC BREAKDOWN")
    print(f"TopK-Drop  k={top_k}  drop={drop_n}  |  {len(port_returns)} trading days")
    print("=" * 70)

    bd = _arithmetic_breakdown("FactorVAE", port_returns, benchmark)

    for k, v in bd.items():
        if k == "label":
            continue
        marker = "  [REPORTED]" if "REPORTED" in k else ""
        key_clean = k.replace("[REPORTED]", "").replace("REPORTED", "").strip()
        if isinstance(v, float):
            print(f"  {key_clean:<35}  {v:>+10.4f}{marker}")
        else:
            print(f"  {key_clean:<35}  {v:>10}{marker}")

    print()
    print("NOTE: The 'Sharpe' column in comparison_table.csv is computed as")
    print("      ann_excess_return / tracking_error_vol  (= Information Ratio).")
    print("      Traditional Sharpe (ann_return / vol_port) is shown above.")
    print()

    # Quick sanity: compare to official compute_performance_metrics
    official = compute_performance_metrics(port_returns, benchmark, turnover)
    print("Official compute_performance_metrics() output:")
    for k, v in official.items():
        print(f"  {k:<30} {v:>+.4f}" if isinstance(v, float) else f"  {k:<30} {v}")

    out_path = ROOT / "results" / "diagnostics" / "metric_arithmetic.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{k}: {v}" for k, v in bd.items()]
    out_path.write_text("\n".join(lines))
    print(f"\nSaved to: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
