"""
Orchestrator for the FactorVAE paper experiments.

Assumes that evaluate.py has already been run so that:
  - results/predictions/predictions.parquet  (FactorVAE test predictions)
  - benchmarks/predictions/*.parquet         (benchmark model predictions)

are present on disk.

Experiment 1 — Cross-Sectional Returns Prediction (Rank IC / Rank ICIR)
    Evaluates the signal quality (Rank IC, Rank ICIR) of all models.
    Outputs: RIC_comparison_ic.png, RIC_rolling_rank_ic.png,
             comparison_table.csv

Experiment 2 — Portfolio Backtest (TopK-Drop + TDrisk)
    Constructs portfolios using predicted returns (and risk-adjusted returns
    for TDrisk). Compares performance metrics across all models.
    Outputs: BKT_comparison_performance.png, BKT_comparison_strategy.png,
             BKT_cumulative_return.png, BKT_cumulative_excess_return.png

Both experiments are produced in a single call to
run_backtest_from_predictions() — they share the same comparison table.

Usage:
    python scripts/run_experiments.py
    python scripts/run_experiments.py --skip-benchmarks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import yaml

from factorvae.utils.run_manager import RunManager

ROOT = _ROOT


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FactorVAE paper experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config.yaml"),
        help="Path to config.yaml.",
    )
    parser.add_argument(
        "--predictions",
        default=str(ROOT / "results" / "predictions" / "predictions.parquet"),
        help="Path to FactorVAE test predictions parquet.",
    )
    parser.add_argument(
        "--benchmark",
        default=str(ROOT / "data" / "processed" / "benchmark.parquet"),
        help="Path to index benchmark parquet [date, return]. Falls back to EW market.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results" / "figures"),
        help="Output directory for all figures and CSVs.",
    )
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="Skip benchmark training (use existing parquets if present).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Create versioned run directory with timestamp + UUID
    runs_root = ROOT / "results" / "runs"
    manager = RunManager(runs_root, run_name="experiments", auto_create=True)
    out_dir = manager.run_dir  # Use run-specific output directory

    # Create subdirectory for full-universe results
    full_universe_dir = out_dir / "full_universe"
    full_universe_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*68}")
    print(f"  Run ID: {manager.run_id}")
    print(f"  Output directory: {out_dir.relative_to(ROOT)}")
    print(f"  `- {full_universe_dir.relative_to(ROOT)}")
    print(f"{'='*68}")

    predictions_path = Path(args.predictions)
    if not predictions_path.exists():
        print(
            f"ERROR: predictions not found at {predictions_path}.\n"
            "Run `python scripts/evaluate.py` first to generate test predictions."
        )
        sys.exit(1)

    # ───────────────────────────────────────────────────────────────────────────
    # Benchmarks — train and save prediction parquets
    # ───────────────────────────────────────────────────────────────────────────
    if args.skip_benchmarks:
        print("\n[Benchmarks skipped — using existing parquets]")
    else:
        print("\n" + "=" * 68)
        print("  Training benchmark models")
        print("=" * 68)
        from benchmarks.run_benchmarks import main as _run_benchmarks
        _run_benchmarks()

    # ───────────────────────────────────────────────────────────────────────────
    # Experiments 1 & 2 — IC comparison + portfolio backtest
    # (single call; TDrisk row is included automatically when eta > 0)
    # ───────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  Experiment 1: Cross-Sectional Returns Prediction (Rank IC)")
    print("  Experiment 2: Portfolio Backtest (TopK-Drop + TDrisk)")
    print("=" * 68)

    from scripts.backtest import run_backtest_from_predictions  # noqa: E402

    preds = pd.read_parquet(predictions_path)
    preds["date"] = pd.to_datetime(preds["date"])

    eta = config.get("evaluation", {}).get("risk_aversion_eta", 0.0)
    k   = config.get("evaluation", {}).get("top_k", 50)
    n   = config.get("evaluation", {}).get("drop_n", 5)
    print(f"\nConfig: k={k}, n={n}, η={eta:.2f} (TDrisk)")
    print(f"Predictions: {len(preds)} rows, "
          f"{preds['date'].nunique()} test dates, "
          f"{preds['ticker'].nunique()} tickers")

    run_backtest_from_predictions(
        factorvaepreds=preds,
        config=config,
        root=ROOT,
        benchmark_path=Path(args.benchmark),
        out_dir=full_universe_dir,
    )

    exp1_outputs = [
        full_universe_dir / "RIC_comparison_ic.png",
        full_universe_dir / "RIC_rolling_rank_ic.png",
        full_universe_dir / "comparison_table.csv",
    ]
    exp2_outputs = [
        full_universe_dir / "BKT_comparison_performance.png",
        full_universe_dir / "BKT_comparison_strategy.png",
        full_universe_dir / "BKT_cumulative_return.png",
        full_universe_dir / "BKT_cumulative_excess_return.png",
    ]
    print("\n── Experiment 1 outputs ────────────────────────────────────────────")
    for p in exp1_outputs:
        status = "OK" if p.exists() else "MISSING"
        print(f"   [{status}]  {p.relative_to(ROOT)}")

    print("── Experiment 2 outputs ────────────────────────────────────────────")
    for p in exp2_outputs:
        status = "OK" if p.exists() else "MISSING"
        print(f"   [{status}]  {p.relative_to(ROOT)}")

    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  All experiments complete.")
    print("=" * 68)
    print(f"\n\u2713 Results saved to: {out_dir.relative_to(ROOT)}/")
    print(f"  `\u2500 full_universe/   (Exp 1&2: IC comparison + portfolio backtest)")
    print()


if __name__ == "__main__":
    main()
