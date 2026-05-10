"""
Orchestrator for the three experiments described in the FactorVAE paper.

Assumes that evaluate.py has already been run so that:
  - results/predictions/predictions.parquet  (FactorVAE test predictions)
  - benchmarks/predictions/*.parquet         (benchmark model predictions)

are present on disk.

Experiment 1 — Cross-Sectional Returns Prediction (Rank IC / Rank ICIR)
    Evaluates the signal quality (Rank IC, Rank ICIR) of all models.
    Outputs: RIC_comparison_ic.png, RIC_rolling_rank_ic.png,
             comparison_table.csv

Experiment 2 — Robustness (holdout-retrain)
    Removes m stocks from training, retrains ALL models, evaluates IC on
    the held-out stocks only. Shared held-out tickers per trial ensure
    fair cross-model comparison.
    Outputs: ROB_holdout_m{m}.png, ROB_holdout_comparison.csv,
             ROB_holdout_summary.png

Experiment 3 — Portfolio Backtest (TopK-Drop + TDrisk)
    Constructs portfolios using predicted returns (and risk-adjusted returns
    for TDrisk). Compares performance metrics across all models.
    Outputs: BKT_comparison_performance.png, BKT_comparison_strategy.png,
             BKT_cumulative_return.png, BKT_cumulative_excess_return.png

Experiments 1 and 3 are produced in a single call to
run_backtest_from_predictions() — they share the same comparison table.

Usage:
    # Run all three experiments (Exp 2 is compute-heavy):
    python scripts/run_experiments.py

    # Skip robustness (fast, Exp 1 + 3 only):
    python scripts/run_experiments.py --skip-robustness

    # Custom holdout sizes and trial count:
    python scripts/run_experiments.py --m 20 --trials 3

    # Quick smoke-test (small epoch budget):
    python scripts/run_experiments.py --m 20 --trials 1 --max-epochs 5
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
        description="Run FactorVAE paper experiments 1, 2, and 3.",
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
        "--skip-robustness",
        action="store_true",
        help="Skip Experiment 2 (holdout-retrain is compute-heavy).",
    )
    parser.add_argument(
        "--m",
        type=int,
        nargs="+",
        default=[20],
        help="Holdout sizes for Experiment 2 (e.g. --m 20). Default: 20.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of holdout trials per m value in Experiment 2. Default: 3.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max_epochs from config for Experiment 2 (quick tests).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed for Experiment 2.",
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

    # Create subdirectories with clear naming: full_universe vs. robustness_missing
    full_universe_dir = out_dir / "full_universe"
    full_universe_dir.mkdir(parents=True, exist_ok=True)
    robustness_missing_dir = out_dir / "robustness_missing"
    robustness_missing_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*68}")
    print(f"  Run ID: {manager.run_id}")
    print(f"  Output directory: {out_dir.relative_to(ROOT)}")
    print(f"  |- Exp 1&3 (full universe):   {full_universe_dir.relative_to(ROOT)}")
    print(f"  `- Exp 2 (robustness/missing): {robustness_missing_dir.relative_to(ROOT)}")
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
    # Experiments 1 & 3 — IC comparison + portfolio backtest
    # (single call; TDrisk row is included automatically when eta > 0)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  Experiment 1: Cross-Sectional Returns Prediction (Rank IC)")
    print("  Experiment 3: Portfolio Backtest (TopK-Drop + TDrisk)")
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
    exp3_outputs = [
        full_universe_dir / "BKT_comparison_performance.png",
        full_universe_dir / "BKT_comparison_strategy.png",
        full_universe_dir / "BKT_cumulative_return.png",
        full_universe_dir / "BKT_cumulative_excess_return.png",
    ]
    print("\n── Experiment 1 outputs ────────────────────────────────────────────")
    for p in exp1_outputs:
        status = "OK" if p.exists() else "MISSING"
        print(f"   [{status}]  {p.relative_to(ROOT)}")

    print("── Experiment 3 outputs ────────────────────────────────────────────")
    for p in exp3_outputs:
        status = "OK" if p.exists() else "MISSING"
        print(f"   [{status}]  {p.relative_to(ROOT)}")

    # ─────────────────────────────────────────────────────────────────────────
    # Experiment 2 — Holdout robustness (compute-heavy)
    # ─────────────────────────────────────────────────────────────────────────
    if args.skip_robustness:
        print("\n[Experiment 2 skipped — use --skip-robustness=False to run]")
    else:
        print("\n" + "=" * 68)
        print("  Experiment 2: Robustness (Holdout-Retrain)")
        print(f"  m values = {args.m}  |  trials = {args.trials}")
        if args.max_epochs:
            print(f"  max_epochs override = {args.max_epochs}")
        print("=" * 68)
        print("  NOTE: Each trial retrains ALL models. This may take a long time.")
        print(f"  Estimated retrains: {len(args.m) * args.trials} × (FactorVAE + MLP + GRU + Linear)")

        from factorvae.evaluation.robustness import robustness_holdout_all_models
        import numpy as np

        all_results: dict[int, list[dict]] = {}
        aggregate_rows: list[dict] = []

        for m_val in args.m:
            print(f"\n{'─' * 68}")
            print(f"  Running Experiment 2: m={m_val}")
            print(f"{'─' * 68}")
            trials = robustness_holdout_all_models(
                config=config,
                m=m_val,
                n_trials=args.trials,
                seed=args.seed,
                max_epochs_override=args.max_epochs,
                progress=True,
            )
            all_results[m_val] = trials

            # Save per-m CSV
            csv_rows: list[dict] = []
            for tr in trials:
                for model_name, metrics in tr["results"].items():
                    csv_rows.append({
                        "m":         m_val,
                        "trial":     tr["trial"],
                        "model":     model_name,
                        "rank_ic":   metrics["rank_ic"],
                        "rank_icir": metrics["rank_icir"],
                        "held_out":  ";".join(tr["held_out"]),
                    })
            per_m_csv = robustness_missing_dir / f"ROB_holdout_m{m_val}.csv"
            pd.DataFrame(csv_rows).to_csv(per_m_csv, index=False)
            print(f"Results saved → {per_m_csv.relative_to(ROOT)}")

            # Aggregate rows
            if trials:
                for mo in trials[0]["results"]:
                    ic_vals   = [t["results"][mo]["rank_ic"]   for t in trials]
                    icir_vals = [t["results"][mo]["rank_icir"] for t in trials]
                    ic_vals   = [v for v in ic_vals   if not np.isnan(v)]
                    icir_vals = [v for v in icir_vals if not np.isnan(v)]
                    aggregate_rows.append({
                        "m":              m_val,
                        "model":          mo,
                        "n_trials":       len(trials),
                        "rank_ic_mean":   float(np.mean(ic_vals))         if ic_vals   else float("nan"),
                        "rank_ic_std":    float(np.std(ic_vals, ddof=1))  if len(ic_vals)  > 1 else 0.0,
                        "rank_icir_mean": float(np.mean(icir_vals))       if icir_vals else float("nan"),
                    })

        # Aggregate CSV
        agg_csv = robustness_missing_dir / "ROB_holdout_comparison.csv"
        pd.DataFrame(aggregate_rows).to_csv(agg_csv, index=False)
        print(f"\nAggregate results saved → {agg_csv.relative_to(ROOT)}")

        # Figures — reuse robustness_holdout figure functions
        from scripts.robustness_holdout import (
            _load_baseline_ic,
            _make_figure_per_m,
            _make_summary_figure,
        )
        baseline_ic = _load_baseline_ic(predictions_path)
        for m_val, trials in all_results.items():
            _make_figure_per_m(
                trials=trials,
                m=m_val,
                baseline_ic=baseline_ic,
                out_path=robustness_missing_dir / f"ROB_holdout_m{m_val}.png",
            )
        if all_results:
            _make_summary_figure(
                all_results=all_results,
                baseline_ic=baseline_ic,
                out_path=robustness_missing_dir / "ROB_holdout_summary.png",
            )

        exp2_outputs = (
            [robustness_missing_dir / f"ROB_holdout_m{m_val}.png" for m_val in args.m]
            + [robustness_missing_dir / "ROB_holdout_comparison.csv", robustness_missing_dir / "ROB_holdout_summary.png"]
        )
        print("\n── Experiment 2 outputs ────────────────────────────────────────────")
        for p in exp2_outputs:
            status = "OK" if p.exists() else "MISSING"
            print(f"   [{status}]  {p.relative_to(ROOT)}")

    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  All experiments complete.")
    print("=" * 68)
    print(f"\n✓ Results saved to: {out_dir.relative_to(ROOT)}/")
    print(f"  ├─ full_universe/       (Exp 1&3: IC comparison + portfolio backtest)")
    if not args.skip_robustness:
        print(f"  └─ robustness_missing/  (Exp 2: Holdout-retrain robustness test, m=50)")
    print()


if __name__ == "__main__":
    main()
