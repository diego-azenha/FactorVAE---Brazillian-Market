"""
Paper-faithful holdout-retrain robustness test for FactorVAE.

Randomly removes m stocks from the training set, retrains the model from scratch,
then evaluates Rank IC and Rank ICIR exclusively on those held-out stocks in the
test set. This directly measures whether the model's predictions generalise to
stocks it never saw during training.

Run with m=10 (small holdout) and separately with m=50 (large holdout).
Each trial = one full training run; budget ~n_trials × training_time.

Usage:
    python scripts/robustness_holdout.py --m 10 --trials 3
    python scripts/robustness_holdout.py --m 50 --trials 3
    python scripts/robustness_holdout.py --m 10 --trials 2 --max-epochs 5  # quick test
    python scripts/robustness_holdout.py --m 10 --trials 3 --baseline results/predictions/predictions.parquet
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
import matplotlib.pyplot as plt
import yaml

from factorvae.evaluation.robustness import robustness_holdout_train_test
from factorvae.evaluation.metrics import compute_rank_ic
from factorvae.evaluation.plot_style import (
    PALETTE, TEXT_PRIMARY, TEXT_SECONDARY,
    add_brand_bar, add_footer, add_title, apply_style, finalize_axes,
)

ROOT = _ROOT


def _load_baseline_ic(predictions_path: Path) -> float | None:
    """Compute full-universe Rank IC from a saved predictions.parquet."""
    import torch
    if not predictions_path.exists():
        return None
    preds = pd.read_parquet(predictions_path)
    preds["date"] = pd.to_datetime(preds["date"])
    ics: list[float] = []
    for _, grp in preds.groupby("date"):
        grp = grp.dropna(subset=["y_true"])
        if len(grp) < 2:
            continue
        ics.append(compute_rank_ic(
            y_true=__import__("torch").tensor(grp["y_true"].values, dtype=torch.float32),
            y_pred=__import__("torch").tensor(grp["mu_pred"].values, dtype=torch.float32),
        ))
    return float(np.mean(ics)) if ics else None


def _make_figure(
    results: list[dict],
    m: int,
    baseline_ic: float | None,
    out_path: Path,
) -> None:
    apply_style()

    ic_vals = [r["rank_ic_holdout"] for r in results]
    trial_labels = [f"Trial {r['trial'] + 1}" for r in results]

    ic_mean = float(np.mean(ic_vals))
    ic_std  = float(np.std(ic_vals, ddof=1)) if len(ic_vals) > 1 else 0.0

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.10, right=0.92)

    x = np.arange(len(results))
    bar_color = PALETTE[0]
    ax.bar(x, ic_vals, color=bar_color, alpha=0.85, width=0.5, zorder=3)

    # Mean ± std band
    ax.axhline(ic_mean, color=bar_color, linewidth=1.4, linestyle="--",
               label=f"Média: {ic_mean:+.4f} ± {ic_std:.4f}")
    ax.axhspan(ic_mean - ic_std, ic_mean + ic_std,
               color=bar_color, alpha=0.12, zorder=2)

    if baseline_ic is not None:
        ax.axhline(baseline_ic, color=PALETTE[1], linewidth=1.2, linestyle="-",
                   label=f"IC universo completo: {baseline_ic:+.4f}")

    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.6, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(trial_labels)
    ax.set_ylabel("Rank IC")
    ax.legend(fontsize=8.5, frameon=False)
    finalize_axes(ax, y_right=False)

    add_brand_bar(fig)
    add_title(
        fig,
        f"Robustez: Rank IC em {m} ações excluídas do treino",
        subtitle=f"{len(results)} trial(s) · retrain completo por trial · universo B3",
    )
    add_footer(fig, source="Economatica. Cálculos do autor")

    fig.savefig(out_path)
    plt.close(fig)
    print(f"Figure saved → {out_path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-faithful holdout-retrain robustness test.")
    parser.add_argument("--config",      default=str(ROOT / "config.yaml"))
    parser.add_argument("--m",           type=int, default=10,
                        help="Number of tickers to hold out per trial.")
    parser.add_argument("--trials",      type=int, default=3,
                        help="Number of independent holdout trials (each = one full training).")
    parser.add_argument("--max-epochs",  type=int, default=None,
                        help="Override max_epochs from config (use small value for quick tests).")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument(
        "--baseline",
        default=str(ROOT / "results" / "predictions" / "predictions.parquet"),
        help="Path to predictions.parquet for full-universe IC baseline.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ── Full-universe baseline IC ─────────────────────────────────────────────
    baseline_ic = _load_baseline_ic(Path(args.baseline))
    if baseline_ic is not None:
        print(f"Full-universe Rank IC baseline: {baseline_ic:+.4f}")
    else:
        print("No predictions.parquet found — baseline IC will not be shown.")

    # ── Run holdout trials ────────────────────────────────────────────────────
    results = robustness_holdout_train_test(
        config=config,
        m=args.m,
        n_trials=args.trials,
        seed=args.seed,
        max_epochs_override=args.max_epochs,
        progress=True,
    )

    # ── Save CSV ──────────────────────────────────────────────────────────────
    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = [
        {
            "trial":               r["trial"],
            "m":                   args.m,
            "rank_ic_holdout":     r["rank_ic_holdout"],
            "rank_icir_holdout":   r["rank_icir_holdout"],
            "n_dates_with_holdout":r["n_dates_with_holdout"],
            "held_out":            ";".join(r["held_out"]),
        }
        for r in results
    ]
    csv_path = fig_dir / f"ROB_holdout_m{args.m}.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"Results saved → {csv_path.relative_to(ROOT)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    ic_vals = [r["rank_ic_holdout"] for r in results]
    print(f"\n── Summary (m={args.m}) ─────────────────────────────────────────────")
    print(f"   Rank IC on held-out stocks: {np.mean(ic_vals):+.4f} ± {np.std(ic_vals, ddof=1) if len(ic_vals)>1 else 0:.4f}")
    if baseline_ic is not None:
        degradation = np.mean(ic_vals) - baseline_ic
        print(f"   vs full-universe IC {baseline_ic:+.4f} → degradation: {degradation:+.4f}")
    print("────────────────────────────────────────────────────────────────────")

    # ── Figure ────────────────────────────────────────────────────────────────
    out_png = fig_dir / f"ROB_holdout_m{args.m}.png"
    _make_figure(results, m=args.m, baseline_ic=baseline_ic, out_path=out_png)


if __name__ == "__main__":
    main()
