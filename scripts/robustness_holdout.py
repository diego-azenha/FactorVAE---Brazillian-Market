"""
Paper-faithful multi-model holdout-retrain robustness test (Experiment 2).

For each value of m, randomly removes m stocks from the training set, retrains
ALL models (FactorVAE, GRU, IPCA, CA) from scratch, then evaluates Rank IC
exclusively on those held-out stocks in the test set.

Held-out tickers are SHARED across models within each trial, ensuring a fair
cross-model comparison of out-of-sample generalisation.

Usage:
    python scripts/robustness_holdout.py --m 20 --trials 3
    python scripts/robustness_holdout.py --m 20 --trials 1 --max-epochs 5  # quick test
    python scripts/robustness_holdout.py --m 20 --trials 3 --baseline results/predictions/predictions.parquet
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
import numpy as np
import yaml

from factorvae.evaluation.robustness import robustness_holdout_all_models
from factorvae.evaluation.metrics import compute_rank_ic
from factorvae.evaluation.plot_style import (
    PALETTE, TEXT_PRIMARY, TEXT_SECONDARY,
    add_brand_bar, add_footer, add_title, apply_style, finalize_axes,
)

ROOT = _ROOT

# Model display order and colors for figures
_MODEL_ORDER  = ["FactorVAE", "GRU", "IPCA", "CA"]
_MODEL_COLORS = {
    "FactorVAE": "#E8192E",
    "GRU":       "#003f88",  # deep navy
    "IPCA":      "#1a6eb5",  # medium blue
    "CA":        "#5b9fd4",  # light blue
}


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


def _make_figure_per_m(
    trials: list[dict],
    m: int,
    baseline_ic: float | None,
    out_path: Path,
) -> None:
    """
    Grouped bar chart: x-axis = models, bars = per-trial IC.
    Includes mean ± std band per model and optional baseline line.
    """
    apply_style()

    models = [mo for mo in _MODEL_ORDER if mo in trials[0]["results"]]
    n_trials = len(trials)
    x = np.arange(len(models))
    bar_w = 0.8 / max(n_trials, 1)
    offsets = np.linspace(-(n_trials - 1) / 2, (n_trials - 1) / 2, n_trials) * bar_w

    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.10, right=0.97)

    for ti, trial_data in enumerate(trials):
        ic_vals = [trial_data["results"].get(mo, {}).get("rank_ic", float("nan")) for mo in models]
        bar_colors = [_MODEL_COLORS.get(mo, PALETTE[-1]) for mo in models]
        ax.bar(x + offsets[ti], ic_vals, width=bar_w * 0.85,
               color=bar_colors, alpha=0.7 + 0.15 * ti / max(n_trials - 1, 1),
               label=f"Trial {ti + 1}", zorder=3)

    # Mean ± std per model
    for mi, mo in enumerate(models):
        vals = [t["results"].get(mo, {}).get("rank_ic", float("nan")) for t in trials]
        vals = [v for v in vals if not np.isnan(v)]
        if not vals:
            continue
        mean_v = float(np.mean(vals))
        std_v  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ax.plot([x[mi] - 0.35, x[mi] + 0.35], [mean_v, mean_v],
                color="black", linewidth=1.2, zorder=5)
        ax.fill_between([x[mi] - 0.35, x[mi] + 0.35],
                        mean_v - std_v, mean_v + std_v,
                        color="black", alpha=0.08, zorder=4)

    if baseline_ic is not None:
        ax.axhline(baseline_ic, color=PALETTE[0], linewidth=1.2, linestyle="--",
                   label=f"FactorVAE IC universo completo: {baseline_ic:+.4f}")

    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.6, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Rank IC nas ações excluídas do treino")
    ax.legend(fontsize=8, frameon=False)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    finalize_axes(ax, y_right=False)

    add_brand_bar(fig)
    add_title(
        fig,
        f"Robustez (Exp. 2): Rank IC em {m} ações excluídas do treino",
        subtitle=f"{n_trials} trial(s) · todos os modelos retreinados sem as ações excluídas · universo B3",
    )
    add_footer(fig, source="Economatica. Cálculos do autor")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Figure saved → {out_path.relative_to(ROOT)}")


def _make_summary_figure(
    all_results: dict[int, list[dict]],
    baseline_ic: float | None,
    out_path: Path,
) -> None:
    """
    Summary figure: grouped bar chart across m values.
    x-axis = model names, grouped bars = m values.
    """
    apply_style()

    m_values = sorted(all_results.keys())
    all_trial_lists = [all_results[m] for m in m_values]
    models = [mo for mo in _MODEL_ORDER
               if any(mo in t["results"] for tl in all_trial_lists for t in tl)]

    x      = np.arange(len(models))
    n_m    = len(m_values)
    bar_w  = 0.8 / max(n_m, 1)
    offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * bar_w

    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.10, right=0.97)

    m_colors = [PALETTE[0], "#c0392b", "#8e44ad", "#16a085"]
    for mi_idx, (m_val, trial_list) in enumerate(zip(m_values, all_trial_lists)):
        means, stds = [], []
        for mo in models:
            vals = [t["results"].get(mo, {}).get("rank_ic", float("nan")) for t in trial_list]
            vals = [v for v in vals if not np.isnan(v)]
            means.append(float(np.mean(vals)) if vals else float("nan"))
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        color = m_colors[mi_idx % len(m_colors)]
        ax.bar(x + offsets[mi_idx], means, width=bar_w * 0.85,
               color=color, alpha=0.80, label=f"m={m_val}", zorder=3)
        ax.errorbar(x + offsets[mi_idx], means, yerr=stds,
                    fmt="none", color="black", capsize=3, linewidth=0.8, zorder=5)

    if baseline_ic is not None:
        ax.axhline(baseline_ic, color=PALETTE[0], linewidth=1.2, linestyle="--",
                   label=f"IC universo completo: {baseline_ic:+.4f}")

    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.6, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Rank IC médio nas ações excluídas")
    ax.legend(fontsize=8.5, frameon=False)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    finalize_axes(ax, y_right=False)

    add_brand_bar(fig)
    add_title(
        fig,
        "Robustez (Exp. 2): Generalização a ações não vistas no treino",
        subtitle="Rank IC por modelo e tamanho do holdout · universo B3",
    )
    add_footer(fig, source="Economatica. Cálculos do autor")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Figure saved → {out_path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-model holdout-retrain robustness test (Experiment 2)."
    )
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument(
        "--m", type=int, nargs="+", default=[20],
        help="One or more holdout sizes (e.g. --m 20). Default: 20.",
    )
    parser.add_argument(
        "--trials", type=int, default=3,
        help="Number of independent holdout trials per m value. Each trial = one full retrain.",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=None,
        help="Override max_epochs from config (use small value for quick tests).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--baseline",
        default=str(ROOT / "results" / "predictions" / "predictions.parquet"),
        help="Path to predictions.parquet for full-universe IC baseline (FactorVAE).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Full-universe baseline IC (FactorVAE trained on full universe) ────────
    baseline_ic = _load_baseline_ic(Path(args.baseline))
    if baseline_ic is not None:
        print(f"Full-universe FactorVAE Rank IC baseline: {baseline_ic:+.4f}")
    else:
        print("No predictions.parquet found — baseline IC line will not be shown.")

    # ── Collect all results (keyed by m) ──────────────────────────────────────
    all_results: dict[int, list[dict]] = {}
    aggregate_rows: list[dict] = []

    for m_val in args.m:
        print(f"\n{'=' * 68}")
        print(f"  Running robustness holdout: m={m_val}")
        print(f"{'=' * 68}")

        trials = robustness_holdout_all_models(
            config=config,
            m=m_val,
            n_trials=args.trials,
            seed=args.seed,
            max_epochs_override=args.max_epochs,
            progress=True,
        )
        all_results[m_val] = trials

        # ── Per-m CSV ─────────────────────────────────────────────────────
        csv_rows: list[dict] = []
        for tr in trials:
            for model_name, metrics in tr["results"].items():
                csv_rows.append({
                    "m":          m_val,
                    "trial":      tr["trial"],
                    "model":      model_name,
                    "rank_ic":    metrics["rank_ic"],
                    "rank_icir":  metrics["rank_icir"],
                    "held_out":   ";".join(tr["held_out"]),
                })
        per_m_csv = fig_dir / f"ROB_holdout_m{m_val}.csv"
        pd.DataFrame(csv_rows).to_csv(per_m_csv, index=False)
        print(f"Results saved → {per_m_csv.relative_to(ROOT)}")

        # ── Per-m figure ──────────────────────────────────────────────────
        _make_figure_per_m(
            trials=trials,
            m=m_val,
            baseline_ic=baseline_ic,
            out_path=fig_dir / f"ROB_holdout_m{m_val}.png",
        )

        # ── Summary stats for aggregate CSV ──────────────────────────────
        models_in_trial = list(trials[0]["results"].keys()) if trials else []
        for mo in models_in_trial:
            ic_vals   = [t["results"][mo]["rank_ic"]   for t in trials]
            icir_vals = [t["results"][mo]["rank_icir"] for t in trials]
            ic_vals   = [v for v in ic_vals   if not np.isnan(v)]
            icir_vals = [v for v in icir_vals if not np.isnan(v)]
            aggregate_rows.append({
                "m":             m_val,
                "model":         mo,
                "n_trials":      len(trials),
                "rank_ic_mean":  float(np.mean(ic_vals))              if ic_vals   else float("nan"),
                "rank_ic_std":   float(np.std(ic_vals,  ddof=1))      if len(ic_vals)  > 1 else 0.0,
                "rank_icir_mean":float(np.mean(icir_vals))            if icir_vals else float("nan"),
            })

        # ── Per-m summary ─────────────────────────────────────────────────
        print(f"\n── Summary (m={m_val}) {'─' * 50}")
        for mo in models_in_trial:
            ic_vals = [t["results"][mo]["rank_ic"] for t in trials]
            ic_vals = [v for v in ic_vals if not np.isnan(v)]
            mean_v  = np.mean(ic_vals) if ic_vals else float("nan")
            std_v   = np.std(ic_vals, ddof=1) if len(ic_vals) > 1 else 0.0
            print(f"   {mo:<20}  Rank IC = {mean_v:+.4f} ± {std_v:.4f}")
        if baseline_ic is not None:
            fv_vals = [t["results"].get("FactorVAE", {}).get("rank_ic", float("nan")) for t in trials]
            fv_vals = [v for v in fv_vals if not np.isnan(v)]
            if fv_vals:
                degradation = np.mean(fv_vals) - baseline_ic
                print(f"   FactorVAE degradation vs full universe: {degradation:+.4f}")
        print("─" * 68)

    # ── Aggregate CSV (all m values together) ────────────────────────────────
    agg_csv = fig_dir / "ROB_holdout_comparison.csv"
    pd.DataFrame(aggregate_rows).to_csv(agg_csv, index=False)
    print(f"\nAggregate results saved → {agg_csv.relative_to(ROOT)}")

    # ── Summary figure (all m values together) ───────────────────────────────
    if len(all_results) > 0:
        _make_summary_figure(
            all_results=all_results,
            baseline_ic=baseline_ic,
            out_path=fig_dir / "ROB_holdout_summary.png",
        )


if __name__ == "__main__":
    main()
