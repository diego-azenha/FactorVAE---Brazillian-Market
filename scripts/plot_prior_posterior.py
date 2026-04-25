"""
Prior vs Posterior snapshot: K × 3 grid of Gaussian density curves.

Each row is one latent factor k ∈ {0, …, K-1}.
Each column is a training stage: epoch_first, epoch_mid, epoch_final.
Each cell shows two Gaussian density curves:
  - Prior  (dashed, navy)   : mu_prior_k, sigma_prior_k
  - Posterior (solid, red)  : mu_post_k, sigma_post_k

The closer the two curves are, the more the prior has converged to the
posterior's shape — i.e. the predictor has learnt a useful prior.

Usage:
    python scripts/plot_prior_posterior.py
    python scripts/plot_prior_posterior.py --version 7
    python scripts/plot_prior_posterior.py --epochs 1 5 10   # explicit epoch indices
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from factorvae.evaluation.plot_style import (
    PALETTE, TEXT_PRIMARY, TEXT_SECONDARY,
    add_brand_bar, add_footer, add_title, apply_style,
)

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "lightning_logs"

# Colors
COLOR_PRIOR    = PALETTE[1]   # navy — "where the model predicts factors will land"
COLOR_POSTERIOR = PALETTE[0]  # brand red — "where the encoder says factors actually are"


def _latest_version(logs: Path) -> Path:
    versions = sorted(
        [p for p in logs.glob("version_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not versions:
        raise FileNotFoundError(f"No lightning_logs/version_* directories found in {logs}")
    return versions[-1]


def _gaussian_curves(
    mu: float, sigma: float, ax: plt.Axes, color: str, linestyle: str, label: str
) -> None:
    """Plot N(mu, sigma^2) density on ax."""
    x_lo = mu - 4.5 * sigma
    x_hi = mu + 4.5 * sigma
    xs = np.linspace(x_lo, x_hi, 200)
    ys = norm.pdf(xs, mu, max(sigma, 1e-6))
    ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=1.4, label=label)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot prior vs posterior snapshots.")
    parser.add_argument("--version", type=int, default=None,
                        help="Version number of lightning_logs (default: latest).")
    parser.add_argument("--epochs", type=int, nargs="+", default=None,
                        help="Up to 3 epoch indices to show (0-based). Default: auto.")
    args = parser.parse_args()

    version_dir = (
        LOGS / f"version_{args.version}"
        if args.version is not None
        else _latest_version(LOGS)
    )
    metrics_path = version_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.csv not found in {version_dir}")

    metrics = pd.read_csv(metrics_path)

    # Keep only rows that have prior/posterior data logged
    if "train_prior_mu_0" not in metrics.columns:
        raise ValueError(
            "Column 'train_prior_mu_0' not found in metrics.csv.\n"
            "Retrain the model — the distribution logging was added in the latest update."
        )
    snap_rows = metrics.dropna(subset=["train_prior_mu_0"]).copy()
    if snap_rows.empty:
        raise ValueError("No rows with prior/posterior data found in metrics.csv.")

    snap_rows = snap_rows.set_index("epoch").sort_index()
    available_epochs = list(snap_rows.index)
    n_avail = len(available_epochs)

    # Infer K from columns named train_prior_mu_{k}
    K = sum(1 for c in snap_rows.columns if c.startswith("train_prior_mu_"))
    if K == 0:
        raise ValueError("No train_prior_mu_* columns found.")

    # Select 3 epochs: first, mid, last
    if args.epochs:
        selected = [available_epochs[min(i, n_avail - 1)] for i in args.epochs[:3]]
    else:
        idx_first = 0
        idx_mid   = n_avail // 2
        idx_last  = n_avail - 1
        selected  = [
            available_epochs[idx_first],
            available_epochs[idx_mid],
            available_epochs[idx_last],
        ]
    # Deduplicate while preserving order
    seen = set()
    selected = [e for e in selected if not (e in seen or seen.add(e))]
    n_cols = len(selected)

    stage_labels = ["Início", "Meio", "Final"] if n_cols == 3 else [f"Época {e}" for e in selected]

    apply_style()

    cell_h = 2.0
    cell_w = 3.5
    fig_h  = max(cell_h * K + 1.5, 6.0)
    fig_w  = cell_w * n_cols + 1.0

    fig, axes = plt.subplots(K, n_cols, figsize=(fig_w, fig_h))
    if K == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    fig.subplots_adjust(
        top=1.0 - 1.6 / fig_h,
        bottom=0.06,
        left=0.08,
        right=0.97,
        hspace=0.55,
        wspace=0.35,
    )

    for col_idx, epoch in enumerate(selected):
        row_data = snap_rows.loc[epoch]
        for k in range(K):
            ax = axes[k, col_idx]

            mu_prior  = float(row_data.get(f"train_prior_mu_{k}",  0.0))
            sig_prior = float(row_data.get(f"train_prior_sig_{k}", 1.0))
            mu_post   = float(row_data.get(f"train_post_mu_{k}",   0.0))
            sig_post  = float(row_data.get(f"train_post_sig_{k}",  1.0))

            # Dynamic x range that covers both distributions
            all_mu  = [mu_prior, mu_post]
            all_sig = [sig_prior, sig_post]
            x_lo = min(all_mu) - 4.5 * max(all_sig)
            x_hi = max(all_mu) + 4.5 * max(all_sig)
            xs   = np.linspace(x_lo, x_hi, 300)

            ax.plot(xs, norm.pdf(xs, mu_prior, max(sig_prior, 1e-6)),
                    color=COLOR_PRIOR, linestyle="--", linewidth=1.4,
                    label="Prior" if (k == 0 and col_idx == 0) else "_nolegend_")
            ax.plot(xs, norm.pdf(xs, mu_post, max(sig_post, 1e-6)),
                    color=COLOR_POSTERIOR, linestyle="-", linewidth=1.4,
                    label="Posterior" if (k == 0 and col_idx == 0) else "_nolegend_")

            ax.set_yticks([])
            ax.spines["bottom"].set_visible(True)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="x", labelsize=7.5, colors="#2D2D2D")
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(bottom=0)

            if col_idx == 0:
                ax.set_ylabel(f"F{k + 1}", fontsize=8.5, color=TEXT_PRIMARY,
                              rotation=0, labelpad=22, va="center")
            if k == 0:
                ax.set_title(f"{stage_labels[col_idx]}\n(época {epoch})",
                             fontsize=9, color=TEXT_PRIMARY, pad=4)

    # Global legend (top-left cell)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=COLOR_PRIOR,     linestyle="--", linewidth=1.4, label="Prior"),
        Line2D([0], [0], color=COLOR_POSTERIOR, linestyle="-",  linewidth=1.4, label="Posterior"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 1.0 - 0.4 / fig_h),
        fontsize=9,
        frameon=False,
    )

    add_brand_bar(fig, y=1.0 - 0.6 / fig_h)
    add_title(
        fig,
        "Prior vs Posterior ao longo do treino",
        subtitle=f"Distribuições gaussianas por fator latente · {K} fatores",
        y_title=1.0 - 1.1 / fig_h,
        y_sub=1.0 - 1.55 / fig_h,
    )
    add_footer(fig, source="Lightning logs. Cálculos do autor", y=0.005)

    out = ROOT / "results" / "figures" / "TRAIN_prior_posterior.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out.relative_to(ROOT)}")
    print(f"  K={K} factors, epochs shown: {selected}")


if __name__ == "__main__":
    main()
