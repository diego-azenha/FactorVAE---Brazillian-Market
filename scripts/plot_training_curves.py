"""
Post-training diagnostic plots from Lightning's CSV logger.

Reads lightning_logs/version_*/metrics.csv (most recent by default), plots a
2×2 panel:
  [0,0] Total training loss
  [0,1] Reconstruction loss (NLL) with N(0,1) marginal reference line
  [1,0] KL divergence loss
  [1,1] Validation Rank IC

Saves results/figures/training_curves.png.

Usage:
    python scripts/plot_training_curves.py
    python scripts/plot_training_curves.py --version 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "lightning_logs"

# NLL lower bound for a unit-variance Gaussian evaluated at its own mean:
# 0.5 * (1 + log(2π)) ≈ 1.4189
_NLL_FLOOR = 0.5 * (1.0 + 1.8378770664)


def _latest_version(logs: Path) -> Path:
    versions = sorted(
        [p for p in logs.glob("version_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not versions:
        raise FileNotFoundError(f"No lightning_logs/version_* directories found in {logs}")
    return versions[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training diagnostics from Lightning logs.")
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Specific version number to plot (default: latest).",
    )
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

    # CSV has mixed rows: some have train columns, some have val columns.
    # Separate by which key column is present.
    train = metrics.dropna(subset=["train_loss"]).copy()
    val   = metrics.dropna(subset=["val_rank_ic"]).copy()

    if train.empty:
        raise ValueError("No training loss rows found in metrics.csv.")
    if val.empty:
        raise ValueError("No validation rank IC rows found in metrics.csv.")

    train = train.set_index("epoch").sort_index()
    val   = val.set_index("epoch").sort_index()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # [0,0] Total training loss
    axes[0, 0].plot(train.index, train["train_loss"], "o-", markersize=4)
    axes[0, 0].set_title("Total training loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")

    # [0,1] Reconstruction loss with marginal reference
    if "train_loss_recon" in train.columns:
        axes[0, 1].plot(
            train.index, train["train_loss_recon"], "o-", markersize=4, color="tab:blue"
        )
        axes[0, 1].axhline(
            _NLL_FLOOR,
            color="grey",
            linestyle="--",
            linewidth=0.8,
            label=f"N(0,1) marginal ({_NLL_FLOOR:.3f})",
        )
        axes[0, 1].legend(fontsize=8)
    else:
        axes[0, 1].text(0.5, 0.5, "train_loss_recon\nnot logged",
                        ha="center", va="center", transform=axes[0, 1].transAxes)
    axes[0, 1].set_title("Reconstruction loss (NLL)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")

    # [1,0] KL divergence
    if "train_loss_kl" in train.columns:
        axes[1, 0].plot(
            train.index, train["train_loss_kl"], "o-", markersize=4, color="tab:orange"
        )
    else:
        axes[1, 0].text(0.5, 0.5, "train_loss_kl\nnot logged",
                        ha="center", va="center", transform=axes[1, 0].transAxes)
    axes[1, 0].set_title("KL divergence loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")

    # [1,1] Validation Rank IC
    axes[1, 1].plot(
        val.index, val["val_rank_ic"], "o-", markersize=4, color="tab:green"
    )
    axes[1, 1].axhline(0, color="grey", linestyle="--", linewidth=0.5)
    axes[1, 1].set_title("Validation Rank IC")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Rank IC")

    fig.suptitle(f"Training diagnostics — {version_dir.name}", fontsize=12)
    fig.tight_layout()

    out = ROOT / "results" / "figures" / "training_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
