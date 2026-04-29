"""
Train FactorVAE.

Usage:
    python scripts/train.py
    python scripts/train.py --config config.yaml
    python scripts/train.py --synthetic   # use synthetic data (no real data needed)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar, Callback

# Use Tensor Cores on Ampere+ GPUs (RTX 30xx and above) for faster matmul.
# 'high' trades a tiny amount of float32 precision for significant throughput gain.
torch.set_float32_matmul_precision("high")

from factorvae.data.datamodule import FactorVAEDataModule
from factorvae.models.factorvae import FactorVAE
from factorvae.training.lightning_module import FactorVAELightning
from factorvae.utils.seeding import seed_everything

ROOT = Path(__file__).resolve().parents[1]


class EpochProgressCallback(Callback):
    """Print remaining epochs at the start of each epoch."""
    def on_epoch_start(self, trainer: L.Trainer, pl_module) -> None:
        current = trainer.current_epoch
        total = trainer.max_epochs
        remaining = total - current - 1
        print(f"[Epoch {current + 1}/{total}] ({remaining} remaining)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--synthetic", action="store_true",
                        help="Use SyntheticDataset instead of real data")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed_everything(config["training"]["seed"])

    datamodule = FactorVAEDataModule(config, use_synthetic=args.synthetic)
    model = FactorVAE(config)
    lm = FactorVAELightning(model, config)

    # ── Dataset split summary ─────────────────────────────────────────────────
    datamodule.setup()

    def _summarize(name: str, ds) -> None:
        if hasattr(ds, "trading_dates") and len(ds.trading_dates) > 0:
            first   = ds.trading_dates[0].strftime("%Y-%m-%d")
            last    = ds.trading_dates[-1].strftime("%Y-%m-%d")
            n_dates = len(ds.trading_dates)
            mean_N  = sum(len(v) for v in ds.universe_by_date.values()) / n_dates
            print(f"  {name:6s}: {n_dates:4d} dates  [{first} → {last}]  mean N_s = {mean_N:.1f}")
        else:
            print(f"  {name:6s}: {len(ds):4d} synthetic samples")

    print("\n── Dataset splits ──────────────────────────────────────────────────")
    _summarize("Train", datamodule._train)
    _summarize("Val",   datamodule._val)
    _summarize("Test",  datamodule._test)
    print("────────────────────────────────────────────────────────────────────\n")

    callbacks = [
        EpochProgressCallback(),
        RichProgressBar(),
        ModelCheckpoint(
            dirpath=str(ROOT / "results" / "checkpoints"),
            filename="best",
            monitor="val_rank_ic",
            mode="max",
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_rank_ic",
            mode="max",
            patience=50,
            verbose=True,
        ),
    ]

    trainer = L.Trainer(
        max_epochs=config["training"]["max_epochs"],
        callbacks=callbacks,
        enable_model_summary=True,
        log_every_n_steps=1,
    )

    trainer.fit(lm, datamodule=datamodule)
    best_score = callbacks[2].best_model_score
    best_path  = callbacks[2].best_model_path
    score_str  = f"{best_score.item():.4f}" if best_score is not None else "N/A"
    print(f"Best val_rank_ic : {score_str}")
    print(f"Best checkpoint  : {best_path}")


if __name__ == "__main__":
    main()
