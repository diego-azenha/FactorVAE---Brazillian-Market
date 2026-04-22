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

import yaml
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from factorvae.data.datamodule import FactorVAEDataModule
from factorvae.models.factorvae import FactorVAE
from factorvae.training.lightning_module import FactorVAELightning
from factorvae.utils.seeding import seed_everything

ROOT = Path(__file__).resolve().parents[1]


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

    callbacks = [
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
            patience=10,
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
    print(f"Best val_rank_ic: {trainer.callback_metrics.get('val_rank_ic', 'N/A')}")


if __name__ == "__main__":
    main()
