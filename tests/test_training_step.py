"""Tests for the full Lightning training step."""

from __future__ import annotations

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path
import tempfile

from factorvae.data.datamodule import FactorVAEDataModule
from factorvae.models.factorvae import FactorVAE
from factorvae.training.lightning_module import FactorVAELightning


def _make_config():
    return {
        "data": {
            "processed_dir": "data/processed",
            "train_start": "2010-01-01",
            "train_end": "2017-12-31",
            "val_start": "2018-01-01",
            "val_end": "2018-12-31",
            "test_start": "2019-01-01",
            "test_end": "2025-12-31",
            "sequence_length": 20,
        },
        "model": {
            "num_features": 10,
            "hidden_dim": 16,
            "num_factors": 4,
            "num_portfolios": 32,
            "leaky_relu_slope": 0.1,
        },
        "training": {
            "batch_size": 1,
            "max_epochs": 2,
            "learning_rate": 1e-3,
            "gamma": 1.0,
            "seed": 42,
            "sigma_floor": 1e-6,
        },
        "evaluation": {
            "top_k": 10,
            "drop_n": 2,
            "risk_aversion_eta": 1.0,
        },
    }


def test_fast_dev_run_completes():
    """fast_dev_run=True runs exactly 1 train + 1 val batch without error."""
    config = _make_config()
    datamodule = FactorVAEDataModule(config, use_synthetic=True)
    model = FactorVAE(config)
    lm = FactorVAELightning(model, config)

    trainer = L.Trainer(
        fast_dev_run=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(lm, datamodule=datamodule)


def test_val_rank_ic_logged():
    """After 1 epoch, val_rank_ic must appear in the logged metrics."""
    config = _make_config()
    datamodule = FactorVAEDataModule(config, use_synthetic=True)
    model = FactorVAE(config)
    lm = FactorVAELightning(model, config)

    trainer = L.Trainer(
        max_epochs=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(lm, datamodule=datamodule)

    assert "val_rank_ic" in trainer.callback_metrics, (
        "val_rank_ic was not logged after training epoch"
    )


def test_checkpoint_reload_identical_predictions():
    """Saving and reloading a checkpoint must produce identical forward_predict output."""
    config = _make_config()
    datamodule = FactorVAEDataModule(config, use_synthetic=True)
    datamodule.setup()

    model = FactorVAE(config)
    lm = FactorVAELightning(model, config)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_cb = ModelCheckpoint(dirpath=tmpdir, save_last=True)
        trainer = L.Trainer(
            max_epochs=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            callbacks=[ckpt_cb],
            num_sanity_val_steps=0,
        )
        trainer.fit(lm, datamodule=datamodule)

        # Get a fixed batch
        x, y, mask = datamodule._val[0]
        x = x.unsqueeze(0)

        # Predictions before reload
        lm.model.eval()
        with torch.no_grad():
            mu_before, sig_before = lm.model.forward_predict(x.squeeze(0))

        # Reload from checkpoint
        lm2 = FactorVAELightning.load_from_checkpoint(
            ckpt_cb.last_model_path,
            model=FactorVAE(config),
            config=config,
        )
        lm2.model.eval()
        with torch.no_grad():
            mu_after, sig_after = lm2.model.forward_predict(x.squeeze(0))

        assert torch.allclose(mu_before, mu_after, atol=1e-5)
        assert torch.allclose(sig_before, sig_after, atol=1e-5)
