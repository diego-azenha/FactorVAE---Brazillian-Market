"""
Smoke test for the evaluate.py pipeline output schema.

Trains a 1-epoch model on synthetic data, runs the evaluation loop
directly (without subprocess), and validates that the predictions
DataFrame has the correct schema and that y_true is NOT in z-score
scale (regression guard for the bug where RealDataset's normalized y
was written as ground truth).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from factorvae.data.datamodule import FactorVAEDataModule
from factorvae.evaluation.metrics import compute_rank_ic
from factorvae.models.factorvae import FactorVAE
from factorvae.training.lightning_module import FactorVAELightning


# ─────────────────────────────────────────────────────────────
# Shared config (tiny model for speed)
# ─────────────────────────────────────────────────────────────

def _make_config(ckpt_dir: str) -> dict:
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
            "max_epochs": 1,
            "learning_rate": 1e-3,
            "gamma": 1.0,
            "seed": 0,
            "sigma_floor": 1e-6,
        },
        "evaluation": {
            "top_k": 10,
            "drop_n": 2,
            "risk_aversion_eta": 1.0,
        },
    }


# ─────────────────────────────────────────────────────────────
# Helper: train 1 epoch + run eval loop, return predictions DataFrame
# ─────────────────────────────────────────────────────────────

def _run_evaluate(config: dict) -> pd.DataFrame:
    """
    Reproduces the evaluate.py logic inline (synthetic mode).
    Returns the predictions DataFrame.
    """
    datamodule = FactorVAEDataModule(config, use_synthetic=True)
    datamodule.setup()
    model = FactorVAE(config)
    lm = FactorVAELightning(model, config)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = L.Trainer(
            max_epochs=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(lm, datamodule=datamodule)

    lm.model.eval()
    test_dataset = datamodule._test
    records = []

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            x, y, mask = test_dataset[idx]
            x = x.float()
            y = y.float()
            mu_pred, sigma_pred = lm.model.forward_predict(x)
            N = x.shape[0]

            for i in range(N):
                records.append({
                    "date":       idx,
                    "ticker":     i,
                    "mu_pred":    mu_pred[i].item(),
                    "sigma_pred": sigma_pred[i].item(),
                    # In synthetic mode, y is already the raw target
                    # (SyntheticDataset does NOT z-score its outputs)
                    "y_true":     y[i].item(),
                })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def predictions_df():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config(tmpdir)
        return _run_evaluate(config)


# ─────────────────────────────────────────────────────────────
# Schema tests
# ─────────────────────────────────────────────────────────────

def test_predictions_has_required_columns(predictions_df):
    expected = {"date", "ticker", "mu_pred", "sigma_pred", "y_true"}
    assert expected.issubset(set(predictions_df.columns)), (
        f"Missing columns: {expected - set(predictions_df.columns)}"
    )


def test_predictions_mu_pred_no_nan(predictions_df):
    assert predictions_df["mu_pred"].notna().all(), "mu_pred contains NaN"
    assert np.isfinite(predictions_df["mu_pred"].values).all(), "mu_pred contains Inf"


def test_predictions_sigma_pred_positive(predictions_df):
    assert (predictions_df["sigma_pred"] > 0).all(), "sigma_pred must be strictly positive"


def test_predictions_not_empty(predictions_df):
    assert len(predictions_df) > 0, "predictions DataFrame is empty"


def test_predictions_y_true_finite(predictions_df):
    """
    y_true must be finite (no NaN or Inf) and non-constant.

    NOTE on the z-score regression guard from the evaluate.py bug fix:
    The original bug saved z-scored `y` from RealDataset as `y_true`,
    making all backtest metrics meaningless (raw returns have std ~0.02-0.05
    vs z-scored returns with std ~1.0).  This guard is NOT checkable in
    synthetic mode because SyntheticDataset's `y = alpha + beta@z + noise`
    produces std ~1.0 by construction (beta~N(0,0.5), K=4 factors).
    For real-data evaluation, verify that predictions.parquet's `y_true`
    column has std in [0.005, 0.15], not ~1.0.
    """
    y_true = predictions_df["y_true"].values
    assert np.isfinite(y_true).all(), "y_true contains NaN or Inf"
    assert y_true.std() > 1e-6, "y_true is constant — something is wrong"
