"""
Tests for RealDataset.

All tests skip gracefully when data/processed/ is not available.
Run scripts/build_features.py first to enable these tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

_DATA_MISSING = not (
    (PROCESSED / "features.parquet").exists()
    and (PROCESSED / "returns.parquet").exists()
    and (PROCESSED / "universe.parquet").exists()
)
skip_if_no_data = pytest.mark.skipif(_DATA_MISSING, reason="data/processed/ not available")


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def train_dataset():
    from factorvae.data.dataset import RealDataset
    return RealDataset(
        processed_dir=PROCESSED,
        start_date="2010-01-01",
        end_date="2018-12-31",
        sequence_length=20,
    )


@pytest.fixture(scope="module")
def small_dataset():
    """Tiny date range so tests are fast."""
    from factorvae.data.dataset import RealDataset
    return RealDataset(
        processed_dir=PROCESSED,
        start_date="2015-01-01",
        end_date="2015-06-30",
        sequence_length=20,
    )


# ─────────────────────────────────────────────────────────────
# Structural / shape tests
# ─────────────────────────────────────────────────────────────

@skip_if_no_data
def test_dataset_non_empty(train_dataset):
    assert len(train_dataset) > 0, "Dataset should have at least one trading date"


@skip_if_no_data
def test_item_shapes(small_dataset):
    assert len(small_dataset) > 0, "small_dataset is empty"
    x, y, mask = small_dataset[0]
    T = small_dataset.T
    C = small_dataset.C

    assert x.ndim == 3, f"x must be 3-D, got {x.ndim}"
    N = x.shape[0]
    assert x.shape == (N, T, C), f"x shape {x.shape} != ({N}, {T}, {C})"
    assert y.shape == (N,), f"y shape {y.shape} != ({N},)"
    assert mask.shape == (N,), f"mask shape {mask.shape} != ({N},)"
    assert mask.dtype == torch.bool


@skip_if_no_data
def test_item_dtypes(small_dataset):
    x, y, mask = small_dataset[0]
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


@skip_if_no_data
def test_num_features_matches_config(small_dataset):
    import yaml
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    expected_C = cfg["model"]["num_features"]
    assert small_dataset.C == expected_C, (
        f"RealDataset.C={small_dataset.C} != config num_features={expected_C}"
    )


# ─────────────────────────────────────────────────────────────
# Normalisation tests
# ─────────────────────────────────────────────────────────────

@skip_if_no_data
def test_x_cross_sectional_zscore(small_dataset):
    """
    After cross-sectional z-scoring: mean over tickers ≈ 0, std ≈ 1
    for every (timestep, feature) pair, provided N > 1.
    """
    x, _, _ = small_dataset[0]      # (N, T, C)
    N = x.shape[0]
    if N < 5:
        pytest.skip("Too few tickers for meaningful normalization check")

    x_np = x.numpy()
    cs_mean = x_np.mean(axis=0)   # (T, C)
    cs_std  = x_np.std(axis=0)    # (T, C)
    np.testing.assert_allclose(cs_mean, 0.0, atol=1e-5,
        err_msg="Cross-sectional mean of x should be ~0")
    np.testing.assert_allclose(cs_std, 1.0, atol=1e-3,
        err_msg="Cross-sectional std of x should be ~1")


@skip_if_no_data
def test_y_cross_sectional_zscore(small_dataset):
    """y should have mean≈0 and std≈1 across tickers."""
    _, y, _ = small_dataset[0]
    N = y.shape[0]
    if N < 5:
        pytest.skip("Too few tickers for meaningful normalization check")
    y_np = y.numpy()
    np.testing.assert_allclose(y_np.mean(), 0.0, atol=1e-5,
        err_msg="Cross-sectional mean of y should be ~0")
    np.testing.assert_allclose(y_np.std(), 1.0, atol=1e-3,
        err_msg="Cross-sectional std of y should be ~1")


# ─────────────────────────────────────────────────────────────
# No-lookahead test
# ─────────────────────────────────────────────────────────────

@skip_if_no_data
def test_no_future_features_in_lookback(small_dataset):
    """
    For every sample i, all T lookback dates must be strictly <= trading_dates[i].
    This rules out accidental forward-looking bias in the feature window.
    """
    import pandas as pd

    for i in range(min(len(small_dataset), 10)):   # spot-check first 10 dates
        date_ts = small_dataset.trading_dates[i]
        tickers = small_dataset.universe_by_date[date_ts]
        for ticker in tickers[:3]:   # check first 3 tickers per date
            feat_df = small_dataset._features_by_ticker[ticker]
            loc = feat_df.index.searchsorted(date_ts, side="right")
            window_dates = feat_df.index[loc - small_dataset.T : loc]
            assert all(d <= date_ts for d in window_dates), (
                f"Future date found in lookback window for {ticker} at {date_ts}"
            )


@skip_if_no_data
def test_no_future_features_in_lookback_all(small_dataset):
    """Verifica TODA a cross-section de TODAS as datas do small_dataset."""
    for i in range(len(small_dataset)):
        date_ts = small_dataset.trading_dates[i]
        tickers = small_dataset.universe_by_date[date_ts]
        for ticker in tickers:
            feat_df = small_dataset._features_by_ticker[ticker]
            loc = feat_df.index.searchsorted(date_ts, side="right")
            window_dates = feat_df.index[loc - small_dataset.T : loc]
            assert window_dates[-1] <= date_ts, (
                f"Feature futura no lookback: {ticker} em {date_ts} "
                f"tem janela até {window_dates[-1]}"
            )


@skip_if_no_data
def test_trading_dates_sorted(small_dataset):
    dates = small_dataset.trading_dates
    for i in range(len(dates) - 1):
        assert dates[i] < dates[i + 1], (
            f"trading_dates not sorted at index {i}: {dates[i]} >= {dates[i+1]}"
        )


@skip_if_no_data
def test_universe_by_date_keys_match_trading_dates(small_dataset):
    for date_ts in small_dataset.trading_dates:
        assert date_ts in small_dataset.universe_by_date, (
            f"{date_ts} in trading_dates but missing from universe_by_date"
        )
        assert len(small_dataset.universe_by_date[date_ts]) > 0


# ─────────────────────────────────────────────────────────────
# Temporal split non-overlap test
# ─────────────────────────────────────────────────────────────

@skip_if_no_data
def test_train_val_test_splits_do_not_overlap():
    import yaml
    from factorvae.data.dataset import RealDataset

    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    dc = cfg["data"]

    train_ds = RealDataset(PROCESSED, dc["train_start"], dc["train_end"])
    val_ds   = RealDataset(PROCESSED, dc["val_start"],   dc["val_end"])
    test_ds  = RealDataset(PROCESSED, dc["test_start"],  dc["test_end"])

    train_dates = set(train_ds.trading_dates)
    val_dates   = set(val_ds.trading_dates)
    test_dates  = set(test_ds.trading_dates)

    assert train_dates.isdisjoint(val_dates),  "Train/Val dates overlap!"
    assert val_dates.isdisjoint(test_dates),   "Val/Test dates overlap!"
    assert train_dates.isdisjoint(test_dates), "Train/Test dates overlap!"
