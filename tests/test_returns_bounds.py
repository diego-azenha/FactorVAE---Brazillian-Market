"""
Tests for forward return data quality.

Verifies that the processed returns.parquet produced by build_features.py
contains only economically plausible values — no artefacts from unadjusted
splits, penny stocks, or near-zero price denominators.

All tests skip gracefully if data/processed/ does not exist.
Run scripts/build_features.py first to enable them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

_DATA_MISSING = not (PROCESSED / "returns.parquet").exists()
skip_if_no_data = pytest.mark.skipif(
    _DATA_MISSING, reason="data/processed/ not available — run scripts/build_features.py first"
)


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "returns.parquet")


@skip_if_no_data
def test_no_extreme_returns(returns):
    """
    No forward return should exceed 50% in absolute value.
    Returns above this threshold are almost certainly data artefacts
    (unadjusted splits, penny-stock division, corporate events).
    """
    MAX_ABS = 0.50
    bad = returns[returns["forward_return"].abs() > MAX_ABS]
    assert len(bad) == 0, (
        f"{len(bad)} rows exceed |return| > {MAX_ABS}: "
        f"max={returns['forward_return'].abs().max():.4f}\n"
        f"Sample:\n{bad.head(5)}"
    )


@skip_if_no_data
def test_no_nan_returns(returns):
    """forward_return must have no NaN after build_features filtering."""
    n_nan = returns["forward_return"].isna().sum()
    assert n_nan == 0, f"Found {n_nan} NaN values in forward_return"


@skip_if_no_data
def test_returns_not_empty(returns):
    assert len(returns) > 0, "returns.parquet is empty"


@skip_if_no_data
def test_returns_reasonable_scale(returns):
    """
    The cross-sectional std of daily returns should be in the range 0.005–0.10.
    Too small: data is not returns. Too large: artefacts remain.
    """
    std = returns["forward_return"].std()
    assert 0.005 <= std <= 0.10, (
        f"Returns std={std:.4f} is outside expected range [0.005, 0.10] — "
        "check for remaining artefacts or data type mismatch."
    )
