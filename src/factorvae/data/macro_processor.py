"""
Macro feature processor for FactorVAE.

Converts raw macro indicator levels (wide CSV) into stationary,
normalized features with no look-ahead bias.

Design principles (from macro_encoder_plano_v2.md):
  - Log-returns for price-like series (FX, equity indices, volatility, commodities)
  - Level z-score + diff for rate-like series (swap, CDS, NTN-B)
  - Rolling z-score uses only strictly past data (closed='left' / shift(1))
    to avoid subtle leakage where date t's own value enters its z-score
  - MacroNormalizer (in datamodule.py) handles cross-split normalization;
    this module handles intra-series temporal stationarization

Usage:
    from factorvae.data.macro_processor import build_macro_features

    raw = pd.read_csv("data/raw/macro_wide.csv", index_col="date", parse_dates=True)
    features = build_macro_features(raw, window=252)
    # features: DataFrame indexed by date, columns = named macro features
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column name constants — adjust to match actual macro_wide.csv column names
# ---------------------------------------------------------------------------

# Price-like series: only log-returns are used
# Column names must match the raw macro_wide.csv header exactly (spaces, not underscores)
LOGRET_COLS = [
    "USDBRL Curncy",
    "VIX Index",
    "BCOMINTR Index",
    "BCOMAGTR Index",
    "MXEF Index",
    "MOVE Index",
]

# Rate-like series: level z-score AND first difference
LEVEL_AND_DIFF_COLS = [
    "BZDIOVRA Index",                  # DI-over (proxy for short rate; swap 1Y preferred)
    "BRAZIL CDS USD SR 5Y D14 Corp",  # CDS 5Y
]

# Rolling window for z-score computation (trading days)
DEFAULT_WINDOW = 252
# Minimum periods before a z-score is considered valid
MIN_PERIODS = 63


def _rolling_zscore_lagged(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    """
    Z-score rolling using only data strictly before date t.

    shift(1) ensures date t's own value does NOT enter its mean/std,
    preventing the subtle look-ahead where a contemporaneous outlier
    inflates its own normalization denominator.

    Args:
        s:           Time series (already in the target units, e.g. log-ret or level)
        window:      Rolling window size (number of observations)
        min_periods: Minimum valid observations before emitting a result

    Returns:
        Series of z-scores with the same index as s.
    """
    shifted = s.shift(1)
    mu  = shifted.rolling(window, min_periods=min_periods).mean()
    std = shifted.rolling(window, min_periods=min_periods).std().replace(0, np.nan).fillna(1e-8)
    return (s - mu) / std


def build_macro_features(
    raw: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    min_periods: int = MIN_PERIODS,
    logret_cols: list[str] | None = None,
    level_and_diff_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build stationary macro features from a wide-format raw DataFrame.

    Args:
        raw:               Wide DataFrame indexed by date, one column per macro series.
                           Values should be raw levels (prices, rates, index values).
        window:            Rolling window for z-score normalization (trading days).
        min_periods:       Minimum observations before z-score is emitted.
        logret_cols:       Columns to treat as price-like (log-return then z-score).
                           Defaults to module-level LOGRET_COLS.
        level_and_diff_cols: Columns to treat as rate-like (level z-score + diff z-score).
                           Defaults to module-level LEVEL_AND_DIFF_COLS.

    Returns:
        DataFrame indexed by date with one column per derived feature.
        NaN rows (insufficient history) are dropped before returning.

    Feature naming convention:
        <col>_logret_z   — log-return of price-like series, z-scored
        <col>_level_z    — level z-score of rate-like series
        <col>_diff_z     — first-difference z-score of rate-like series
    """
    if logret_cols is None:
        logret_cols = LOGRET_COLS
    if level_and_diff_cols is None:
        level_and_diff_cols = LEVEL_AND_DIFF_COLS

    # Forward-fill to cover non-trading-day gaps (weekends, holidays)
    df = raw.sort_index().ffill()

    out: dict[str, pd.Series] = {}

    # --- Price-like series: log-return → rolling z-score ---
    for col in logret_cols:
        if col not in df.columns:
            continue
        logret = np.log(df[col] / df[col].shift(1))
        out[f"{col}_logret_z"] = _rolling_zscore_lagged(logret, window, min_periods)

    # --- Rate-like series: level z-score + diff z-score ---
    for col in level_and_diff_cols:
        if col not in df.columns:
            continue
        level = df[col]
        diff  = level.diff()
        out[f"{col}_level_z"] = _rolling_zscore_lagged(level, window, min_periods)
        out[f"{col}_diff_z"]  = _rolling_zscore_lagged(diff,  window, min_periods)

    result = pd.DataFrame(out)
    return result.dropna(how="all")


def build_macro_long(
    raw: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    min_periods: int = MIN_PERIODS,
    logret_cols: list[str] | None = None,
    level_and_diff_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Convenience wrapper: builds macro features and converts to long format.

    Returns:
        Long-format DataFrame with columns [date, feature_name, value],
        compatible with data/processed/macro.parquet.
    """
    wide = build_macro_features(
        raw, window, min_periods, logret_cols, level_and_diff_cols
    )
    long = (
        wide
        .reset_index()
        .rename(columns={"index": "date"})
        .melt(id_vars="date", var_name="feature_name", value_name="value")
        .dropna(subset=["value"])
    )
    long["date"] = pd.to_datetime(long["date"])
    return long.sort_values(["date", "feature_name"]).reset_index(drop=True)
