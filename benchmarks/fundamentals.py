"""
Features loader for benchmarks (IPCA, CA, GRU).

Loads technical features from features.parquet in processed_dir.
Same 20 features used by FactorVAE.

FundamentalsLoader.get(date, tickers) returns an (N, L) float32 array that is:
  1. Subset to requested tickers.
  2. Cross-sectionally z-scored per feature (over available tickers on that date).
  3. NaN (ticker absent or no data) → 0.

Also exposes shared data-loading helpers used by benchmark modules.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── FundamentalsLoader ────────────────────────────────────────────────────────

class FundamentalsLoader:
    """Load technical features from features.parquet and serve normalised cross-sections."""

    def __init__(self, processed_dir: Path | str):
        """
        Load features.parquet from processed_dir.
        
        processed_dir: Path to data/processed/ folder containing features.parquet
        """
        processed_dir = Path(processed_dir)
        features_path = processed_dir / "features.parquet"
        
        if not features_path.exists():
            raise FileNotFoundError(f"features.parquet not found in {processed_dir}")
        
        # Load features in long format: (date, ticker, feature_cols)
        features_df = pd.read_parquet(features_path)
        features_df["date"] = pd.to_datetime(features_df["date"])
        
        # Identify feature columns (all except date and ticker)
        self.feature_names = [c for c in features_df.columns if c not in ("date", "ticker")]
        self.L: int = len(self.feature_names)
        
        # Store as MultiIndex (date, ticker) for fast O(1) lookups
        self._features: pd.DataFrame = (
            features_df.set_index(["date", "ticker"])[self.feature_names]
            .astype(np.float32)
        )
        
        print(f"  FundamentalsLoader: {self.L} technical features "
              f"({', '.join(self.feature_names[:3])}...) from {features_path.name}")

    def get(self, date: pd.Timestamp, tickers: list[str]) -> np.ndarray:
        """
        Return (N, L) float32 array of cross-sectionally z-scored technical features.

        Missing entries (ticker absent on that date) become 0 after z-scoring.
        """
        N = len(tickers)
        result = np.full((N, self.L), np.nan, dtype=np.float32)

        # Fetch features for all tickers on this date
        try:
            date_data = self._features.loc[date]  # Series or DataFrame
        except KeyError:
            # Date not in features
            result[:] = np.nan
        else:
            if isinstance(date_data, pd.Series):
                # Single ticker on this date
                pass
            else:
                # DataFrame: reindex to requested tickers
                date_data = date_data.reindex(tickers)
                result = date_data.values.astype(np.float32)

        # Cross-sectional z-score per feature; NaN → 0
        for j in range(self.L):
            col = result[:, j]
            valid = ~np.isnan(col)
            if valid.sum() >= 2:
                mu = col[valid].mean()
                sd = col[valid].std()
                if sd > 1e-10:
                    col[valid] = (col[valid] - mu) / (sd + 1e-8)
                else:
                    col[valid] = 0.0
            else:
                col[valid] = 0.0
            col[~valid] = 0.0
            result[:, j] = col

        return result  # (N, L) float32


# ── Shared data-loading helpers ───────────────────────────────────────────────

def load_universe_by_date(config: dict) -> dict[pd.Timestamp, list[str]]:
    """Return {date → [tickers]} for ALL dates in universe.parquet."""
    dc = config["data"]
    universe = pd.read_parquet(Path(dc["processed_dir"]) / "universe.parquet")
    universe["date"] = pd.to_datetime(universe["date"])
    # Keep only is_valid rows if the column exists
    if "is_valid" in universe.columns:
        universe = universe[universe["is_valid"]]
    return (
        universe.groupby("date")["ticker"]
        .apply(list)
        .to_dict()
    )


def load_returns_series(config: dict) -> pd.Series:
    """Return MultiIndex Series keyed by (date, ticker) → forward_return."""
    dc = config["data"]
    ret = pd.read_parquet(Path(dc["processed_dir"]) / "returns.parquet")
    ret["date"] = pd.to_datetime(ret["date"])
    return ret.set_index(["date", "ticker"])["forward_return"]


def dates_in_range(
    ub_date: dict[pd.Timestamp, Any],
    start: str,
    end: str,
) -> list[pd.Timestamp]:
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    return sorted(d for d in ub_date if t0 <= d <= t1)


def build_date_data(
    dates: list[pd.Timestamp],
    ub_date: dict[pd.Timestamp, list[str]],
    ret_s: pd.Series,
    loader: FundamentalsLoader,
    exclude_set: set[str] | None = None,
) -> list[dict]:
    """
    For each date build (Z_t, r_t, x_t) where:
      Z_t : (N_t, L)  — cross-sectionally z-scored fundamentals
      r_t : (N_t,)    — forward returns (non-NaN rows kept)
      x_t : (L,)      — managed portfolio = lstsq(Z_t, r_t)

    Dates with fewer than L+2 valid stocks are skipped.

    Returns list of dicts with keys: date, tickers, Z, r, x.
    """
    exclude_set = exclude_set or set()
    data: list[dict] = []

    for date in dates:
        tickers = [t for t in ub_date.get(date, []) if t not in exclude_set]
        if not tickers:
            continue

        Z = loader.get(date, tickers)  # (N, L)

        r_vals: list[float] = []
        for t in tickers:
            try:
                r_vals.append(float(ret_s.loc[(date, t)]))
            except KeyError:
                r_vals.append(float("nan"))
        r = np.array(r_vals, dtype=np.float32)

        # Drop rows where return is NaN
        valid = ~np.isnan(r)
        Z_v, r_v = Z[valid], r[valid]
        N, L = Z_v.shape

        if N < L + 2 or float(r_v.std()) < 1e-10:
            continue

        # Managed portfolio: OLS of returns on fundamentals
        try:
            x_t = np.linalg.lstsq(Z_v, r_v, rcond=None)[0]  # (L,)
        except np.linalg.LinAlgError:
            continue

        data.append({
            "date":    date,
            "tickers": [t for t, v in zip(tickers, valid) if v],
            "Z":       Z_v.astype(np.float32),
            "r":       r_v.astype(np.float32),
            "x":       x_t.astype(np.float32),
        })

    return data
