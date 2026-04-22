"""
Dataset classes for FactorVAE.

SyntheticDataset: generates data with known factor structure for testing.
RealDataset: reads processed parquet files; requires data/processed/ to exist.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────
# Synthetic Dataset
# ─────────────────────────────────────────────────────────────

class SyntheticDataset(Dataset):
    """
    Generates a fixed set of cross-sections with known factor structure.

    Each sample represents one trading date s:
        x[s]: (N, T, C)  — random features with embedded factor signal
        y[s]: (N,)        — y = alpha + beta @ z + noise
        mask[s]: (N,)     — all True (no missing stocks)

    The generator uses a fixed seed per sample so results are reproducible.
    """

    def __init__(
        self,
        num_dates: int = 100,
        N: int = 80,
        T: int = 20,
        C: int = 10,
        K_true: int = 4,
        seed: int = 42,
    ):
        self.num_dates = num_dates
        self.N = N
        self.T = T
        self.C = C
        self.K_true = K_true
        self.seed = seed

        # Pre-generate all samples for speed
        rng = torch.Generator()
        rng.manual_seed(seed)
        self._x: list[Tensor] = []
        self._y: list[Tensor] = []

        for s in range(num_dates):
            x, y = self._generate_sample(s, rng)
            self._x.append(x)
            self._y.append(y)

    def _generate_sample(self, s: int, rng: torch.Generator) -> tuple[Tensor, Tensor]:
        alpha = torch.randn(self.N, generator=rng) * 0.02
        beta = torch.randn(self.N, self.K_true, generator=rng) * 0.5
        z = torch.randn(self.K_true, generator=rng)
        noise = torch.randn(self.N, generator=rng) * 0.01
        y = alpha + beta @ z + noise  # (N,)

        x = torch.randn(self.N, self.T, self.C, generator=rng)
        # Embed factor signal in last timestep, first K_true features
        for k in range(self.K_true):
            x[:, -1, k] += beta[:, k] * z[k]

        return x.float(), y.float()

    def __len__(self) -> int:
        return self.num_dates

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        x = self._x[idx]
        y = self._y[idx]
        mask = torch.ones(self.N, dtype=torch.bool)
        return x, y, mask


# ─────────────────────────────────────────────────────────────
# Real Dataset
# ─────────────────────────────────────────────────────────────

class RealDataset(Dataset):
    """
    Reads features.parquet, returns.parquet, universe.parquet from processed_dir.

    Builds a list of trading dates in [start_date, end_date] for which at least
    one ticker is universe-valid AND has a complete T-day feature lookback.

    Each __getitem__(i) returns (x, y, mask) for the i-th trading date s:

        x     : (N_s, T, C) — cross-sectionally z-scored features
        y     : (N_s,)      — cross-sectionally z-scored forward returns
        mask  : (N_s,)      — True for every ticker in universe (always True here)

    Public attributes (used by evaluate.py):
        trading_dates      : list[pd.Timestamp] — one per __getitem__ index
        universe_by_date   : dict[pd.Timestamp, list[str]] — tickers at each date
    """

    def __init__(
        self,
        processed_dir: str | Path,
        start_date: str,
        end_date: str,
        sequence_length: int = 20,
        feature_cols: list[str] | None = None,
    ):
        processed_dir = Path(processed_dir)

        # ── Load parquets ────────────────────────────────────────────────
        features_long: pd.DataFrame = pd.read_parquet(processed_dir / "features.parquet")
        returns_long: pd.DataFrame  = pd.read_parquet(processed_dir / "returns.parquet")
        universe_long: pd.DataFrame = pd.read_parquet(processed_dir / "universe.parquet")

        # Ensure datetime
        for df in (features_long, returns_long, universe_long):
            df["date"] = pd.to_datetime(df["date"])

        # Determine feature columns
        if feature_cols is None:
            feature_cols = [c for c in features_long.columns if c not in ("date", "ticker")]
        self.feature_cols: list[str] = feature_cols
        self.C: int = len(feature_cols)
        self.T: int = sequence_length

        # ── Build wide feature dicts: ticker → DataFrame(date → features) ─
        # Index by date for fast O(1) lookups in __getitem__
        features_long = features_long.set_index("date")
        self._features_by_ticker: dict[str, pd.DataFrame] = {
            ticker: grp[feature_cols].sort_index()
            for ticker, grp in features_long.groupby("ticker")
        }

        # ── Build return lookup: (date, ticker) → forward_return ─────────
        returns_long = returns_long.set_index(["date", "ticker"])
        self._returns: pd.Series = returns_long["forward_return"]

        # ── Collect all sorted trading dates present in features ──────────
        all_dates_sorted: np.ndarray = np.sort(features_long.index.unique().to_numpy())

        # ── Resolve valid (date, tickers) in [start_date, end_date] ──────
        start_ts = pd.Timestamp(start_date)
        end_ts   = pd.Timestamp(end_date)

        valid_set: set[tuple] = set(
            zip(universe_long["date"], universe_long["ticker"])
        )

        trading_dates: list[pd.Timestamp] = []
        universe_by_date: dict[pd.Timestamp, list[str]] = {}

        for date_np in all_dates_sorted:
            date_ts = pd.Timestamp(date_np)
            if date_ts < start_ts or date_ts > end_ts:
                continue

            # Tickers valid on this date
            tickers = [
                t for t in self._features_by_ticker
                if (date_ts, t) in valid_set
            ]
            if not tickers:
                continue

            # Verify each ticker actually has T lookback rows ending at this date
            confirmed: list[str] = []
            for ticker in tickers:
                feat_df = self._features_by_ticker[ticker]
                loc = feat_df.index.searchsorted(date_ts, side="right")
                if loc >= self.T:
                    window = feat_df.iloc[loc - self.T : loc]
                    if len(window) == self.T and window.index[-1] == date_ts:
                        confirmed.append(ticker)

            if confirmed:
                trading_dates.append(date_ts)
                universe_by_date[date_ts] = sorted(confirmed)

        self.trading_dates:    list[pd.Timestamp]            = trading_dates
        self.universe_by_date: dict[pd.Timestamp, list[str]] = universe_by_date

    # ── Length ────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.trading_dates)

    # ── Item ──────────────────────────────────────────────────────────────
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        date_ts = self.trading_dates[idx]
        tickers = self.universe_by_date[date_ts]
        N = len(tickers)

        # Build x: (N, T, C)
        x_list: list[np.ndarray] = []
        for ticker in tickers:
            feat_df = self._features_by_ticker[ticker]
            loc = feat_df.index.searchsorted(date_ts, side="right")
            window = feat_df.iloc[loc - self.T : loc].values  # (T, C)
            x_list.append(window)
        x_np = np.stack(x_list, axis=0).astype(np.float32)  # (N, T, C)

        # Cross-sectional z-score per timestep per feature
        eps = 1e-8
        mean = x_np.mean(axis=0, keepdims=True)   # (1, T, C)
        std  = x_np.std(axis=0, keepdims=True)    # (1, T, C)
        x_np = (x_np - mean) / (std + eps)

        # Build y: (N,) forward returns — cross-sectionally z-scored
        y_vals: list[float] = []
        for ticker in tickers:
            try:
                y_vals.append(float(self._returns.loc[(date_ts, ticker)]))
            except KeyError:
                y_vals.append(0.0)
        y_np = np.array(y_vals, dtype=np.float32)
        y_mean = y_np.mean()
        y_std  = y_np.std()
        y_np = (y_np - y_mean) / (y_std + eps)

        x    = torch.from_numpy(x_np)
        y    = torch.from_numpy(y_np)
        mask = torch.ones(N, dtype=torch.bool)
        return x, y, mask
