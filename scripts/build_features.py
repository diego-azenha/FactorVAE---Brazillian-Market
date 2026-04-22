"""
Build features from raw price/volume CSVs and write parquet files to data/processed/.

Computes 20 technical features per (date, ticker):
  0  ret_1d      : 1-day return
  1  ret_2d      : 2-day return
  2  ret_5d      : 5-day return
  3  ret_10d     : 10-day return
  4  ret_20d     : 20-day return
  5  vol_5d      : realized volatility (std of daily rets, 5-day window)
  6  vol_10d     : realized volatility 10-day
  7  vol_20d     : realized volatility 20-day
  8  vol_ratio   : vol_5d / (vol_20d + eps)
  9  vol_z       : (vol_5d - vol_20d.rolling(60).mean) / (vol_20d.rolling(60).std + eps)
  10 turnover_1d : volume / volume.rolling(20).mean
  11 turnover_5d : volume.rolling(5).mean / volume.rolling(20).mean
  12 vwap_dev    : (close - vwap_proxy) / (vwap_proxy + eps)  vwap_proxy = volume-weighted close 5d
  13 rsi_14      : RSI with period 14
  14 ma_dev_5    : close / close.rolling(5).mean - 1
  15 ma_dev_10   : close / close.rolling(10).mean - 1
  16 ma_dev_20   : close / close.rolling(20).mean - 1
  17 skew_20     : rolling 20-day skewness of daily returns
  18 kurt_20     : rolling 20-day excess kurtosis of daily returns
  19 amihud      : abs(ret_1d) / (turnover_1d + eps)  illiquidity proxy

Output:
  data/processed/features.parquet  cols: [date, ticker, f0..f19]
  data/processed/returns.parquet   cols: [date, ticker, forward_return]
  data/processed/universe.parquet  cols: [date, ticker, is_valid]
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
PRICES_PATH = ROOT / "data" / "prices_wide.csv"
VOLUME_PATH = ROOT / "data" / "volume_wide.csv"
OUT_DIR = ROOT / "data" / "processed"

FEATURE_NAMES = [
    "ret_1d", "ret_2d", "ret_5d", "ret_10d", "ret_20d",
    "vol_5d", "vol_10d", "vol_20d",
    "vol_ratio", "vol_z",
    "turnover_1d", "turnover_5d",
    "vwap_dev",
    "rsi_14",
    "ma_dev_5", "ma_dev_10", "ma_dev_20",
    "skew_20", "kurt_20",
    "amihud",
]
assert len(FEATURE_NAMES) == 20, "Must have exactly 20 features"

T_WARMUP = 60  # rows needed before first valid feature date


def _rsi(ret: pd.Series, period: int = 14) -> pd.Series:
    gain = ret.clip(lower=0)
    loss = (-ret).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - 100 / (1 + rs)


def compute_features(prices: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-ticker features. Returns long-format DataFrame with columns
    [date, ticker, f0..f19] aligned by date.
    """
    eps = 1e-9
    tickers = prices.columns.tolist()
    records: list[pd.DataFrame] = []

    for ticker in tickers:
        close = prices[ticker].dropna()
        vol = volume[ticker].reindex(close.index).fillna(0.0)

        if len(close) < T_WARMUP + 20:
            continue

        ret = close.pct_change()

        # --- returns ---
        r1 = ret
        r2 = close.pct_change(2)
        r5 = close.pct_change(5)
        r10 = close.pct_change(10)
        r20 = close.pct_change(20)

        # --- volatility ---
        v5 = ret.rolling(5).std()
        v10 = ret.rolling(10).std()
        v20 = ret.rolling(20).std()
        vol_ratio = v5 / (v20 + eps)
        vol_z_mean = v20.rolling(60).mean()
        vol_z_std = v20.rolling(60).std()
        vol_z = (v5 - vol_z_mean) / (vol_z_std + eps)

        # --- volume / turnover ---
        vol_ma20 = vol.rolling(20).mean()
        to1 = vol / (vol_ma20 + eps)
        vol_ma5 = vol.rolling(5).mean()
        to5 = vol_ma5 / (vol_ma20 + eps)

        # --- VWAP deviation (5-day volume-weighted close) ---
        vwap_num = (close * vol).rolling(5).sum()
        vwap_den = vol.rolling(5).sum()
        vwap = vwap_num / (vwap_den + eps)
        vwap_dev = (close - vwap) / (vwap + eps)

        # --- RSI-14 ---
        rsi = _rsi(ret)

        # --- MA deviations ---
        ma5 = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        ma20_ma = close.rolling(20).mean()
        ma_dev5 = close / (ma5 + eps) - 1
        ma_dev10 = close / (ma10 + eps) - 1
        ma_dev20 = close / (ma20_ma + eps) - 1

        # --- higher moments of returns (20d window) ---
        skew20 = ret.rolling(20).skew()
        kurt20 = ret.rolling(20).kurt()  # excess kurtosis

        # --- Amihud illiquidity ---
        amihud = r1.abs() / (to1 + eps)

        df_feat = pd.DataFrame(
            {
                "ret_1d": r1,
                "ret_2d": r2,
                "ret_5d": r5,
                "ret_10d": r10,
                "ret_20d": r20,
                "vol_5d": v5,
                "vol_10d": v10,
                "vol_20d": v20,
                "vol_ratio": vol_ratio,
                "vol_z": vol_z,
                "turnover_1d": to1,
                "turnover_5d": to5,
                "vwap_dev": vwap_dev,
                "rsi_14": rsi,
                "ma_dev_5": ma_dev5,
                "ma_dev_10": ma_dev10,
                "ma_dev_20": ma_dev20,
                "skew_20": skew20,
                "kurt_20": kurt20,
                "amihud": amihud,
            },
            index=close.index,
        )
        df_feat.index.name = "date"
        df_feat = df_feat.dropna(how="any")
        df_feat["ticker"] = ticker
        records.append(df_feat.reset_index())

    features_long = pd.concat(records, ignore_index=True)
    features_long["date"] = pd.to_datetime(features_long["date"])
    return features_long


def compute_returns(
    prices: pd.DataFrame,
    min_price: float = 0.10,
    max_abs_return: float = 0.50,
) -> pd.DataFrame:
    """
    Compute forward_return = (p_{t+2} - p_{t+1}) / p_{t+1} for each ticker.
    On trading day t: shift(-1) gives p_{t+1}, shift(-2) gives p_{t+2}.

    Filters applied before saving:
      - Rows where p_{t+1} < min_price are dropped (penny stocks / near-zero
        prices produce artificially huge returns from unadjusted splits).
      - Rows where |forward_return| > max_abs_return are dropped (>50% in one
        day is almost certainly a data artefact, not a real return).
    """
    records = []
    for ticker in prices.columns:
        close = prices[ticker].dropna()
        p_t1 = close.shift(-1)
        p_t2 = close.shift(-2)
        fwd = (p_t2 - p_t1) / p_t1
        # Drop penny-stock rows
        fwd = fwd.where(p_t1 >= min_price)
        # Drop artefact / unadjusted-split rows
        fwd = fwd.where(fwd.abs() <= max_abs_return)
        df = pd.DataFrame({"date": close.index, "ticker": ticker, "forward_return": fwd.values})
        df = df.dropna(subset=["forward_return"])
        records.append(df)
    out = pd.concat(records, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out


def compute_universe(features_long: pd.DataFrame, sequence_length: int = 20) -> pd.DataFrame:
    """
    A (date, ticker) is valid on date s if the feature table contains rows for
    EXACTLY the T=sequence_length consecutive trading days ending at s
    (i.e. the T-day lookback window is fully populated, no gaps).

    Algorithm per ticker:
      1. Sort feature dates for that ticker.
      2. Assign a sequential rank to each date.
      3. Compute a rolling window of size T: if all T dates in the window are
         consecutive (rank of last - rank of first == T-1), the last date is valid.
    """
    records: list[pd.DataFrame] = []

    for ticker, grp in features_long.groupby("ticker", sort=False):
        dates_sorted = grp["date"].sort_values().reset_index(drop=True)
        n = len(dates_sorted)
        if n < sequence_length:
            continue

        # For each position i >= T-1, check that the window [i-T+1 .. i] spans
        # exactly T consecutive dates (no missing days in the feature table).
        # Two dates are consecutive in trading-day terms if their positional
        # difference within the sorted ticker series is 1, so a T-length
        # window [i-T+1 .. i] is gap-free iff position[i] - position[i-T+1] == T-1.
        # Refine: also require no calendar gaps larger than 7 days within the window
        # (handles long halts / delistings that still have sequential index positions).
        valid_dates: list[pd.Timestamp] = []
        for i in range(sequence_length - 1, n):
            window = dates_sorted.iloc[i - sequence_length + 1 : i + 1]
            max_gap = (window.diff().dropna()).dt.days.max()
            if max_gap is not None and max_gap <= 7:  # tolerate weekends + 2 holidays
                valid_dates.append(dates_sorted.iloc[i])

        if valid_dates:
            df = pd.DataFrame({"date": valid_dates, "ticker": ticker, "is_valid": True})
            records.append(df)

    if not records:
        return pd.DataFrame(columns=["date", "ticker", "is_valid"])

    universe = pd.concat(records, ignore_index=True)
    universe["date"] = pd.to_datetime(universe["date"])
    return universe


def main() -> None:
    parser = argparse.ArgumentParser(description="Build processed features from raw CSVs")
    parser.add_argument("--prices", default=str(PRICES_PATH))
    parser.add_argument("--volume", default=str(VOLUME_PATH))
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading prices...")
    prices = pd.read_csv(args.prices, index_col="date", parse_dates=True)
    print(f"  shape: {prices.shape}")

    print("Loading volume...")
    volume = pd.read_csv(args.volume, index_col="date", parse_dates=True)
    print(f"  shape: {volume.shape}")

    # Align index
    common_idx = prices.index.intersection(volume.index)
    prices = prices.loc[common_idx]
    volume = volume.loc[common_idx]

    print("Computing features (this may take a few minutes)...")
    features_long = compute_features(prices, volume)
    print(f"  features rows: {len(features_long):,}  |  dates: {features_long['date'].nunique()}  |  tickers: {features_long['ticker'].nunique()}")

    print("Computing forward returns...")
    returns = compute_returns(prices)
    print(f"  returns rows: {len(returns):,}")

    print("Computing universe (T=20 consecutive lookback)...")
    universe = compute_universe(features_long, sequence_length=20)
    print(f"  universe rows: {len(universe):,}")

    features_long.to_parquet(out_dir / "features.parquet", index=False)
    returns.to_parquet(out_dir / "returns.parquet", index=False)
    universe.to_parquet(out_dir / "universe.parquet", index=False)
    print(f"Saved to {out_dir}/")

    # ── Diagnostics ────────────────────────────────────────────────────────
    print("\n── Diagnostics ──────────────────────────────────────────────")
    print(f"  Feature columns : {[c for c in features_long.columns if c not in ('date','ticker')]}")
    print(f"  C (num features): {len([c for c in features_long.columns if c not in ('date','ticker')])}")

    import yaml as _yaml
    cfg_path = ROOT / "config.yaml"
    splits: dict[str, tuple[str, str]] = {}
    if cfg_path.exists():
        with open(cfg_path) as _f:
            _cfg = _yaml.safe_load(_f)
        dc = _cfg["data"]
        splits = {
            "train": (dc["train_start"], dc["train_end"]),
            "val":   (dc["val_start"],   dc["val_end"]),
            "test":  (dc["test_start"],  dc["test_end"]),
        }

    for split_name, (s_start, s_end) in splits.items():
        mask = (
            (universe["date"] >= pd.Timestamp(s_start))
            & (universe["date"] <= pd.Timestamp(s_end))
        )
        sub = universe[mask]
        n_dates = sub["date"].nunique()
        mean_n = sub.groupby("date")["ticker"].count().mean() if n_dates > 0 else 0
        print(f"  {split_name:5s}: {n_dates:4d} dates  |  mean N_s = {mean_n:.1f} tickers/date")

    fwd_valid = returns["forward_return"].dropna()
    print(f"  Forward return  : min={fwd_valid.min():.4f}  max={fwd_valid.max():.4f}  "
          f"mean={fwd_valid.mean():.6f}  std={fwd_valid.std():.4f}")
    print("─────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
