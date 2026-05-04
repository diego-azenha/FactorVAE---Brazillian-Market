"""
Day 0 diagnostic — Universe survivorship and look-ahead audit.

Checks:
  1. Survivorship bias: are tickers that delisted early 2010-2018 present in
     prices_wide.csv? If the CSV only contains currently-listed stocks,
     all training/test cross-sections are biased.
  2. Ticker leakage across splits: do any tickers appear exclusively in
     test but not in train? (Indicates future-added data)
  3. Cross-section size over time: sharp drops/jumps indicate data cuts.
  4. Forward-return look-ahead: verify that forward_return on date t is computed
     from prices at t+1 and t+2, NOT from future data.

Usage:
    python scripts/diagnose_universe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)

    processed_dir = ROOT / config["data"]["processed_dir"]
    prices_path   = ROOT / "data" / "prices_wide.csv"

    print("\n" + "=" * 65)
    print("UNIVERSE SURVIVORSHIP & LOOK-AHEAD AUDIT")
    print("=" * 65)

    # ── 1. Raw prices coverage ────────────────────────────────────────────
    print("\n[1] Raw prices_wide.csv coverage")
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    prices.index = pd.to_datetime(prices.index)
    tickers_all = prices.columns.tolist()

    print(f"  Date range   : {prices.index.min().date()} -> {prices.index.max().date()}")
    print(f"  Total tickers: {len(tickers_all)}")

    # First and last valid price per ticker
    first_date = prices.apply(lambda col: col.first_valid_index())
    last_date  = prices.apply(lambda col: col.last_valid_index())

    early_end = pd.Timestamp(config["data"]["train_end"])
    delisted_before_train_end = (last_date <= early_end).sum()
    never_in_train = (first_date >= pd.Timestamp(config["data"]["test_start"])).sum()

    print(f"  Tickers with last valid price <= train_end ({config['data']['train_end']}): "
          f"{delisted_before_train_end}  (should be > 0 if no survivorship bias)")
    print(f"  Tickers first appearing in test period (>= {config['data']['test_start']}): "
          f"{never_in_train}  (look-ahead risk if > 0)")

    # Distribution of last valid date
    last_year = last_date.dt.year.value_counts().sort_index()
    print("\n  Last valid price year distribution (should have entries every year):")
    for yr, cnt in last_year.items():
        bar = "#" * (cnt // 2)
        print(f"    {yr}  {cnt:4d}  {bar}")

    # ── 2. Universe parquet ───────────────────────────────────────────────
    print("\n[2] Processed universe.parquet")
    universe = pd.read_parquet(processed_dir / "universe.parquet")
    universe["date"] = pd.to_datetime(universe["date"])

    # Cross-section size over time
    cs_size = universe.groupby("date")["ticker"].count()

    train_start = pd.Timestamp(config["data"]["train_start"])
    train_end   = pd.Timestamp(config["data"]["train_end"])
    val_start   = pd.Timestamp(config["data"]["val_start"])
    val_end     = pd.Timestamp(config["data"]["val_end"])
    test_start  = pd.Timestamp(config["data"]["test_start"])
    test_end    = pd.Timestamp(config["data"]["test_end"])

    cs_train = cs_size[(cs_size.index >= train_start) & (cs_size.index <= train_end)]
    cs_val   = cs_size[(cs_size.index >= val_start)   & (cs_size.index <= val_end)]
    cs_test  = cs_size[(cs_size.index >= test_start)  & (cs_size.index <= test_end)]

    def _stats(s: pd.Series, label: str):
        print(f"  {label:<12}: n_dates={len(s):4d}  "
              f"cs_mean={s.mean():.0f}  cs_min={s.min():.0f}  cs_max={s.max():.0f}")

    _stats(cs_train, "Train")
    _stats(cs_val,   "Val (2018)")
    _stats(cs_test,  "Test")

    # ── 3. Ticker overlap across splits ───────────────────────────────────
    print("\n[3] Ticker overlap across splits")
    tickers_train = set(universe.loc[universe["date"].between(train_start, train_end), "ticker"])
    tickers_val   = set(universe.loc[universe["date"].between(val_start,   val_end),   "ticker"])
    tickers_test  = set(universe.loc[universe["date"].between(test_start,  test_end),  "ticker"])

    only_test  = tickers_test - tickers_train - tickers_val
    only_train = tickers_train - tickers_test

    print(f"  Train-only tickers (delisted before test): {len(only_train)}")
    print(f"  Test-only  tickers (absent from train):    {len(only_test)}")
    if only_test:
        print(f"    Examples: {sorted(only_test)[:10]}")

    # ── 4. Forward-return look-ahead check ────────────────────────────────
    print("\n[4] Forward-return construction check")
    returns = pd.read_parquet(processed_dir / "returns.parquet")
    returns["date"] = pd.to_datetime(returns["date"])

    # Sample: for a given ticker, verify that return on date t = (p[t+2]-p[t+1])/p[t+1]
    sample_ticker = returns["ticker"].value_counts().index[0]
    sample_returns = returns[returns["ticker"] == sample_ticker].set_index("date")["forward_return"].sort_index()
    sample_prices  = prices[sample_ticker].dropna()

    mismatches = 0
    checks = 0
    for date in sample_returns.index[:50]:
        try:
            iloc = sample_prices.index.get_loc(date)
            if iloc + 2 >= len(sample_prices):
                continue
            p_t1 = sample_prices.iloc[iloc + 1]
            p_t2 = sample_prices.iloc[iloc + 2]
            expected = (p_t2 - p_t1) / p_t1
            actual   = sample_returns.loc[date]
            if abs(expected - actual) > 1e-6:
                mismatches += 1
            checks += 1
        except Exception:
            pass

    if mismatches == 0:
        print(f"  [OK] Forward returns match prices[t+1->t+2] formula  ({checks} checks on '{sample_ticker}')")
    else:
        print(f"  [FAIL] {mismatches}/{checks} mismatches! Look-ahead or formula error detected.")

    # ── 5. Survivorship verdict ───────────────────────────────────────────
    print("\n[5] Survivorship verdict")
    if delisted_before_train_end == 0:
        print("  [X] SURVIVORSHIP BIAS LIKELY: no tickers with last price <= train_end.")
        print("    prices_wide.csv may contain only currently-listed stocks.")
        print("    All returns are upward-biased; benchmark comparison is invalid.")
    elif delisted_before_train_end < 10:
        print(f"  [~] PARTIAL BIAS: only {delisted_before_train_end} delisted tickers found. "
              "Likely undercounts historical churn.")
    else:
        print(f"  [OK] {delisted_before_train_end} tickers delisted before train_end. "
              "Survivorship bias appears manageable.")

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir = ROOT / "results" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    cs_size.to_csv(out_dir / "cross_section_size.csv", header=True)
    pd.DataFrame({
        "ticker": list(only_test),
        "note": "test-only (absent from train)"
    }).to_csv(out_dir / "test_only_tickers.csv", index=False)

    print(f"\nArtifacts saved to: results/diagnostics/")


if __name__ == "__main__":
    main()
