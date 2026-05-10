"""
Download Ibovespa index returns from Yahoo Finance.

Saves daily returns to data/processed/ibov_returns.parquet for use as
the benchmark in backtest.py.

Usage:
    python scripts/download_ibov.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Install with:")
    print("  pip install yfinance")
    exit(1)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Download Ibovespa index and save returns to parquet."""
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ibov_returns.parquet"

    print("Downloading Ibovespa (^BVSP) index from Yahoo Finance...")
    print("  Start date: 2007-01-01")
    print("  End date: today")

    try:
        ibov = yf.download("^BVSP", start="2007-01-01", progress=False)
    except Exception as e:
        print(f"ERROR downloading data: {e}")
        exit(1)

    if ibov.empty:
        print("ERROR: No data returned from Yahoo Finance")
        exit(1)

    # Handle MultiIndex columns (ticker level)
    if isinstance(ibov.columns, pd.MultiIndex):
        ibov.columns = ibov.columns.get_level_values(0)

    # Calculate daily returns
    ibov["return"] = ibov["Close"].pct_change() if "Close" in ibov.columns else ibov["Adj Close"].pct_change()

    # Keep only date and return columns
    ibov_returns = ibov[["return"]].copy()
    ibov_returns.index.name = "date"
    ibov_returns = ibov_returns.reset_index()

    # Save to parquet
    ibov_returns.to_parquet(out_path, index=False)

    print(f"\n✓ Saved {len(ibov_returns):,} daily observations to {out_path.relative_to(ROOT)}")
    print(f"  Date range: {ibov_returns['date'].min().date()} to {ibov_returns['date'].max().date()}")
    print(f"  Mean daily return: {ibov_returns['return'].mean():+.6f}")
    print(f"  Annualized return: {(1 + ibov_returns['return'].mean())**252 - 1:+.2%}")


if __name__ == "__main__":
    main()
