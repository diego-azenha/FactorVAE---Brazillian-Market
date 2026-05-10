"""Debug script to investigate negative backtest returns."""

import pandas as pd
import yaml
from pathlib import Path

# Load config and check test period
with open('config.yaml') as f:
    config = yaml.safe_load(f)

dc = config['data']
print(f"Test period: {dc['test_start']} to {dc['test_end']}")

# Check market performance (Ibovespa)
ibov = pd.read_parquet('data/processed/ibov_returns.parquet')
ibov['date'] = pd.to_datetime(ibov['date'])

test_start = pd.Timestamp(dc['test_start'])
test_end = pd.Timestamp(dc['test_end'])
test_mask = (ibov['date'] >= test_start) & (ibov['date'] <= test_end)
test_ibov = ibov[test_mask]

print(f"\nIbovespa in test period:")
print(f"  Days: {len(test_ibov)}")
print(f"  Mean daily return: {test_ibov['return'].mean():+.6f}")
print(f"  Annualized: {(1 + test_ibov['return'].mean())**252 - 1:+.2%}")
print(f"  Cumulative: {(1 + test_ibov['return']).prod() - 1:+.2%}")
print(f"  Min day: {test_ibov['return'].min():+.4f}")
print(f"  Max day: {test_ibov['return'].max():+.4f}")

# Check actual returns from the universe
if Path('data/processed/returns.parquet').exists():
    rets = pd.read_parquet('data/processed/returns.parquet')
    rets['date'] = pd.to_datetime(rets['date'])
    test_rets = rets[
        (rets['date'] >= test_start) & 
        (rets['date'] <= test_end)
    ]
    print(f"\nStock returns in test period:")
    print(f"  Rows: {len(test_rets)}")
    print(f"  Mean forward return: {test_rets['forward_return'].mean():+.6f}")
    print(f"  Median forward return: {test_rets['forward_return'].median():+.6f}")
    print(f"  Std: {test_rets['forward_return'].std():.6f}")
    print(f"  Min: {test_rets['forward_return'].min():+.4f}")
    print(f"  Max: {test_rets['forward_return'].max():+.4f}")
    
    # Sample 1 date
    sample_date = test_rets['date'].unique()[100]
    sample_day = test_rets[test_rets['date'] == sample_date]
    print(f"\nSample day {sample_date.date()}: {len(sample_day)} stocks")
    print(f"  Mean return: {sample_day['forward_return'].mean():+.6f}")
    print(f"  Mean abs return: {sample_day['forward_return'].abs().mean():+.6f}")

print(f"\n{'='*60}")
print("Analysis: Why are returns so negative?")
print(f"{'='*60}")

if test_ibov['return'].mean() < -0.0001:
    print("\n⚠️  BEAR MARKET: Index itself is down significantly")
    print(f"   Ibovespa cumulative return: {(1 + test_ibov['return']).prod() - 1:+.2%}")
    print("   Any long-only stock strategy will struggle.")
else:
    print("\n✓ Market is positive, issue is with signal/strategy")
