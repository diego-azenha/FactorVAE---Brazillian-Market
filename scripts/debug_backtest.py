"""Debug TopK-Drop strategy execution."""

import pandas as pd
import yaml
import numpy as np
from pathlib import Path
from factorvae.evaluation.backtest import topk_drop_strategy

# Load predictions
pred_path = Path('results/predictions/predictions.parquet')
if not pred_path.exists():
    print(f"ERROR: {pred_path} not found")
    print("Run: python scripts/evaluate.py")
    exit(1)

preds = pd.read_parquet(pred_path)
preds['date'] = pd.to_datetime(preds['date'])

# Filter to test period only
with open('config.yaml') as f:
    config = yaml.safe_load(f)

dc = config['data']
test_start = pd.Timestamp(dc['test_start'])
test_end = pd.Timestamp(dc['test_end'])

test_mask = (preds['date'] >= test_start) & (preds['date'] <= test_end)
test_preds = preds[test_mask].copy()

print(f"Test predictions: {len(test_preds)} rows")
print(f"  Dates: {test_preds['date'].min().date()} to {test_preds['date'].max().date()}")
print(f"  Tickers: {test_preds['ticker'].nunique()}")
print(f"  Dates: {test_preds['date'].nunique()}")

# Analyze predictions
print(f"\nPrediction statistics:")
print(f"  mu_pred mean: {test_preds['mu_pred'].mean():+.6f}")
print(f"  mu_pred std:  {test_preds['mu_pred'].std():.6f}")
print(f"  mu_pred range: [{test_preds['mu_pred'].min():.4f}, {test_preds['mu_pred'].max():.4f}]")
print(f"  mu_pred NaN: {test_preds['mu_pred'].isna().sum()}")

print(f"\nActual returns (y_true):")
print(f"  y_true mean: {test_preds['y_true'].mean():+.6f}")
print(f"  y_true NaN: {test_preds['y_true'].isna().sum()}")

# Sample date: check signal-return correlation
sample_dates = sorted(test_preds['date'].unique())[100:105]
for date in sample_dates:
    day = test_preds[test_preds['date'] == date].dropna(subset=['y_true'])
    if len(day) > 5:
        corr = day['mu_pred'].corr(day['y_true'])
        print(f"\n{date.date()}: {len(day)} stocks, correlation: {corr:+.4f}")
        print(f"  Top 3 by mu_pred: {day.nlargest(3, 'mu_pred')['y_true'].mean():+.6f}")
        print(f"  Bottom 3 by mu_pred: {day.nsmallest(3, 'mu_pred')['y_true'].mean():+.6f}")

# Run backtest
k = config['evaluation']['top_k']
n = config['evaluation']['drop_n']
fee = 0.005

print(f"\n{'='*60}")
print(f"Running TopK-Drop backtest (k={k}, n={n}, fee={fee:.1%})")
print(f"{'='*60}\n")

backtest_df = topk_drop_strategy(test_preds, k=k, n=n, eta=0.0, fee_rate=fee)

print(f"Backtest results:")
print(f"  Trading days: {len(backtest_df)}")
print(f"  Mean daily return: {backtest_df['portfolio_return'].mean():+.6f}")
print(f"  Annualized: {(1 + backtest_df['portfolio_return'].mean())**252 - 1:+.2%}")
print(f"  Cumulative: {(1 + backtest_df['portfolio_return']).prod() - 1:+.2%}")
print(f"  Sharpe (assuming rf=0): {backtest_df['portfolio_return'].mean() / backtest_df['portfolio_return'].std() * np.sqrt(252):.2f}")

print(f"\nTurnover:")
print(f"  Mean: {backtest_df['turnover'].mean():.2%}")
print(f"  Min: {backtest_df['turnover'].min():.2%}")
print(f"  Max: {backtest_df['turnover'].max():.2%}")

# Estimate fees
estimated_fees = (backtest_df['turnover'] * fee).sum()
print(f"\nEstimated total fees: {estimated_fees:+.2%}")
print(f"Fees per day: {(backtest_df['turnover'] * fee).mean():+.4f} (~{(backtest_df['turnover'] * fee).mean() * 252 * 100:.1f} bps/year)")

# WITHOUT fees
backtest_gross = backtest_df.copy()
backtest_gross['portfolio_return'] = backtest_gross['portfolio_return'] + (backtest_df['turnover'] * fee)
print(f"\nGross return (before fees): {(1 + backtest_gross['portfolio_return']).prod() - 1:+.2%}")
