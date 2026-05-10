"""Detailed backtest debugging with daily breakdown."""

import pandas as pd
import yaml
import numpy as np
from pathlib import Path

# Load
pred_path = Path('results/predictions/predictions.parquet')
preds = pd.read_parquet(pred_path)
preds['date'] = pd.to_datetime(preds['date'])

with open('config.yaml') as f:
    config = yaml.safe_load(f)

dc = config['data']
test_start = pd.Timestamp(dc['test_start'])
test_end = pd.Timestamp(dc['test_end'])

test_mask = (preds['date'] >= test_start) & (preds['date'] <= test_end)
test_preds = preds[test_mask].copy()

# Manual TopK-Drop with detailed logging
k = config['evaluation']['top_k']
n = config['evaluation']['drop_n']
fee_rate = 0.005

dates = sorted(test_preds['date'].unique())
current_portfolio = set()
daily_records = []

for idx, date in enumerate(dates[:10]):  # First 10 days
    day_df = test_preds[test_preds['date'] == date].copy()
    day_df = day_df.sort_values('mu_pred', ascending=False)
    
    top_candidates = set(day_df.head(k + n)['ticker'])
    retained = current_portfolio & top_candidates
    needed = k - len(retained)
    candidates_sorted = day_df[~day_df['ticker'].isin(retained)]['ticker'].tolist()
    new_stocks = set(candidates_sorted[:needed])
    new_portfolio = retained | new_stocks
    
    turnover = len(new_portfolio - current_portfolio) / k if current_portfolio else 1.0
    
    held = day_df[day_df['ticker'].isin(new_portfolio)]
    valid_returns = held['y_true'].dropna()
    
    gross_return = valid_returns.mean() if len(valid_returns) > 0 else 0.0
    net_return = gross_return - fee_rate * turnover
    fee = fee_rate * turnover
    
    daily_records.append({
        'date': date,
        'portfolio_size': len(new_portfolio),
        'held_with_returns': len(valid_returns),
        'turnover': turnover,
        'fee': fee,
        'gross_return': gross_return,
        'net_return': net_return,
    })
    
    print(f"{date.date()} | Port={len(new_portfolio):2d} | Held={len(valid_returns):3d} | TO={turnover:.2%} | Fee={fee:.4f} | Gross={gross_return:+.4f} | Net={net_return:+.4f}")
    
    current_portfolio = new_portfolio

daily_df = pd.DataFrame(daily_records)
print(f"\n{'='*80}")
print(f"Summary (first 10 days):")
print(f"  Mean turnover: {daily_df['turnover'].mean():.2%}")
print(f"  Mean fee: {daily_df['fee'].mean():.4f} per day = {daily_df['fee'].mean() * 252:.2%} annualized")
print(f"  Mean gross return: {daily_df['gross_return'].mean():+.4f}")
print(f"  Mean net return: {daily_df['net_return'].mean():+.4f}")
print(f"  Cumulative gross: {(1 + daily_df['gross_return']).prod() - 1:+.2%}")
print(f"  Cumulative net: {(1 + daily_df['net_return']).prod() - 1:+.2%}")

# Check for NaN issues
print(f"\n{'='*80}")
print(f"Data quality check:")
print(f"  % of days with valid returns (not all NaN): {(daily_df['held_with_returns'] > 0).sum() / len(daily_df):.1%}")
print(f"  % of days with empty portfolio: {(daily_df['portfolio_size'] == 0).sum() / len(daily_df):.1%}")
