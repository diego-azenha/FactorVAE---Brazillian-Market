"""
Apply portfolio turnover constraint to benchmark predictions.

The constraint enforces:
    |P_t ∩ P_{t-1}| >= k - n
    
I.e., at most n stocks can be added/removed per day.

This matches the constraint applied in FactorVAE backtest, ensuring fair comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_turnover_constraint(
    predictions: pd.DataFrame,
    k: int = 50,
    n: int = 5,
) -> pd.DataFrame:
    """
    Apply turnover constraint to daily predictions.
    
    For each date, select the top k stocks such that |P_t ∩ P_{t-1}| >= k - n.
    Predictions for stocks outside the constrained portfolio are set to NaN.
    
    Args:
        predictions : DataFrame with columns [date, ticker, mu_pred, y_true]
                      (or mu_pred may be named differently depending on model)
        k           : Portfolio size (target number of stocks per day)
        n           : Max turnover per day (max k stocks added/removed)
    
    Returns:
        DataFrame with same structure, but predictions set to NaN for
        stocks outside the constrained portfolio.
    """
    if predictions.empty:
        return predictions
    
    df = predictions.copy()
    
    # Ensure date column is datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        date_col = "date"
    else:
        raise ValueError("predictions must contain 'date' column")
    
    # Find prediction score column (mu_pred, score, etc.)
    score_col = None
    for col in ["mu_pred", "score", "prediction", "pred"]:
        if col in df.columns:
            score_col = col
            break
    if score_col is None:
        raise ValueError(f"Could not find prediction score column. Available: {df.columns.tolist()}")
    
    # Sort by date, then by score (descending) within each date
    df = df.sort_values([date_col, score_col], ascending=[True, False]).reset_index(drop=True)
    
    portfolio: set[str] = set()
    mask = np.zeros(len(df), dtype=bool)
    
    for date_i, (date_val, group) in enumerate(df.groupby(date_col, sort=False)):
        group_idx = group.index
        tickers = group["ticker"].values
        
        # Top k + n candidates (more than necessary to account for overlap)
        candidates = set(tickers[:k + n])
        
        # Retained stocks: intersection of current portfolio with candidates
        retained = portfolio & candidates
        
        # New slots needed to reach k
        needed = k - len(retained)
        
        # Add top new stocks to fill slots
        new_stocks: set[str] = set()
        for ticker in tickers:
            if len(new_stocks) >= needed:
                break
            if ticker not in retained:
                new_stocks.add(ticker)
        
        # Updated portfolio for next day
        portfolio = retained | new_stocks
        
        # Mark this day's portfolio in the mask
        for idx, ticker in zip(group_idx, tickers):
            if ticker in portfolio:
                mask[idx] = True
    
    # Set predictions to NaN for tickers outside the constrained portfolio
    df.loc[~mask, score_col] = np.nan
    
    return df
