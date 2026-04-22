"""
Linear benchmark: Ridge regression on last-timestep features.

For each training date, flatten x[:, -1, :] into (N_s, C) and y into (N_s,),
concatenate across all train dates, fit Ridge(alpha=1.0). At test time,
apply the same linear model per date.

Using only the last timestep is the standard convention for linear factor models
(no temporal structure assumed), which makes this a fair linear ablation: if
FactorVAE barely beats this, the GRU + VAE machinery is not extracting nonlinear
or temporal structure.

Usage (via run_benchmarks.py):
    python benchmarks/run_benchmarks.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from factorvae.data.dataset import RealDataset

ROOT = Path(__file__).resolve().parents[1]


def _stack_dataset(
    dataset: RealDataset,
) -> tuple[np.ndarray, np.ndarray, list, list]:
    """Flatten a RealDataset into (X, y, dates, tickers) arrays."""
    X_all:         list[np.ndarray] = []
    y_all:         list[np.ndarray] = []
    date_labels:   list             = []
    ticker_labels: list[str]        = []

    for idx in range(len(dataset)):
        x, y, _ = dataset[idx]                         # (N, T, C), (N,)
        date_ts  = dataset.trading_dates[idx]
        tickers  = dataset.universe_by_date[date_ts]

        X_last = x[:, -1, :].numpy()                   # (N, C) last-timestep features
        X_all.append(X_last)
        y_all.append(y.numpy())
        date_labels.extend([date_ts] * len(tickers))
        ticker_labels.extend(tickers)

    return (
        np.concatenate(X_all, axis=0),
        np.concatenate(y_all, axis=0),
        date_labels,
        ticker_labels,
    )


def train_and_predict(config: dict, alpha: float = 1.0) -> pd.DataFrame:
    """
    Train a Ridge model on the train split, predict on the test split.

    Args:
        config: parsed config.yaml dict
        alpha:  Ridge regularisation strength (default 1.0)

    Returns:
        DataFrame with columns [date, ticker, mu_pred, sigma_pred, y_true]
    """
    dc = config["data"]

    train_ds = RealDataset(
        dc["processed_dir"], dc["train_start"], dc["train_end"], dc["sequence_length"]
    )
    test_ds = RealDataset(
        dc["processed_dir"], dc["test_start"], dc["test_end"], dc["sequence_length"]
    )

    print(f"  Ridge: fitting on {len(train_ds)} train dates…")
    X_train, y_train, _, _ = _stack_dataset(train_ds)
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    print(f"  Ridge: predicting on {len(test_ds)} test dates…")
    X_test, _, dates, tickers = _stack_dataset(test_ds)
    mu_pred: np.ndarray = model.predict(X_test)

    raw_returns: pd.Series = (
        pd.read_parquet(Path(dc["processed_dir"]) / "returns.parquet")
        .assign(date=lambda df: pd.to_datetime(df["date"]))
        .set_index(["date", "ticker"])["forward_return"]
    )

    records = []
    for date_ts, ticker, mu in zip(dates, tickers, mu_pred):
        try:
            y_true = float(raw_returns.loc[(date_ts, ticker)])
        except KeyError:
            y_true = float("nan")

        records.append({
            "date":       date_ts.strftime("%Y-%m-%d"),
            "ticker":     ticker,
            "mu_pred":    float(mu),
            "sigma_pred": 0.0,
            "y_true":     y_true,
        })

    return pd.DataFrame(records)
