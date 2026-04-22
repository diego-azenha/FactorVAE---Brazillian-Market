"""
Momentum benchmark: mu_pred = ret_20d (a feature already in the dataset).

No training, no parameters. Reads the test split from RealDataset, extracts
the z-scored ret_20d feature value at the last timestep, and uses it as the
predicted return signal.

The schema of the output parquet matches results/predictions/predictions.parquet
so that all downstream backtest and comparison code can consume every model
prediction source uniformly.

Usage (via run_benchmarks.py):
    python benchmarks/run_benchmarks.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from factorvae.data.dataset import RealDataset

ROOT = Path(__file__).resolve().parents[1]


def generate_predictions(config: dict) -> pd.DataFrame:
    """
    Generate momentum predictions for the test split.

    Args:
        config: parsed config.yaml dict

    Returns:
        DataFrame with columns [date, ticker, mu_pred, sigma_pred, y_true]
    """
    dc = config["data"]

    dataset = RealDataset(
        processed_dir=dc["processed_dir"],
        start_date=dc["test_start"],
        end_date=dc["test_end"],
        sequence_length=dc["sequence_length"],
    )

    if "ret_20d" not in dataset.feature_cols:
        raise ValueError(
            f"'ret_20d' not found in dataset.feature_cols: {dataset.feature_cols}. "
            "Momentum benchmark requires this feature to be present."
        )
    ret_20d_idx = dataset.feature_cols.index("ret_20d")

    # Raw forward returns for y_true (economic units, not z-scored)
    raw_returns: pd.Series = (
        pd.read_parquet(Path(dc["processed_dir"]) / "returns.parquet")
        .assign(date=lambda df: pd.to_datetime(df["date"]))
        .set_index(["date", "ticker"])["forward_return"]
    )

    records = []
    for idx in range(len(dataset)):
        date_ts = dataset.trading_dates[idx]
        tickers = dataset.universe_by_date[date_ts]
        x, _, _ = dataset[idx]                         # (N, T, C)
        ret_20d_last = x[:, -1, ret_20d_idx].numpy()   # (N,) — z-scored last timestep

        for i, ticker in enumerate(tickers):
            try:
                y_true = float(raw_returns.loc[(date_ts, ticker)])
            except KeyError:
                y_true = float("nan")

            records.append({
                "date":       date_ts.strftime("%Y-%m-%d"),
                "ticker":     ticker,
                "mu_pred":    float(ret_20d_last[i]),
                "sigma_pred": 0.0,
                "y_true":     y_true,
            })

    return pd.DataFrame(records)
