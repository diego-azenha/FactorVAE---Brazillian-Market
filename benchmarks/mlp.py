"""
MLP benchmark: rede feedforward simples sobre features do último timestep.

A cada data, pega x[:, -1, :] (N, C), passa por MLP de 2 camadas, prediz y.
Treinamento: concatena todas as datas de treino, minibatch padrão.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from factorvae.data.dataset import RealDataset

ROOT = Path(__file__).resolve().parents[1]


class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _stack_last_timestep(dataset: RealDataset):
    X_all, y_all, date_labels, ticker_labels = [], [], [], []
    for idx in range(len(dataset)):
        x, y, _ = dataset[idx]
        date_ts = dataset.trading_dates[idx]
        tickers = dataset.universe_by_date[date_ts]
        X_all.append(x[:, -1, :].numpy())
        y_all.append(y.numpy())
        date_labels.extend([date_ts] * len(tickers))
        ticker_labels.extend(tickers)
    return (np.concatenate(X_all), np.concatenate(y_all),
            date_labels, ticker_labels)


def train_and_predict(config: dict,
                      hidden: int = 64,
                      epochs: int = 20,
                      lr: float = 1e-3,
                      batch_size: int = 256,
                      seed: int = 42) -> pd.DataFrame:
    torch.manual_seed(seed)
    dc = config["data"]

    train_ds = RealDataset(dc["processed_dir"], dc["train_start"], dc["train_end"],
                           dc["sequence_length"])
    test_ds  = RealDataset(dc["processed_dir"], dc["test_start"],  dc["test_end"],
                           dc["sequence_length"])

    print(f"  MLP: treinando em {len(train_ds)} datas…")
    X_tr, y_tr, _, _ = _stack_last_timestep(train_ds)
    C = X_tr.shape[1]

    model = SimpleMLP(C, hidden=hidden)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                        batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optim.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optim.step()

    print(f"  MLP: prevendo em {len(test_ds)} datas…")
    X_te, _, dates, tickers = _stack_last_timestep(test_ds)
    model.eval()
    with torch.no_grad():
        mu_pred = model(torch.tensor(X_te, dtype=torch.float32)).numpy()

    raw_returns = (
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
