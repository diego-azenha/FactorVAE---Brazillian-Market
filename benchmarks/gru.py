"""
GRU benchmark: GRU que prediz y diretamente a partir de (N, T, C).
Sem VAE, sem estrutura fatorial — isola o ganho do GRU puro.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from factorvae.data.dataset import RealDataset

ROOT = Path(__file__).resolve().parents[1]


class SimpleGRU(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 20):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (N, T, C)
        _, h_n = self.gru(x)          # h_n: (1, N, H)
        return self.head(h_n.squeeze(0)).squeeze(-1)  # (N,)


def train_and_predict(config: dict,
                      hidden: int = 20,
                      epochs: int = 15,
                      lr: float = 1e-3,
                      seed: int = 42) -> pd.DataFrame:
    torch.manual_seed(seed)
    dc = config["data"]

    train_ds = RealDataset(dc["processed_dir"], dc["train_start"], dc["train_end"],
                           dc["sequence_length"])
    test_ds  = RealDataset(dc["processed_dir"], dc["test_start"],  dc["test_end"],
                           dc["sequence_length"])

    C = train_ds.C
    model = SimpleGRU(C, hidden=hidden)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"  GRU: treinando em {len(train_ds)} datas × {epochs} épocas…")
    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(len(train_ds))
        total_loss = 0.0
        for idx in indices:
            x, y, _ = train_ds[idx]
            optim.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        print(f"    epoch {epoch + 1}: loss = {total_loss / len(train_ds):.4f}")

    print(f"  GRU: prevendo em {len(test_ds)} datas…")
    model.eval()
    raw_returns = (
        pd.read_parquet(Path(dc["processed_dir"]) / "returns.parquet")
        .assign(date=lambda df: pd.to_datetime(df["date"]))
        .set_index(["date", "ticker"])["forward_return"]
    )

    records = []
    with torch.no_grad():
        for idx in range(len(test_ds)):
            x, _, _ = test_ds[idx]
            mu_pred = model(x).numpy()
            date_ts = test_ds.trading_dates[idx]
            tickers = test_ds.universe_by_date[date_ts]
            for i, ticker in enumerate(tickers):
                try:
                    y_true = float(raw_returns.loc[(date_ts, ticker)])
                except KeyError:
                    y_true = float("nan")
                records.append({
                    "date":       date_ts.strftime("%Y-%m-%d"),
                    "ticker":     ticker,
                    "mu_pred":    float(mu_pred[i]),
                    "sigma_pred": 0.0,
                    "y_true":     y_true,
                })
    return pd.DataFrame(records)
