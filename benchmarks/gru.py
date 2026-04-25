"""
GRU benchmark: GRU que prediz y diretamente a partir de (N, T, C).
Sem VAE, sem estrutura fatorial — isola o ganho do GRU puro.

Treinamento alinhado ao FactorVAE para comparação justa:
  - Mesmo train/val/test split do config
  - Early stopping em val Rank IC (patience=10), mesma métrica e critério do FactorVAE
  - Melhor checkpoint salvo em memória (não o estado final)
  - max_epochs, lr e seed lidos do config.yaml
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from factorvae.data.dataset import RealDataset
from factorvae.evaluation.metrics import compute_rank_ic

ROOT = Path(__file__).resolve().parents[1]


class SimpleGRU(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 20):
        super().__init__()
        self.gru  = nn.GRU(in_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, C)
        _, h_n = self.gru(x)                          # h_n: (1, N, H)
        return self.head(h_n.squeeze(0)).squeeze(-1)  # (N,)


def _val_rank_ic(model: SimpleGRU, val_ds: RealDataset) -> float:
    """Mean Rank IC over the val split — same metric FactorVAE optimises."""
    model.eval()
    ics: list[float] = []
    with torch.no_grad():
        for idx in range(len(val_ds)):
            sample = val_ds[idx]
            x, y = sample[0], sample[-2]
            mu = model(x)   # (N,)
            ics.append(compute_rank_ic(y, mu))
    return float(np.mean(ics)) if ics else float("-inf")


def train_and_predict(
    config: dict,
    hidden: int = 20,
) -> pd.DataFrame:
    """
    Train GRU with early stopping on val Rank IC; predict on test split.

    Hyperparameters (max_epochs, lr, seed) are read from config["training"]
    so they match the FactorVAE training run exactly.
    """
    tc = config["training"]
    dc = config["data"]

    seed       = tc["seed"]
    max_epochs = tc["max_epochs"]
    lr         = tc["learning_rate"]
    patience   = 10   # mirrors FactorVAE EarlyStopping(patience=10)

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = RealDataset(dc["processed_dir"], dc["train_start"], dc["train_end"],
                           dc["sequence_length"])
    val_ds   = RealDataset(dc["processed_dir"], dc["val_start"],   dc["val_end"],
                           dc["sequence_length"])
    test_ds  = RealDataset(dc["processed_dir"], dc["test_start"],  dc["test_end"],
                           dc["sequence_length"])

    C = train_ds.C
    model     = SimpleGRU(C, hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"  GRU: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test dates")

    best_val_ic   = float("-inf")
    best_state    = copy.deepcopy(model.state_dict())
    epochs_no_imp = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        indices    = np.random.permutation(len(train_ds))
        for idx in indices:
            sample = train_ds[idx]
            x, y   = sample[0], sample[-2]
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_ic   = _val_rank_ic(model, val_ds)
        improved = val_ic > best_val_ic
        marker   = " *" if improved else ""
        print(f"    epoch {epoch + 1:3d}/{max_epochs}  "
              f"train_loss={total_loss / len(train_ds):.4f}  "
              f"val_rank_ic={val_ic:+.4f}{marker}")

        if improved:
            best_val_ic = val_ic
            best_state  = copy.deepcopy(model.state_dict())
            epochs_no_imp = 0
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= patience:
                print(f"    Early stopping at epoch {epoch + 1} "
                      f"(no improvement for {patience} epochs)")
                break

    # Load best checkpoint (same as FactorVAE using ModelCheckpoint)
    model.load_state_dict(best_state)
    print(f"  GRU: best val Rank IC = {best_val_ic:+.4f}")

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
            sample  = test_ds[idx]
            x       = sample[0]
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
