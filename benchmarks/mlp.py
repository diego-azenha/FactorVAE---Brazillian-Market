"""
MLP benchmark: rede feedforward simples sobre features do último timestep.

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
from torch.utils.data import DataLoader, TensorDataset

from factorvae.data.dataset import RealDataset
from factorvae.evaluation.metrics import compute_rank_ic

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


def _stack_last_timestep(
    dataset: RealDataset,
) -> tuple[np.ndarray, np.ndarray, list, list]:
    """Flatten (N, T, C) dataset → (N_total, C) using only the last timestep."""
    X_all, y_all, date_labels, ticker_labels = [], [], [], []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        x, y = sample[0], sample[-2]  # robust to 3-tuple or 4-tuple: y is always second-to-last
        date_ts = dataset.trading_dates[idx]
        tickers = dataset.universe_by_date[date_ts]
        X_all.append(x[:, -1, :].numpy())
        y_all.append(y.numpy())
        date_labels.extend([date_ts] * len(tickers))
        ticker_labels.extend(tickers)
    return (
        np.concatenate(X_all),
        np.concatenate(y_all),
        date_labels,
        ticker_labels,
    )


def _val_rank_ic(model: SimpleMLP, val_ds: RealDataset) -> float:
    """Mean Rank IC over the val split — same metric FactorVAE optimises."""
    model.eval()
    ics: list[float] = []
    with torch.no_grad():
        for idx in range(len(val_ds)):
            sample = val_ds[idx]
            x, y = sample[0], sample[-2]
            mu = model(x[:, -1, :])   # (N,)
            ics.append(compute_rank_ic(y, mu))
    return float(np.mean(ics)) if ics else float("-inf")


def train_and_predict(
    config: dict,
    hidden: int = 64,
    batch_size: int = 256,
) -> pd.DataFrame:
    """
    Train MLP with early stopping on val Rank IC; predict on test split.

    Hyperparameters (max_epochs, lr, seed) are read from config["training"]
    so they match the FactorVAE training run exactly.
    """
    tc = config["training"]
    dc = config["data"]

    seed      = tc["seed"]
    max_epochs = tc["max_epochs"]
    lr        = tc["learning_rate"]
    patience  = 10   # mirrors FactorVAE EarlyStopping(patience=10)

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = RealDataset(dc["processed_dir"], dc["train_start"], dc["train_end"],
                           dc["sequence_length"])
    val_ds   = RealDataset(dc["processed_dir"], dc["val_start"],   dc["val_end"],
                           dc["sequence_length"])
    test_ds  = RealDataset(dc["processed_dir"], dc["test_start"],  dc["test_end"],
                           dc["sequence_length"])

    print(f"  MLP: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test dates")

    X_tr, y_tr, _, _ = _stack_last_timestep(train_ds)
    C = X_tr.shape[1]

    model     = SimpleMLP(C, hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True
    )

    best_val_ic   = float("-inf")
    best_state    = copy.deepcopy(model.state_dict())
    epochs_no_imp = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        val_ic = _val_rank_ic(model, val_ds)
        improved = val_ic > best_val_ic
        marker = " *" if improved else ""
        print(f"    epoch {epoch + 1:3d}/{max_epochs}  val_rank_ic={val_ic:+.4f}{marker}")

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
    print(f"  MLP: best val Rank IC = {best_val_ic:+.4f}")

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
