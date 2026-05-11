"""
CA benchmark: Conditional Autoencoder.
Gu, Kelly & Xiu (2021) — CA1 variant.

Model:
  r_{i,t} = β_{i,t-1}^T f_t + u_{i,t}
  β_{i,t-1} = BetaNet(z_{i,t-1})   — non-linear loading function
  f_t        = FactorNet(x_t)       — linear map of managed portfolio

  x_t = lstsq(Z_{t-1}, r_t)        — managed portfolio (L cross-sectional regressions)

Training: Adam + MSE loss + L2 regularisation. Early stopping on val Rank IC.

Out-of-sample prediction (same as IPCA):
  μ_{i,t} = β_{i,t}^T λ̂
where λ̂ = mean in-sample factor estimates.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.fundamentals import (
    FundamentalsLoader,
    build_date_data,
    dates_in_range,
    load_returns_series,
    load_universe_by_date,
)
from factorvae.evaluation.metrics import compute_rank_ic


# ── Model architecture ────────────────────────────────────────────────────────

class BetaNet(nn.Module):
    """CA1 variant: one hidden layer with BatchNorm + ReLU."""

    def __init__(self, L: int, K: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(L, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, K),
        )

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """Z: (N, L) → beta: (N, K)."""
        return self.net(Z)


class FactorNet(nn.Module):
    """Linear map from managed portfolio to factors. No bias — economic constraint."""

    def __init__(self, L: int, K: int):
        super().__init__()
        self.linear = nn.Linear(L, K, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (L,) → f: (K,)."""
        return self.linear(x)


# ── Validation utility ────────────────────────────────────────────────────────

def _val_rank_ic(
    beta_net: BetaNet,
    lambda_hat: torch.Tensor,
    val_data: list[dict],
) -> float:
    """Mean Rank IC on validation set using frozen lambda_hat."""
    beta_net.eval()
    ics: list[float] = []
    with torch.no_grad():
        for d in val_data:
            Z_t = torch.tensor(d["Z"], dtype=torch.float32)
            r_t = torch.tensor(d["r"], dtype=torch.float32)
            mu = beta_net(Z_t) @ lambda_hat  # (N,)
            ics.append(compute_rank_ic(r_t, mu))
    return float(np.mean(ics)) if ics else float("-inf")


# ── Public API ────────────────────────────────────────────────────────────────

def train_and_predict(
    config: dict,
    K: int = 3,
    hidden: int = 32,
    exclude_tickers: list[str] | None = None,
) -> pd.DataFrame:
    """
    Train CA, predict on test split.

    Parameters
    ----------
    config          : parsed config.yaml dict
    K               : number of latent factors
    hidden          : hidden units in BetaNet (CA1 = 32)
    exclude_tickers : tickers to withhold from training

    Returns
    -------
    DataFrame with columns [date, ticker, mu_pred, sigma_pred, y_true].
    """
    dc = config["data"]
    tc = config["training"]
    exclude_set = set(exclude_tickers or [])

    seed         = tc["seed"]
    max_epochs   = tc["max_epochs"]
    lr           = tc["learning_rate"]
    weight_decay = float(tc.get("weight_decay", 1e-4))
    patience     = 10

    torch.manual_seed(seed)
    np.random.seed(seed)

    loader = FundamentalsLoader(Path(dc["processed_dir"]))
    L = loader.L

    print(f"  CA: K={K}, L={L}, hidden={hidden}, max_epochs={max_epochs}")

    ub_date = load_universe_by_date(config)
    ret_s   = load_returns_series(config)

    train_dates = dates_in_range(ub_date, dc["train_start"], dc["train_end"])
    val_dates   = dates_in_range(ub_date, dc["val_start"],   dc["val_end"])
    test_dates  = dates_in_range(ub_date, dc["test_start"],  dc["test_end"])

    print(f"  CA: loading {len(train_dates)} train + {len(val_dates)} val dates…")
    train_data = build_date_data(train_dates, ub_date, ret_s, loader, exclude_set)
    val_data   = build_date_data(val_dates,   ub_date, ret_s, loader)
    print(f"  CA: {len(train_data)} usable train / {len(val_data)} usable val dates")

    if not train_data:
        raise RuntimeError("CA: no usable training dates — check fundamentals data coverage.")

    # Pre-convert training data to tensors (avoids repeated allocation in the loop)
    train_tensors = [
        (
            torch.tensor(d["Z"], dtype=torch.float32),
            torch.tensor(d["r"], dtype=torch.float32),
            torch.tensor(d["x"], dtype=torch.float32),
        )
        for d in train_data
    ]

    # ── Model + optimiser ─────────────────────────────────────────────────────
    beta_net   = BetaNet(L, K, hidden)
    factor_net = FactorNet(L, K)
    optimizer  = torch.optim.Adam(
        list(beta_net.parameters()) + list(factor_net.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.MSELoss()

    best_val_ic    = float("-inf")
    best_beta_state   = copy.deepcopy(beta_net.state_dict())
    best_factor_state = copy.deepcopy(factor_net.state_dict())
    best_lambda: torch.Tensor | None = None
    epochs_no_imp  = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in tqdm(range(max_epochs), desc="  CA", unit="epoch", leave=True):
        beta_net.train()
        factor_net.train()
        total_loss = 0.0
        idx_order = np.random.permutation(len(train_tensors))

        for idx in idx_order:
            Z_t, r_t, x_t = train_tensors[idx]
            f_t    = factor_net(x_t)   # (K,)
            beta_t = beta_net(Z_t)     # (N, K)
            r_hat  = beta_t @ f_t      # (N,)
            optimizer.zero_grad()
            with autocast():
                loss   = criterion(r_hat, r_t)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Compute λ̂ from current factor_net (frozen)
        beta_net.eval()
        factor_net.eval()
        with torch.no_grad():
            lambda_hat = torch.stack(
                [factor_net(Z_t_x) for _, _, Z_t_x in train_tensors]
            ).mean(dim=0)  # (K,)

        val_ic  = _val_rank_ic(beta_net, lambda_hat, val_data)
        marker  = " *" if val_ic > best_val_ic else ""
        tqdm.write(
            f"    epoch {epoch + 1:3d}/{max_epochs}  "
            f"train_loss={total_loss / len(train_data):.4f}  "
            f"val_rank_ic={val_ic:+.4f}{marker}"
        )

        if val_ic > best_val_ic:
            best_val_ic       = val_ic
            best_beta_state   = copy.deepcopy(beta_net.state_dict())
            best_factor_state = copy.deepcopy(factor_net.state_dict())
            best_lambda       = lambda_hat.clone()
            epochs_no_imp     = 0
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= patience:
                tqdm.write(f"    Early stopping at epoch {epoch + 1} "
                          f"(no improvement for {patience} epochs)")
                break

    # Restore best checkpoint
    beta_net.load_state_dict(best_beta_state)
    if best_lambda is None:
        # Fallback: use last lambda_hat (happens if epoch 0 is the best)
        best_lambda = lambda_hat
    tqdm.write(f"  CA: best val Rank IC = {best_val_ic:+.4f}")

    # ── Test predictions ──────────────────────────────────────────────────────
    tqdm.write(f"  CA: predicting on {len(test_dates)} test dates…")
    beta_net.eval()
    records: list[dict] = []
    with torch.no_grad():
        for date in test_dates:
            tickers = ub_date.get(date, [])
            if not tickers:
                continue
            Z = loader.get(date, tickers)  # (N, L) — all tickers, no exclusion
            Z_t = torch.tensor(Z, dtype=torch.float32)
            mu = (beta_net(Z_t) @ best_lambda).numpy()  # (N,)

            for i, ticker in enumerate(tickers):
                try:
                    y_true: float = float(ret_s.loc[(date, ticker)])
                except KeyError:
                    y_true = float("nan")
                records.append({
                    "date":       date.strftime("%Y-%m-%d"),
                    "ticker":     ticker,
                    "mu_pred":    float(mu[i]),
                    "sigma_pred": 0.0,
                    "y_true":     y_true,
                })

    df = pd.DataFrame(records)
    tqdm.write(f"  CA: {len(df):,} prediction rows")
    return df
