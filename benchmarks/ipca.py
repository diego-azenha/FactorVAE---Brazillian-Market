"""
IPCA benchmark: Instrumented Principal Components Analysis.
Kelly, Pruitt & Su (2019).

Model: r_{i,t} = z_{i,t-1}^T Γ_β f_t + ε_{i,t}
  Γ_β ∈ R^{L×K}  — global loading map (same for all assets and dates)
  f_t ∈ R^K      — latent factor (dynamic, estimated per date)
  z_{i,t}        — L fundamentalist characteristics (cross-sectionally z-scored)

Estimation: Alternating Least Squares (ALS).

Out-of-sample prediction:
  μ_{i,t} = z_{i,t}^T Γ̂_β λ̂
where λ̂ = mean of in-sample factor estimates (risk premium proxy).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
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


# ── ALS fitting ───────────────────────────────────────────────────────────────

def _als_fit(
    train_data: list[dict],
    K: int,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit IPCA via ALS.

    Returns
    -------
    Gamma     : (L, K)  loading map
    lambda_hat: (K,)    mean factor (risk premium estimate)
    """
    L = train_data[0]["Z"].shape[1]

    # ── Initialization from managed-portfolio covariance ─────────────────────
    X = np.stack([d["x"] for d in train_data]).astype(np.float64)  # (T, L)
    cov = X.T @ X  # (L, L)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    Q, _ = np.linalg.qr(eigvecs[:, order[:K]])
    Gamma = Q[:, :K].astype(np.float64)  # (L, K), orthonormal columns

    factors: list[np.ndarray] = [np.zeros(K)] * len(train_data)

    for iteration in tqdm(range(max_iter), desc="  IPCA ALS", unit="iter", leave=False):
        # ── Factor update (closed-form per date) ─────────────────────────────
        new_factors: list[np.ndarray] = []
        for d in train_data:
            Z = d["Z"].astype(np.float64)  # (N, L)
            r = d["r"].astype(np.float64)  # (N,)
            B = Z @ Gamma                  # (N, K)
            BtB = B.T @ B                  # (K, K)
            try:
                f = np.linalg.lstsq(BtB, B.T @ r, rcond=None)[0]  # (K,)
            except np.linalg.LinAlgError:
                f = np.zeros(K)
            new_factors.append(f)
        factors = new_factors

        # ── Gamma update (stacked OLS via design matrix) ──────────────────────
        # For each (i, t): design row = kron(f_t, z_{i,t}) ∈ R^{K*L}
        # Normal equations: A = Σ_t X_t^T X_t,  b = Σ_t X_t^T r_t
        # where X_t = kron(f_t.reshape(1,-1), Z_t)  shape (N_t, K*L)
        KL = K * L
        A = np.zeros((KL, KL), dtype=np.float64)
        b = np.zeros(KL, dtype=np.float64)
        for d, f in zip(train_data, factors):
            Z = d["Z"].astype(np.float64)  # (N, L)
            r = d["r"].astype(np.float64)  # (N,)
            # X_t[i, k*L+l] = f[k] * Z[i, l]
            X_t = np.kron(f.reshape(1, -1), Z)  # (N, K*L)
            A += X_t.T @ X_t                    # (K*L, K*L)
            b += X_t.T @ r                      # (K*L,)

        try:
            vec_Gamma = np.linalg.lstsq(A, b, rcond=None)[0]  # (K*L,)
        except np.linalg.LinAlgError:
            break

        # vec_Gamma[k*L + l] = Gamma[l, k]
        Gamma_new = vec_Gamma.reshape(K, L).T  # (L, K)

        # Re-orthogonalise
        Q, _ = np.linalg.qr(Gamma_new)
        Gamma_new = Q[:, :K]

        delta = float(np.max(np.abs(Gamma_new - Gamma)))
        Gamma = Gamma_new

        if delta < tol:
            print(f"    IPCA ALS converged at iteration {iteration + 1}  (Δ={delta:.2e})")
            break
    else:
        print(f"    IPCA ALS reached max_iter={max_iter}  (last Δ={delta:.2e})")

    # ── Final factors & risk premium ─────────────────────────────────────────
    final_factors: list[np.ndarray] = []
    for d in train_data:
        Z = d["Z"].astype(np.float64)
        r = d["r"].astype(np.float64)
        B = Z @ Gamma
        BtB = B.T @ B
        try:
            f = np.linalg.lstsq(BtB, B.T @ r, rcond=None)[0]
        except np.linalg.LinAlgError:
            f = np.zeros(K)
        final_factors.append(f)

    lambda_hat = np.stack(final_factors).mean(axis=0)  # (K,)
    return Gamma.astype(np.float32), lambda_hat.astype(np.float32)


# ── Public API ────────────────────────────────────────────────────────────────

def train_and_predict(
    config: dict,
    K: int = 3,
    max_iter: int = 500,
    tol: float = 1e-6,
    exclude_tickers: list[str] | None = None,
) -> pd.DataFrame:
    """
    Train IPCA on the training split, predict on the test split.

    Parameters
    ----------
    config          : parsed config.yaml dict
    K               : number of latent factors
    max_iter        : maximum ALS iterations
    tol             : ALS convergence threshold (max |ΔΓ|)
    exclude_tickers : tickers to withhold from training

    Returns
    -------
    DataFrame with columns [date, ticker, mu_pred, sigma_pred, y_true].
    """
    dc = config["data"]
    exclude_set = set(exclude_tickers or [])

    loader = FundamentalsLoader(Path(dc["processed_dir"]))
    L = loader.L

    print(f"  IPCA: K={K}, L={L}, max_iter={max_iter}")

    ub_date = load_universe_by_date(config)
    ret_s   = load_returns_series(config)

    train_dates = dates_in_range(ub_date, dc["train_start"], dc["train_end"])
    test_dates  = dates_in_range(ub_date, dc["test_start"],  dc["test_end"])

    print(f"  IPCA: building training data for {len(train_dates)} dates…")
    train_data = build_date_data(train_dates, ub_date, ret_s, loader, exclude_set)
    print(f"  IPCA: {len(train_data)} usable training dates → ALS fitting…")

    Gamma, lambda_hat = _als_fit(train_data, K, max_iter, tol)
    print(f"  IPCA: Γ shape={Gamma.shape},  λ̂={np.round(lambda_hat, 4)}")

    # ── Test predictions ──────────────────────────────────────────────────────
    print(f"  IPCA: predicting on {len(test_dates)} test dates…")
    records: list[dict] = []
    for date in test_dates:
        tickers = ub_date.get(date, [])
        if not tickers:
            continue
        Z = loader.get(date, tickers)  # (N, L) — includes ALL tickers (no exclusion)
        mu = (Z @ Gamma @ lambda_hat).astype(float)  # (N,)

        for i, ticker in enumerate(tickers):
            try:
                y_true: float | float = float(ret_s.loc[(date, ticker)])
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
    print(f"  IPCA: {len(df):,} prediction rows")
    return df
