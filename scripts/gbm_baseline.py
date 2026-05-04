"""
Day 0.5 diagnostic — GBM feature-to-return baseline (upper-bound IC check).

Trains a gradient-boosted tree regressor on the pooled (stock, date) training
panel — using the last time-step cross-sectionally z-scored features to predict
z-scored forward returns — then evaluates per-date Rank IC on the test set.

This IC is the practical upper bound for any pure feature-only model:
  • If GBM IC ≈ prior IC:  the prior predictor is near-optimal; architecture
                           changes will not help — no more signal to extract.
  • If GBM IC >> prior IC: there is untapped signal in the features; the
                           predictor architecture / training is the bottleneck.

Uses sklearn HistGradientBoostingRegressor (no extra dependency).
If lightgbm is installed it is used automatically for speed.

Usage
-----
    python scripts/gbm_baseline.py
    python scripts/gbm_baseline.py --checkpoint results/checkpoints/best.ckpt
    python scripts/gbm_baseline.py --use-sequence          # flatten full T-step window
    python scripts/gbm_baseline.py --n-estimators 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from factorvae.data.datamodule import FactorVAEDataModule
from factorvae.evaluation.metrics import compute_rank_ic
from factorvae.models.factorvae import FactorVAE
from factorvae.training.lightning_module import FactorVAELightning
from factorvae.utils.seeding import seed_everything

ROOT = Path(__file__).resolve().parents[1]


def _build_panel(
    dataset,
    use_sequence: bool,
    desc: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, date_labels) pooled over all dates in the dataset."""
    X_list, y_list, date_list = [], [], []

    for idx in tqdm(range(len(dataset)), desc=desc, unit="date"):
        batch = dataset[idx]
        # (x, y, mask) or (x, m, y, mask)
        x = batch[0].numpy()    # (N, T, C)
        y = batch[-2].numpy()   # (N,)

        feats = x.reshape(x.shape[0], -1) if use_sequence else x[:, -1, :]
        X_list.append(feats)
        y_list.append(y)

        date_label = (
            dataset.trading_dates[idx].strftime("%Y-%m-%d")
            if hasattr(dataset, "trading_dates")
            else str(idx)
        )
        date_list.extend([date_label] * x.shape[0])

    return np.vstack(X_list), np.concatenate(y_list), date_list


def _build_gbm(n_estimators: int, seed: int):
    """Return a fitted-ready GBM model (lgb preferred, sklearn fallback)."""
    try:
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        ), "LightGBM"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_iter=n_estimators,
            learning_rate=0.05,
            max_leaf_nodes=63,
            min_samples_leaf=20,
            random_state=seed,
        ), "HistGBM (sklearn)"


def _eval_prior(model_vae, dataset, device) -> pd.DataFrame:
    """Evaluate FactorVAE prior (production) IC on the dataset for comparison."""
    records = []
    model_vae.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            batch = dataset[idx]
            if len(batch) == 4:
                x, m, y, _ = batch
                m = m.float().to(device)
            else:
                x, y, _ = batch
                m = None
            x = x.float().to(device)
            y_cpu = y.float()

            mu_pred, _ = model_vae.forward_predict(x, m=m)
            rank_ic = compute_rank_ic(y_cpu, mu_pred.cpu())
            date_label = (
                dataset.trading_dates[idx].strftime("%Y-%m-%d")
                if hasattr(dataset, "trading_dates")
                else str(idx)
            )
            records.append({"date": date_label, "prior_ic": rank_ic})
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        default=str(ROOT / "config.yaml"))
    parser.add_argument("--checkpoint",    default=str(ROOT / "results" / "checkpoints" / "best.ckpt"))
    parser.add_argument("--use-sequence",  action="store_true",
                        help="Flatten full T-step window; default uses last timestep only.")
    parser.add_argument("--n-estimators",  type=int, default=500)
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    ckpt_data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "hyper_parameters" in ckpt_data:
        config = ckpt_data["hyper_parameters"]

    seed_everything(args.seed)

    # ── GBM model factory ────────────────────────────────────────────────────
    gbm, gbm_name = _build_gbm(args.n_estimators, args.seed)
    feature_mode = "full T-step sequence (flattened)" if args.use_sequence else "last time-step only"
    print(f"\nGBM engine  : {gbm_name}")
    print(f"Features    : {feature_mode}")
    print(f"Estimators  : {args.n_estimators}")

    # ── Data ─────────────────────────────────────────────────────────────────
    datamodule = FactorVAEDataModule(config)
    datamodule.setup()

    # ── Build training panel ─────────────────────────────────────────────────
    print()
    X_train, y_train, _ = _build_panel(datamodule._train, args.use_sequence, "Building train panel")
    X_val,   y_val,   _ = _build_panel(datamodule._val,   args.use_sequence, "Building val panel  ")
    T = config["data"]["sequence_length"]
    C = config["model"]["num_features"]
    n_feats = T * C if args.use_sequence else C
    print(f"\nTrain panel : {X_train.shape[0]:,} observations  x  {n_feats} features")
    print(f"Val panel   : {X_val.shape[0]:,} observations")

    # ── Train GBM ────────────────────────────────────────────────────────────
    print(f"\nTraining {gbm_name}...", flush=True)
    gbm.fit(X_train, y_train)
    print("Training complete.")

    # ── Per-date IC on val and test sets ─────────────────────────────────────
    def _per_date_ic(dataset, split_label: str) -> pd.DataFrame:
        records = []
        for idx in tqdm(range(len(dataset)), desc=f"Eval {split_label}", unit="date"):
            batch = dataset[idx]
            x = batch[0].numpy()
            y = batch[-2]

            feats = x.reshape(x.shape[0], -1) if args.use_sequence else x[:, -1, :]
            preds = gbm.predict(feats).astype(np.float32)

            y_t   = torch.tensor(y.numpy() if hasattr(y, "numpy") else np.array(y), dtype=torch.float32)
            p_t   = torch.tensor(preds, dtype=torch.float32)
            rank_ic = compute_rank_ic(y_t, p_t)

            date_label = (
                dataset.trading_dates[idx].strftime("%Y-%m-%d")
                if hasattr(dataset, "trading_dates")
                else str(idx)
            )
            records.append({
                "date":    date_label,
                "gbm_ic":  rank_ic,
                "n_stocks": x.shape[0],
            })
        return pd.DataFrame(records)

    df_val  = _per_date_ic(datamodule._val,  "val ")
    df_test = _per_date_ic(datamodule._test, "test")

    # ── Load FactorVAE prior for side-by-side comparison ─────────────────────
    vae_model = FactorVAE(config)
    lm = FactorVAELightning.load_from_checkpoint(
        args.checkpoint, model=vae_model, config=config
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm.model.to(device)

    print("\nEvaluating FactorVAE prior IC for comparison...", flush=True)
    prior_val  = _eval_prior(lm.model, datamodule._val,  device)
    prior_test = _eval_prior(lm.model, datamodule._test, device)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("GBM BASELINE IC  vs  FACTORVAE PRIOR IC")
    print(f"Engine: {gbm_name}  |  Features: {feature_mode}")
    print("=" * 65)

    for split_label, df_gbm, df_prior in [
        ("Val",  df_val,  prior_val),
        ("Test", df_test, prior_test),
    ]:
        gbm_mean  = df_gbm["gbm_ic"].mean()
        gbm_std   = df_gbm["gbm_ic"].std()
        gbm_pos   = (df_gbm["gbm_ic"] > 0).mean() * 100

        prior_mean = df_prior["prior_ic"].mean()
        prior_std  = df_prior["prior_ic"].std()
        prior_pos  = (df_prior["prior_ic"] > 0).mean() * 100

        gap = gbm_mean - prior_mean

        print(f"\n  {split_label} set ({len(df_gbm)} dates):")
        print(f"  {'Metric':<30}  {'GBM':>8}  {'Prior':>8}  {'Gap':>8}")
        print("  " + "-" * 58)
        print(f"  {'Mean Rank IC':<30}  {gbm_mean:>+8.4f}  {prior_mean:>+8.4f}  {gap:>+8.4f}")
        print(f"  {'Std  Rank IC':<30}  {gbm_std:>8.4f}  {prior_std:>8.4f}")
        print(f"  {'% Positive IC':<30}  {gbm_pos:>7.1f}%  {prior_pos:>7.1f}%")

    # ── Interpretation ────────────────────────────────────────────────────────
    test_gbm_mean   = df_test["gbm_ic"].mean()
    test_prior_mean = prior_test["prior_ic"].mean()
    gap_test        = test_gbm_mean - test_prior_mean

    print("\nINTERPRETATION:")
    if abs(gap_test) < 0.02:
        print(f"  [NEAR-OPTIMAL] GBM IC ({test_gbm_mean:+.4f}) ~= Prior IC ({test_prior_mean:+.4f})")
        print("  The predictor is already near the feature-extractable upper bound.")
        print("  Architecture changes are unlikely to help.")
    elif gap_test > 0.02:
        print(f"  [GAP EXISTS] GBM IC ({test_gbm_mean:+.4f}) > Prior IC ({test_prior_mean:+.4f})")
        print(f"  Gap = {gap_test:+.4f}. Untapped signal in features.")
        print("  The predictor architecture / training is the bottleneck.")
        print("  Closing this gap is tractable through architecture improvements.")
    else:
        print(f"  [PRIOR BEATS GBM] Prior IC ({test_prior_mean:+.4f}) > GBM IC ({test_gbm_mean:+.4f})")
        print("  The VAE prior predictor outperforms a flat tree model.")
        print("  The temporal/cross-sectional structure in FactorVAE is adding value.")

    # ── Per-year GBM IC on test ───────────────────────────────────────────────
    df_test["date"] = pd.to_datetime(df_test["date"])
    prior_test["date"] = pd.to_datetime(prior_test["date"])
    merged = df_test.merge(prior_test, on="date", how="inner")
    merged["year"] = merged["date"].dt.year

    print("\nYEAR-BY-YEAR (TEST SET):")
    print(f"  {'Year':<6}  {'N':>5}  {'GBM IC':>8}  {'Prior IC':>9}  {'Gap':>8}")
    print("  " + "-" * 42)
    for year, grp in merged.groupby("year"):
        g = grp["gbm_ic"].mean()
        p = grp["prior_ic"].mean()
        print(f"  {year:<6}  {len(grp):>5}  {g:>+8.4f}  {p:>+9.4f}  {g - p:>+8.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir = ROOT / "results" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_test.to_csv(out_dir / "gbm_baseline_test.csv", index=False)
    df_val.to_csv(out_dir / "gbm_baseline_val.csv", index=False)
    print(f"\nResults saved to: {(out_dir / 'gbm_baseline_test.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
