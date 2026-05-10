"""
Moderate robustness test for the Brazilian equity universe.

The original paper (China A-shares, ~3500 tickers) randomly removes 50–200 stocks
and checks that Rank IC degrades gracefully. That methodology does not translate
to a ~130-stock Brazilian universe: removing 50 stocks destroys the cross-section.

This module uses a fractional drop instead:
  - Drop `drop_frac` of each date's available stocks (default 15% ≈ 20 stocks).
  - Repeat for `n_trials` independent random seeds.
  - Report mean and std of Rank IC across trials, plus the full-universe baseline.

A well-calibrated model should show:
  - IC_mean_drop ≈ IC_full  (small degradation — signal is spread across the universe)
  - IC_std_drop  small      (stability — not driven by a handful of stocks)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from factorvae.evaluation.metrics import compute_rank_ic, compute_rank_icir


def robustness_drop_test(
    predictions: pd.DataFrame,
    drop_frac: float = 0.15,
    n_trials: int = 5,
    seed: int = 42,
) -> dict:
    """
    Assess how much Rank IC degrades when a fraction of stocks is randomly removed.

    Args:
        predictions: DataFrame with columns [date, ticker, mu_pred, y_true].
                     Rows with NaN y_true are excluded before dropping.
        drop_frac:   Fraction of stocks to drop per date per trial (0 < drop_frac < 1).
                     Default 0.15 — drops ~20 of ~130 stocks, leaving ~110.
        n_trials:    Number of independent drop trials. Default 5.
        seed:        Base random seed; trial i uses seed + i.

    Returns:
        dict with keys:
            rank_ic_full   : Rank IC on the complete universe (baseline)
            rank_ic_mean   : Mean Rank IC across all (date, trial) pairs
            rank_ic_std    : Std Rank IC across trials (per-trial means)
            drop_frac      : drop_frac used
            n_trials       : n_trials used
            avg_n_full     : Average number of stocks per date (full universe)
            avg_n_dropped  : Average number of stocks per date after dropping
    """
    predictions = predictions.copy()
    predictions["date"] = pd.to_datetime(predictions["date"])

    # ── Full-universe baseline ────────────────────────────────────────────
    full_ics: list[float] = []
    dates = sorted(predictions["date"].unique())

    for date in dates:
        grp = predictions[predictions["date"] == date].dropna(subset=["y_true"])
        if len(grp) < 5:
            continue
        y_true = torch.tensor(grp["y_true"].values, dtype=torch.float32)
        mu     = torch.tensor(grp["mu_pred"].values, dtype=torch.float32)
        full_ics.append(compute_rank_ic(y_true, mu))

    rank_ic_full = float(np.mean(full_ics)) if full_ics else float("nan")

    # ── Drop trials ───────────────────────────────────────────────────────
    trial_means: list[float] = []
    n_full_per_date: list[int]    = []
    n_dropped_per_date: list[int] = []

    for trial in range(n_trials):
        rng = np.random.default_rng(seed + trial)
        trial_ics: list[float] = []

        for date in dates:
            grp = predictions[predictions["date"] == date].dropna(subset=["y_true"])
            n = len(grp)
            if n < 5:
                continue

            n_drop = max(1, int(np.round(n * drop_frac)))
            n_keep = n - n_drop
            if n_keep < 5:
                # Guarantee a minimum cross-section of 5 regardless of universe size
                n_keep = 5
            idx = rng.choice(n, size=n_keep, replace=False)
            sub = grp.iloc[idx]

            y_true = torch.tensor(sub["y_true"].values, dtype=torch.float32)
            mu     = torch.tensor(sub["mu_pred"].values, dtype=torch.float32)
            trial_ics.append(compute_rank_ic(y_true, mu))

            if trial == 0:  # collect sizes once
                n_full_per_date.append(n)
                n_dropped_per_date.append(n_keep)

        if trial_ics:
            trial_means.append(float(np.mean(trial_ics)))

    rank_ic_mean = float(np.mean(trial_means)) if trial_means else float("nan")
    rank_ic_std  = float(np.std(trial_means, ddof=1)) if len(trial_means) > 1 else 0.0
    avg_n_full    = float(np.mean(n_full_per_date))    if n_full_per_date    else float("nan")
    avg_n_dropped = float(np.mean(n_dropped_per_date)) if n_dropped_per_date else float("nan")

    return {
        "rank_ic_full":   rank_ic_full,
        "rank_ic_mean":   rank_ic_mean,
        "rank_ic_std":    rank_ic_std,
        "drop_frac":      drop_frac,
        "n_trials":       n_trials,
        "avg_n_full":     avg_n_full,
        "avg_n_dropped":  avg_n_dropped,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Paper-faithful holdout-retrain robustness test
# ─────────────────────────────────────────────────────────────────────────────

def robustness_holdout_train_test(
    config: dict,
    m: int,
    n_trials: int = 3,
    seed: int = 42,
    max_epochs_override: int | None = None,
    progress: bool = True,
) -> list[dict]:
    """
    Paper-faithful robustness test: remove m stocks from the training set,
    retrain the model, then evaluate Rank IC only on those held-out stocks.

    For each trial:
      1. Randomly select m tickers S from the training universe.
      2. Train a fresh FactorVAE on D_train \\ S (all training dates, minus those m tickers).
      3. Predict on the full test set (all tickers, including S).
      4. Filter predictions to S only.
      5. Compute Rank IC and Rank ICIR averaged across test dates.

    Args:
        config:               Parsed config.yaml dict.
        m:                    Number of tickers to hold out per trial.
        n_trials:             Number of independent trials.
        seed:                 Base RNG seed (trial i uses seed + i).
        max_epochs_override:  Override config's max_epochs (useful for quick tests).
        progress:             Print progress to stdout.

    Returns:
        List of dicts, one per trial, with keys:
            trial           : int
            held_out        : list[str]
            rank_ic_holdout : float   — mean Rank IC over test dates on held-out stocks
            rank_icir_holdout: float  — Rank ICIR on held-out stocks
            n_dates_with_holdout: int — test dates where ≥1 held-out stock appeared
    """
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping
    from torch.utils.data import DataLoader
    from tqdm import tqdm as _tqdm
    import tempfile

    from factorvae.data.dataset import RealDataset
    from factorvae.data.datamodule import FactorVAEDataModule
    from factorvae.models.factorvae import FactorVAE
    from factorvae.training.lightning_module import FactorVAELightning
    from factorvae.utils.seeding import seed_everything

    dc    = config["data"]
    mc    = config["model"]
    tc    = config["training"]
    pdir  = Path(dc["processed_dir"])

    max_epochs = max_epochs_override if max_epochs_override is not None else tc["max_epochs"]

    # ── Discover full training universe ──────────────────────────────────────
    universe_long = pd.read_parquet(pdir / "universe.parquet")
    universe_long["date"] = pd.to_datetime(universe_long["date"])
    train_start = pd.Timestamp(dc["train_start"])
    train_end   = pd.Timestamp(dc["train_end"])
    train_universe_mask = (
        (universe_long["date"] >= train_start) &
        (universe_long["date"] <= train_end)
    )
    all_train_tickers: list[str] = sorted(
        universe_long.loc[train_universe_mask, "ticker"].unique().tolist()
    )
    total = len(all_train_tickers)
    if progress:
        print(f"\n── Holdout robustness test ─────────────────────────────────────────")
        print(f"   Training universe: {total} unique tickers")
        print(f"   Holding out m={m} per trial, n_trials={n_trials}")

    if m >= total:
        raise ValueError(
            f"m={m} ≥ training universe size ({total}). "
            "Reduce m to leave at least 1 ticker for training."
        )

    # ── Build val/test datasets (shared across all trials — no holdout) ───────
    val_ds  = RealDataset(pdir, dc["val_start"],  dc["val_end"],  dc["sequence_length"])
    test_ds = RealDataset(pdir, dc["test_start"], dc["test_end"], dc["sequence_length"])
    val_dl  = DataLoader(val_ds,  batch_size=1, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: list[dict] = []

    for trial in range(n_trials):
        rng = np.random.default_rng(seed + trial)
        held_out: list[str] = list(rng.choice(all_train_tickers, size=m, replace=False))
        held_out_set = set(held_out)

        if progress:
            print(f"\n   Trial {trial + 1}/{n_trials} — held-out: {held_out[:5]}{'...' if m > 5 else ''}")

        seed_everything(tc["seed"] + trial)

        # ── Train dataset: exclude held-out tickers ───────────────────────
        train_ds = RealDataset(
            pdir, dc["train_start"], dc["train_end"], dc["sequence_length"],
            exclude_tickers=held_out,
        )
        train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)

        # ── Fresh model ───────────────────────────────────────────────────
        model = FactorVAE(config)
        lm = FactorVAELightning(model, config)

        # Create temporary directory for checkpoints
        with tempfile.TemporaryDirectory() as tmpdir:
            early_stop = EarlyStopping(
                monitor="val_rank_ic",
                patience=15,
                mode="max",
                verbose=False,
            )
            trainer = L.Trainer(
                max_epochs=max_epochs,
                accelerator="auto",
                devices=1,
                enable_checkpointing=True,
                default_root_dir=tmpdir,
                callbacks=[early_stop],
                enable_progress_bar=progress,
                enable_model_summary=False,
                logger=False,
            )
            trainer.fit(lm, train_dl, val_dl)

        # ── Inference on full test set ────────────────────────────────────
        lm.model.to(device)
        lm.model.eval()

        date_ics:  list[float] = []
        date_icirs: list[float] = []
        n_dates_with_holdout = 0

        with torch.no_grad():
            for idx in range(len(test_ds)):
                batch = test_ds[idx]
                if len(batch) == 4:
                    x, mac, y_z, mask = batch
                    x   = x.float().to(device)
                    mac = mac.float().to(device)
                    mu_pred, _ = lm.model.forward_predict(x, m=mac)
                else:
                    x, y_z, mask = batch
                    x = x.float().to(device)
                    mu_pred, _ = lm.model.forward_predict(x)

                mu_pred = mu_pred.cpu()
                date_ts = test_ds.trading_dates[idx]
                tickers_at_date = test_ds.universe_by_date[date_ts]

                # Filter to held-out tickers only
                held_indices = [
                    i for i, t in enumerate(tickers_at_date) if t in held_out_set
                ]
                if len(held_indices) < 2:
                    continue

                n_dates_with_holdout += 1
                y_sub  = y_z[held_indices]
                mu_sub = mu_pred[held_indices]
                date_ics.append(compute_rank_ic(y_sub, mu_sub))

        mean_ic   = float(np.mean(date_ics))   if date_ics else float("nan")
        mean_icir = compute_rank_icir(date_ics) if date_ics else float("nan")

        if progress:
            print(f"   → Rank IC on held-out: {mean_ic:+.4f}  |  "
                  f"ICIR: {mean_icir:+.4f}  |  "
                  f"dates with ≥2 held-out: {n_dates_with_holdout}")

        results.append({
            "trial":              trial,
            "held_out":           held_out,
            "rank_ic_holdout":    mean_ic,
            "rank_icir_holdout":  mean_icir,
            "n_dates_with_holdout": n_dates_with_holdout,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Multi-model holdout robustness (Experiment 2 — paper-faithful)
# ─────────────────────────────────────────────────────────────────────────────

def _ic_on_held_out(preds_df: pd.DataFrame, held_out_set: set) -> tuple[float, float]:
    """Compute mean Rank IC and Rank ICIR for a model on held-out stocks only."""
    held = preds_df[preds_df["ticker"].isin(held_out_set)].copy()
    held["date"] = pd.to_datetime(held["date"])
    ics: list[float] = []
    for _, grp in held.groupby("date"):
        grp = grp.dropna(subset=["y_true"])
        if len(grp) < 2:
            continue
        y_t = torch.tensor(grp["y_true"].values, dtype=torch.float32)
        mu  = torch.tensor(grp["mu_pred"].values,  dtype=torch.float32)
        ics.append(compute_rank_ic(y_t, mu))
    mean_ic = float(np.mean(ics)) if ics else float("nan")
    icir    = compute_rank_icir(ics) if len(ics) > 1 else float("nan")
    return mean_ic, icir


def robustness_holdout_all_models(
    config: dict,
    m: int,
    n_trials: int = 3,
    seed: int = 42,
    max_epochs_override: int | None = None,
    progress: bool = True,
) -> list[dict]:
    """
    Paper-faithful multi-model holdout robustness test (Experiment 2).

    Each trial holds out the SAME m tickers from training for ALL models,
    enabling a fair cross-model comparison of out-of-sample generalisation.

    Models evaluated:
      - FactorVAE : retrained without held-out tickers
      - GRU       : retrained without held-out tickers
      - IPCA      : re-estimated without held-out tickers
      - CA        : retrained without held-out tickers

    Args:
        config:              Parsed config.yaml dict.
        m:                   Number of tickers to hold out per trial.
        n_trials:            Number of independent trials.
        seed:                Base RNG seed; trial i uses seed + i.
        max_epochs_override: Override config's max_epochs for quick testing.
        progress:            Print progress to stdout.

    Returns:
        List of dicts, one per trial:
            trial      : int
            m          : int
            held_out   : list[str]
            results    : dict[model_name -> {"rank_ic": float, "rank_icir": float}]
    """
    import sys
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping
    from pathlib import Path as _Path
    from torch.utils.data import DataLoader
    import tempfile

    from factorvae.data.dataset import RealDataset
    from factorvae.data.datamodule import FactorVAEDataModule
    from factorvae.models.factorvae import FactorVAE
    from factorvae.training.lightning_module import FactorVAELightning
    from factorvae.utils.seeding import seed_everything

    # Add project root to sys.path so benchmarks/ is importable
    _root = _Path(__file__).resolve().parents[3]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from benchmarks import gru  as _gru
    from benchmarks import ipca as _ipca
    from benchmarks import ca   as _ca

    dc   = config["data"]
    tc   = config["training"]
    pdir = _Path(dc["processed_dir"])
    max_epochs = max_epochs_override if max_epochs_override is not None else tc["max_epochs"]

    # ── Discover full training universe ───────────────────────────────────────
    universe_long = pd.read_parquet(pdir / "universe.parquet")
    universe_long["date"] = pd.to_datetime(universe_long["date"])
    train_start = pd.Timestamp(dc["train_start"])
    train_end   = pd.Timestamp(dc["train_end"])
    train_mask  = (universe_long["date"] >= train_start) & (universe_long["date"] <= train_end)
    all_train_tickers: list[str] = sorted(
        universe_long.loc[train_mask, "ticker"].unique().tolist()
    )
    total = len(all_train_tickers)

    if m >= total:
        raise ValueError(
            f"m={m} \u2265 training universe size ({total}). "
            "Reduce m to leave at least 1 ticker for training."
        )

    if progress:
        print(f"\n\u2500\u2500 Multi-model holdout robustness (m={m}) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
        print(f"   Training universe: {total} tickers | held-out m={m} | trials={n_trials}")

    # Build shared val/test datasets (unchanged across trials)
    val_ds  = RealDataset(pdir, dc["val_start"],  dc["val_end"],  dc["sequence_length"])
    test_ds = RealDataset(pdir, dc["test_start"], dc["test_end"], dc["sequence_length"])
    val_dl  = DataLoader(val_ds,  batch_size=1, shuffle=False, num_workers=0)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_trials: list[dict] = []

    for trial in range(n_trials):
        rng      = np.random.default_rng(seed + trial)
        held_out: list[str] = list(rng.choice(all_train_tickers, size=m, replace=False))
        held_out_set = set(held_out)

        if progress:
            print(f"\n   Trial {trial + 1}/{n_trials} — held-out: "
                  f"{held_out[:3]}{'...' if m > 3 else ''} ({m} stocks)")

        trial_results: dict[str, dict] = {}

        # ── FactorVAE ─────────────────────────────────────────────────────
        if progress:
            print("   [1/5] FactorVAE: retraining…")
        seed_everything(tc["seed"] + trial)
        train_ds_fv = RealDataset(
            pdir, dc["train_start"], dc["train_end"], dc["sequence_length"],
            exclude_tickers=held_out,
        )
        train_dl_fv = DataLoader(train_ds_fv, batch_size=1, shuffle=True, num_workers=0)
        _cfg_override = {**config, "training": {**tc, "max_epochs": max_epochs}}
        fv_model = FactorVAE(_cfg_override)
        fv_lm    = FactorVAELightning(fv_model, _cfg_override)
        
        # Create temporary directory for checkpoints
        with tempfile.TemporaryDirectory() as tmpdir:
            early_stop = EarlyStopping(
                monitor="val_rank_ic",
                patience=15,
                mode="max",
                verbose=False,
            )
            trainer  = L.Trainer(
                max_epochs=max_epochs,
                accelerator="auto",
                devices=1,
                enable_checkpointing=True,
                default_root_dir=tmpdir,
                callbacks=[early_stop],
                enable_progress_bar=progress,
                enable_model_summary=False,
                logger=False,
            )
            trainer.fit(fv_lm, train_dl_fv, val_dl)
        fv_lm.model.to(device)
        fv_lm.model.eval()
        fv_date_ics: list[float] = []
        with torch.no_grad():
            for idx in range(len(test_ds)):
                batch = test_ds[idx]
                if len(batch) == 4:
                    x, mac, y_z, mask = batch
                    mu_pred, _ = fv_lm.model.forward_predict(
                        x.float().to(device), m=mac.float().to(device)
                    )
                else:
                    x, y_z, mask = batch
                    mu_pred, _ = fv_lm.model.forward_predict(x.float().to(device))
                mu_pred  = mu_pred.cpu()
                date_ts  = test_ds.trading_dates[idx]
                tickers  = test_ds.universe_by_date[date_ts]
                held_idx = [i for i, t in enumerate(tickers) if t in held_out_set]
                if len(held_idx) < 2:
                    continue
                fv_date_ics.append(compute_rank_ic(y_z[held_idx], mu_pred[held_idx]))
        fv_ic   = float(np.mean(fv_date_ics))   if fv_date_ics else float("nan")
        fv_icir = compute_rank_icir(fv_date_ics) if fv_date_ics else float("nan")
        trial_results["FactorVAE"] = {"rank_ic": fv_ic, "rank_icir": fv_icir}
        if progress:
            print(f"      → Rank IC={fv_ic:+.4f}  ICIR={fv_icir:+.4f}")

        # ── GRU ───────────────────────────────────────────────────────────
        if progress:
            print("   [2/3] GRU: retraining…")
        gru_cfg   = {**config, "training": {**tc, "max_epochs": max_epochs}}
        gru_preds = _gru.train_and_predict(gru_cfg, exclude_tickers=held_out)
        gru_ic, gru_icir = _ic_on_held_out(gru_preds, held_out_set)
        trial_results["GRU"] = {"rank_ic": gru_ic, "rank_icir": gru_icir}
        if progress:
            print(f"      → Rank IC={gru_ic:+.4f}  ICIR={gru_icir:+.4f}")

        # ── IPCA ──────────────────────────────────────────────────────────
        if progress:
            print("   [3/3] IPCA: re-estimating…")
        ipca_preds = _ipca.train_and_predict(config, exclude_tickers=held_out)
        ipca_ic, ipca_icir = _ic_on_held_out(ipca_preds, held_out_set)
        trial_results["IPCA"] = {"rank_ic": ipca_ic, "rank_icir": ipca_icir}
        if progress:
            print(f"      → Rank IC={ipca_ic:+.4f}  ICIR={ipca_icir:+.4f}")

        # ── CA ────────────────────────────────────────────────────────────
        if progress:
            print("   [4/4] CA: retraining…")
        ca_cfg   = {**config, "training": {**tc, "max_epochs": max_epochs}}
        ca_preds = _ca.train_and_predict(ca_cfg, exclude_tickers=held_out)
        ca_ic, ca_icir = _ic_on_held_out(ca_preds, held_out_set)
        trial_results["CA"] = {"rank_ic": ca_ic, "rank_icir": ca_icir}
        if progress:
            print(f"      → Rank IC={ca_ic:+.4f}  ICIR={ca_icir:+.4f}")

        all_trials.append({
            "trial":   trial,
            "m":       m,
            "held_out": held_out,
            "results": trial_results,
        })

    return all_trials
