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
    from torch.utils.data import DataLoader
    from tqdm import tqdm as _tqdm

    from factorvae.data.dataset import RealDataset
    from factorvae.data.datamodule import FactorVAEDataModule, MacroNormalizer
    from factorvae.models.factorvae import FactorVAE
    from factorvae.training.lightning_module import FactorVAELightning
    from factorvae.utils.seeding import seed_everything

    dc    = config["data"]
    mc    = config["model"]
    tc    = config["training"]
    pdir  = Path(dc["processed_dir"])

    use_macro = dc.get("use_macro", False)
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

    # ── Build macro normalizer once (if needed) ───────────────────────────────
    macro_normalizer: "MacroNormalizer | None" = None
    if use_macro:
        macro_wide = (
            pd.read_parquet(pdir / "macro.parquet")
            .assign(date=lambda df: pd.to_datetime(df["date"]))
            .pivot(index="date", columns="feature_name", values="value")
            .sort_index()
            .ffill()
        )
        macro_normalizer = MacroNormalizer(macro_wide, dc["train_start"], dc["train_end"])

    # ── Build val/test datasets (shared across all trials — no holdout) ───────
    ds_kwargs = {"use_macro": use_macro, "macro_normalizer": macro_normalizer}
    val_ds  = RealDataset(pdir, dc["val_start"],  dc["val_end"],  dc["sequence_length"], **ds_kwargs)
    test_ds = RealDataset(pdir, dc["test_start"], dc["test_end"], dc["sequence_length"], **ds_kwargs)
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
            **ds_kwargs,
        )
        train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)

        # ── Fresh model ───────────────────────────────────────────────────
        model = FactorVAE(config)
        lm = FactorVAELightning(model, config)

        trainer = L.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices=1,
            enable_checkpointing=False,
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
