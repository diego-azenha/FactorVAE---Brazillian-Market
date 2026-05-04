"""
Day 0.5 diagnostic — Cross-fit Posterior IC (leakage falsification test).

The leakage concern
-------------------
The encoder computes mu_post(t) by processing x_t AND y_t where
y_t = forward_return_{t→t+1}.  Evaluating IC(mu_y_rec(t), y_t) is therefore
reconstruction accuracy, not prediction — the oracle has already seen the answer.

The cross-fitting test
-----------------------
For each consecutive pair (t, t+1) in the held-out test set:
  1.  forward_train(x_t, y_t)  → mu_y_rec(t)    [posterior decoder output, oracle]
  2.  forward_predict(x_t)     → mu_y_pred(t)   [prior decoder output, production]
  3.  Intersect universes t ∩ t+1 → common tickers
  4.  Compute:
        crossfit_post_ic  = IC( mu_y_rec(t)[common],  y_{t+1}[common] )
        crossfit_prior_ic = IC( mu_y_pred(t)[common], y_{t+1}[common] )
        insample_post_ic  = IC( mu_y_rec(t),           y_t             )

The gap (crossfit_post_ic − crossfit_prior_ic) isolates the predictive contribution
of encoding y_t beyond what the feature extractor alone already captures.

Decision rule
-------------
  crossfit_post_ic > 0.10 AND drops < 50 % vs insample_post_ic
      → teacher-student gap is real; proceed with Option 7
  crossfit_post_ic drops ≥ 50 % or ≈ crossfit_prior_ic
      → posterior is largely a reconstruction artifact; write honest paper

Also reports
------------
  • Per-regime IC breakdown (COVID 2020, Selic 2022, Lula 2023, Other)
  • Year-by-year mean cross-fit IC
  • Per-dimension KL diagnostic (active factor units)

Usage
-----
    python scripts/crossfit_posterior_ic.py
    python scripts/crossfit_posterior_ic.py --checkpoint results/checkpoints/best.ckpt
    python scripts/crossfit_posterior_ic.py --split val
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

MIN_STOCKS = 5          # skip dates with fewer common stocks
SIGMA_FLOOR = 1e-6

REGIMES: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "COVID (2020)":      (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31")),
    "Selic hike (2022)": (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
    "Lula (2023)":       (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")),
}


def _kl_per_dim(
    mu_post: torch.Tensor,
    sigma_post: torch.Tensor,
    mu_prior: torch.Tensor,
    sigma_prior: torch.Tensor,
) -> torch.Tensor:
    """Return per-factor KL(posterior_k || prior_k), shape (K,)."""
    sq = sigma_post.clamp(min=SIGMA_FLOOR)
    sp = sigma_prior.clamp(min=SIGMA_FLOOR)
    return torch.log(sp / sq) + (sq**2 + (mu_post - mu_prior)**2) / (2.0 * sp**2) - 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default=str(ROOT / "config.yaml"))
    parser.add_argument("--checkpoint", default=str(ROOT / "results" / "checkpoints" / "best.ckpt"))
    parser.add_argument("--split",      default="test", choices=["val", "test"])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    ckpt_data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "hyper_parameters" in ckpt_data:
        config = ckpt_data["hyper_parameters"]

    seed_everything(config["training"]["seed"])

    # ── Load model ─────────────────────────────────────────────────────────
    model_obj = FactorVAE(config)
    lm = FactorVAELightning.load_from_checkpoint(
        args.checkpoint, model=model_obj, config=config
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm.model.to(device)
    lm.model.eval()

    # ── Data ───────────────────────────────────────────────────────────────
    datamodule = FactorVAEDataModule(config)
    datamodule.setup()
    dataset = datamodule._val if args.split == "val" else datamodule._test

    n_dates = len(dataset)
    K = config["model"]["num_factors"]

    records: list[dict] = []
    kl_per_dim_accum: list[torch.Tensor] = []   # one (K,) per date

    with torch.no_grad():
        for idx in tqdm(range(n_dates - 1), desc=f"Cross-fit {args.split}", unit="date"):
            date_t  = dataset.trading_dates[idx]
            date_t1 = dataset.trading_dates[idx + 1]

            tickers_t  = dataset.universe_by_date[date_t]
            tickers_t1 = dataset.universe_by_date[date_t1]
            common = sorted(set(tickers_t) & set(tickers_t1))
            if len(common) < MIN_STOCKS:
                continue

            # Indices into the stock vectors for each date
            t_idx  = [tickers_t.index(tkr)  for tkr in common]
            t1_idx = [tickers_t1.index(tkr) for tkr in common]

            # ── Batch t ────────────────────────────────────────────────────
            batch_t = dataset[idx]
            if len(batch_t) == 4:
                x_t, m_t, y_t, _ = batch_t
                m_t = m_t.float().to(device)
            else:
                x_t, y_t, _ = batch_t
                m_t = None
            x_t = x_t.float().to(device)

            # ── Batch t+1: we only need y_{t+1} ───────────────────────────
            batch_t1 = dataset[idx + 1]
            y_t1 = batch_t1[-2].float()   # y is second-to-last in both (x,y,mask) and (x,m,y,mask)

            # ── Oracle forward pass (encoder sees y_t) ─────────────────────
            out = lm.model.forward_train(x_t, y_t.to(device), m=m_t)
            mu_y_rec  = out["mu_y_rec"].cpu()   # (N_t,)
            mu_post   = out["mu_post"].cpu()    # (K,) factor means
            sigma_post = out["sigma_post"].cpu()
            mu_prior_  = out["mu_prior"].cpu()
            sigma_prior_ = out["sigma_prior"].cpu()

            # ── Production forward pass (predictor only) ───────────────────
            mu_y_pred, _ = lm.model.forward_predict(x_t, m=m_t)
            mu_y_pred = mu_y_pred.cpu()   # (N_t,)

            # ── Per-dimension KL (active units) ────────────────────────────
            kl_k = _kl_per_dim(mu_post, sigma_post, mu_prior_, sigma_prior_)
            kl_per_dim_accum.append(kl_k)

            # ── IC calculations ────────────────────────────────────────────
            # In-sample: oracle predictions vs contemporaneous y_t (all stocks)
            insample_post_ic  = compute_rank_ic(y_t.float(),  mu_y_rec.float())
            insample_prior_ic = compute_rank_ic(y_t.float(),  mu_y_pred.float())

            # Cross-fit: predictions from date t evaluated on date t+1 returns
            mu_rec_common  = mu_y_rec[t_idx]
            mu_pred_common = mu_y_pred[t_idx]
            y_t1_common    = y_t1[t1_idx]

            crossfit_post_ic  = compute_rank_ic(y_t1_common, mu_rec_common.float())
            crossfit_prior_ic = compute_rank_ic(y_t1_common, mu_pred_common.float())

            records.append({
                "date":               date_t.strftime("%Y-%m-%d"),
                "n_stocks":           len(tickers_t),
                "n_common":           len(common),
                "insample_post_ic":   insample_post_ic,
                "insample_prior_ic":  insample_prior_ic,
                "crossfit_post_ic":   crossfit_post_ic,
                "crossfit_prior_ic":  crossfit_prior_ic,
                "net_crossfit_gap":   crossfit_post_ic - crossfit_prior_ic,
            })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ── Summary table ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"CROSS-FIT POSTERIOR IC  --  split={args.split.upper()}")
    print(f"Checkpoint : {Path(args.checkpoint).name}")
    print(f"Dates      : {len(df)}  (consecutive pairs with >= {MIN_STOCKS} common stocks)")
    print("=" * 70)

    metrics = {
        "In-sample posterior IC":   df["insample_post_ic"],
        "In-sample prior IC":       df["insample_prior_ic"],
        "Cross-fit posterior IC":   df["crossfit_post_ic"],
        "Cross-fit prior IC":       df["crossfit_prior_ic"],
        "Net cross-fit gap (P-Q)":  df["net_crossfit_gap"],
    }
    header = f"  {'Metric':<30}  {'Mean':>8}  {'Std':>8}  {'%Pos':>6}"
    print(header)
    print("  " + "-" * 56)
    for label, series in metrics.items():
        print(
            f"  {label:<30}  {series.mean():>+8.4f}  {series.std():>8.4f}  "
            f"{(series > 0).mean() * 100:>5.1f}%"
        )

    # ── Decision interpretation ─────────────────────────────────────────────
    mean_crossfit_post  = df["crossfit_post_ic"].mean()
    mean_insample_post  = df["insample_post_ic"].mean()
    mean_crossfit_prior = df["crossfit_prior_ic"].mean()

    retention = (
        mean_crossfit_post / mean_insample_post
        if abs(mean_insample_post) > 1e-6
        else float("nan")
    )

    print()
    print("DECISION:")
    if mean_crossfit_post > 0.10 and retention > 0.50:
        print(f"  [PROCEED] Cross-fit posterior IC = {mean_crossfit_post:+.4f} "
              f"({retention * 100:.0f}% of in-sample {mean_insample_post:+.4f})")
        print("  Teacher-student gap is real. Proceed with Option 7 and full plan.")
    elif mean_crossfit_post <= 0.10 or retention <= 0.50:
        print(f"  [CAUTION] Cross-fit posterior IC = {mean_crossfit_post:+.4f} "
              f"({retention * 100:.0f}% of in-sample {mean_insample_post:+.4f})")
        print("  Posterior IC degrades significantly under cross-fitting.")
        print("  The posterior is largely a reconstruction artifact.")
        print("  Option 7 will distill noise. Write the honest paper instead.")
    else:
        print(f"  [BORDERLINE] Cross-fit posterior IC = {mean_crossfit_post:+.4f}; "
              f"retention = {retention * 100:.0f}%. Investigate further.")

    net_gap = mean_crossfit_post - mean_crossfit_prior
    print(f"\n  Net predictive contribution of encoding y_t: {net_gap:+.4f}")
    print(f"  (cross-fit posterior IC minus cross-fit prior IC)")

    # ── Per-regime breakdown ────────────────────────────────────────────────
    print("\nPER-REGIME CROSS-FIT IC:")
    print(f"  {'Regime':<22}  {'N':>5}  {'CF-Post':>8}  {'CF-Prior':>9}  {'Net Gap':>8}")
    print("  " + "-" * 56)

    assigned: set[int] = set()
    for regime_name, (start_ts, end_ts) in REGIMES.items():
        mask = (df["date"] >= start_ts) & (df["date"] <= end_ts)
        idx_set = set(df[mask].index)
        assigned |= idx_set
        sub = df[mask]
        if len(sub) == 0:
            continue
        cf_post  = sub["crossfit_post_ic"].mean()
        cf_prior = sub["crossfit_prior_ic"].mean()
        print(
            f"  {regime_name:<22}  {len(sub):>5}  {cf_post:>+8.4f}  "
            f"{cf_prior:>+9.4f}  {cf_post - cf_prior:>+8.4f}"
        )

    # "Other" regime = everything not in a named regime
    other = df[~df.index.isin(assigned)]
    if len(other) > 0:
        cf_post  = other["crossfit_post_ic"].mean()
        cf_prior = other["crossfit_prior_ic"].mean()
        print(
            f"  {'Other':<22}  {len(other):>5}  {cf_post:>+8.4f}  "
            f"{cf_prior:>+9.4f}  {cf_post - cf_prior:>+8.4f}"
        )

    # ── Year-by-year breakdown ──────────────────────────────────────────────
    print("\nYEAR-BY-YEAR CROSS-FIT IC:")
    print(f"  {'Year':<6}  {'N':>5}  {'CF-Post':>8}  {'CF-Prior':>9}  {'Net':>7}")
    print("  " + "-" * 42)
    for year, grp in df.groupby(df["date"].dt.year):
        cf_post  = grp["crossfit_post_ic"].mean()
        cf_prior = grp["crossfit_prior_ic"].mean()
        print(
            f"  {year:<6}  {len(grp):>5}  {cf_post:>+8.4f}  "
            f"{cf_prior:>+9.4f}  {cf_post - cf_prior:>+7.4f}"
        )

    # ── Active units (per-dimension KL) ────────────────────────────────────
    if kl_per_dim_accum:
        kl_matrix = torch.stack(kl_per_dim_accum)      # (n_dates, K)
        mean_kl   = kl_matrix.mean(0).numpy()           # (K,)
        std_kl    = kl_matrix.std(0).numpy()            # (K,)
        active    = int((mean_kl > 0.01).sum())

        print(f"\nACTIVE FACTOR UNITS (K={K}, threshold KL > 0.01):")
        print(f"  Active units: {active} / {K}")
        print(f"  {'Factor':<8}  {'Mean KL':>9}  {'Std KL':>8}  {'Active':>7}")
        print("  " + "-" * 38)
        for k in range(K):
            flag = "  YES" if mean_kl[k] > 0.01 else "   no"
            print(f"  factor_{k:<2}  {mean_kl[k]:>+9.4f}  {std_kl[k]:>8.4f}  {flag}")

        if active < K // 2:
            print(f"\n  [WARNING] Only {active}/{K} factors are active.")
            print("  K may exceed recoverable signal rank — consider reducing num_factors.")

    # ── Save results ────────────────────────────────────────────────────────
    out_dir = ROOT / "results" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"crossfit_ic_{args.split}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nPer-date results saved to: {csv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
