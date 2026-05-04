"""
Day 0 diagnostic — Posterior IC vs Prior IC on held-out test cross-sections.

For each test date:
  - forward_train(x, y)  → mu_post  (encoder sees contemporaneous y)
  - forward_predict(x)   → mu_prior (encoder NOT called)

Reports per-date IC for both, their means, and the gap.
If posterior IC is not meaningfully higher than prior IC on held-out data,
the "signal exists but predictor can't learn it" framing is unsupported.

Usage:
    python scripts/diagnose_posterior_ic.py
    python scripts/diagnose_posterior_ic.py --checkpoint results/checkpoints/best.ckpt
    python scripts/diagnose_posterior_ic.py --split val   # run on val set instead
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default=str(ROOT / "config.yaml"))
    parser.add_argument("--checkpoint", default=str(ROOT / "results" / "checkpoints" / "best.ckpt"))
    parser.add_argument("--split",      default="test", choices=["val", "test"],
                        help="Dataset split to evaluate on (default: test).")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override config with the checkpoint's own hparams so architecture matches
    import torch as _torch
    _ckpt_data = _torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "hyper_parameters" in _ckpt_data:
        config = _ckpt_data["hyper_parameters"]

    seed_everything(config["training"]["seed"])

    # ── Load model ────────────────────────────────────────────────────────────
    model = FactorVAE(config)
    lm = FactorVAELightning.load_from_checkpoint(args.checkpoint, model=model, config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm.model.to(device)
    lm.model.eval()

    # ── Data ─────────────────────────────────────────────────────────────────
    datamodule = FactorVAEDataModule(config)
    datamodule.setup()
    dataset = datamodule._val if args.split == "val" else datamodule._test

    records = []

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc=f"Diagnosing {args.split} set", unit="date"):
            batch = dataset[idx]
            if len(batch) == 4:
                x, m, y, mask = batch
                m = m.float().to(device)
            else:
                x, y, mask = batch
                m = None

            x = x.float().to(device)
            y_cpu = y.float()

            # ── Prior IC (encoder NOT called — same as production inference) ──
            mu_prior, _ = lm.model.forward_predict(x, m=m)
            prior_ic = compute_rank_ic(y_cpu, mu_prior.cpu())

            # ── Posterior IC (encoder sees contemporaneous y — oracle) ────────
            y_dev = y_cpu.to(device)
            out = lm.model.forward_train(x, y_dev, m=m)
            mu_post = out["mu_y_rec"]   # decoder output from posterior factors → predicted returns
            post_ic = compute_rank_ic(y_cpu, mu_post.cpu())

            # Also compute IC between posterior factor means and y directly
            # (to detect if the encoder's latent z itself correlates with y)
            # This requires projecting mu_post (K,) back — instead use mu_y_rec
            # which is the decoder's return reconstruction from the posterior.

            N = x.shape[0]
            date_label = (
                dataset.trading_dates[idx].strftime("%Y-%m-%d")
                if hasattr(dataset, "trading_dates")
                else str(idx)
            )

            records.append({
                "date":      date_label,
                "n_stocks":  N,
                "prior_ic":  prior_ic,
                "post_ic":   post_ic,
                "gap":       post_ic - prior_ic,
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print("\n" + "=" * 60)
    print(f"POSTERIOR IC LEAKAGE CHECK  --  split={args.split.upper()}")
    print(f"Checkpoint: {Path(args.checkpoint).name}")
    print("=" * 60)

    stats = {
        "":                  ["Prior IC (prod)",  "Post IC (oracle)",  "Gap (post-prior)"],
        "Mean":              [df.prior_ic.mean(),  df.post_ic.mean(),   df.gap.mean()],
        "Std":               [df.prior_ic.std(),   df.post_ic.std(),    df.gap.std()],
        "% Positive":        [(df.prior_ic > 0).mean() * 100,
                              (df.post_ic > 0).mean() * 100,
                              (df.gap > 0).mean() * 100],
    }
    summary = pd.DataFrame(stats).set_index("")
    print(summary.to_string(float_format=lambda x: f"{x:+.4f}" if abs(x) < 10 else f"{x:+.1f}%"))

    print()
    print("Interpretation:")
    mean_gap = df.gap.mean()
    mean_post = df.post_ic.mean()
    mean_prior = df.prior_ic.mean()

    if mean_post < 0.005:
        print("  [X] Posterior IC near zero on held-out data -> encoder finds no exploitable signal")
        print("    The 'signal exists but predictor can't learn it' hypothesis is NOT supported.")
    elif mean_gap < 0.003:
        print("  [~] Posterior IC positive but gap vs prior is negligible -> encoder adds little over prior")
    else:
        print(f"  [OK] Posterior IC ({mean_post:+.4f}) > Prior IC ({mean_prior:+.4f}) on held-out data")
        print(f"    Gap = {mean_gap:+.4f}. Signal exists; teacher-student gap is real.")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = ROOT / "results" / "diagnostics" / f"posterior_ic_{args.split}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nPer-date results saved to: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
