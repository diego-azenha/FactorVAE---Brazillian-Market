"""
Evaluate FactorVAE on the test set.

Loads the best checkpoint, runs forward_predict on every test date,
saves results/predictions/predictions.parquet, then automatically runs:
  - Robustness test (fractional stock-drop, 5 trials)
  - Full backtest + comparison table vs benchmark models
  - Three figures saved to results/figures/

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint results/checkpoints/best.ckpt
    python scripts/evaluate.py --synthetic
    python scripts/evaluate.py --skip-backtest
    python scripts/evaluate.py --skip-robustness
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `scripts.backtest` is importable
# when this script is invoked as `python scripts/evaluate.py` (Python adds
# the scripts/ directory itself, not its parent, to sys.path in that case).
_ROOT_EARLY = Path(__file__).resolve().parents[1]
if str(_ROOT_EARLY) not in sys.path:
    sys.path.insert(0, str(_ROOT_EARLY))

import pandas as pd
import torch
import yaml
from tqdm import tqdm

from factorvae.data.datamodule import FactorVAEDataModule
from factorvae.models.factorvae import FactorVAE
from factorvae.training.lightning_module import FactorVAELightning
from factorvae.evaluation.metrics import compute_rank_ic, compute_rank_icir
from factorvae.evaluation.robustness import robustness_drop_test
from factorvae.utils.seeding import seed_everything

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default=str(ROOT / "config.yaml"))
    parser.add_argument("--checkpoint",  default=str(ROOT / "results" / "checkpoints" / "best.ckpt"))
    parser.add_argument("--synthetic",   action="store_true")
    parser.add_argument(
        "--benchmark",
        default=str(ROOT / "data" / "processed" / "benchmark.parquet"),
        help="Parquet with [date, return] for the index benchmark (falls back to EW market).",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip backtest and comparison table (inference + IC only).",
    )
    parser.add_argument(
        "--skip-robustness",
        action="store_true",
        help="Skip the robustness drop test.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed_everything(config["training"]["seed"])

    # Load raw forward returns for ground-truth y_true (NOT z-scored)
    # RealDataset returns z-scored y which is correct for training/IC, but
    # backtest needs actual returns in economic units.
    use_synthetic = args.synthetic
    if not use_synthetic:
        returns_path = Path(config["data"]["processed_dir"]) / "returns.parquet"
        raw_returns: pd.Series = (
            pd.read_parquet(returns_path)
            .assign(date=lambda df: pd.to_datetime(df["date"]))
            .set_index(["date", "ticker"])["forward_return"]
        )
    else:
        raw_returns = None

    # Load model
    model = FactorVAE(config)
    lm = FactorVAELightning.load_from_checkpoint(args.checkpoint, model=model, config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm.model.to(device)
    lm.model.eval()

    # Data
    datamodule = FactorVAEDataModule(config, use_synthetic=args.synthetic)
    datamodule.setup()

    records = []
    rank_ics = []

    test_dataset = datamodule._test

    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Evaluating test set", unit="date"):
            x, y, mask = test_dataset[idx]
            x = x.float().to(device)
            y = y.float()   # z-scored — used only for Rank IC (rank-invariant)
            mu_pred, sigma_pred = lm.model.forward_predict(x)
            mu_pred = mu_pred.cpu()
            sigma_pred = sigma_pred.cpu()

            N = x.shape[0]

            if use_synthetic:
                date_label = idx
                ticker_labels = list(range(N))
            else:
                date_ts = test_dataset.trading_dates[idx]
                date_label = date_ts.strftime("%Y-%m-%d")
                ticker_labels = test_dataset.universe_by_date[date_ts]

            for i in range(N):
                ticker = ticker_labels[i]
                # y_true: raw forward return in economic units for backtest.
                # Fall back to NaN if the (date, ticker) key is absent.
                if raw_returns is not None:
                    try:
                        y_true_val = float(raw_returns.loc[(date_ts, ticker)])
                    except KeyError:
                        y_true_val = float("nan")
                else:
                    y_true_val = y[i].item()

                records.append({
                    "date":       date_label,
                    "ticker":     ticker,
                    "mu_pred":    mu_pred[i].item(),
                    "sigma_pred": sigma_pred[i].item(),
                    "y_true":     y_true_val,
                })

            rank_ics.append(compute_rank_ic(y, mu_pred))

    out_df = pd.DataFrame(records)
    out_path = ROOT / "results" / "predictions" / "predictions.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"Saved {len(out_df)} rows to {out_path}")

    print(f"Test Rank IC:   {sum(rank_ics)/len(rank_ics):.4f}")
    print(f"Test Rank ICIR: {compute_rank_icir(rank_ics):.4f}")

    # ── Robustness test ───────────────────────────────────────────────────────
    if not args.synthetic and not args.skip_robustness:
        print("\n── Robustness test (15% stock drop, 5 trials) ──────────────────────")
        out_df_parsed = out_df.copy()
        out_df_parsed["date"] = pd.to_datetime(out_df_parsed["date"])
        rob = robustness_drop_test(out_df_parsed, drop_frac=0.15, n_trials=5)
        print(f"  Full-universe Rank IC : {rob['rank_ic_full']:+.4f}")
        print(f"  Drop-{rob['drop_frac']*100:.0f}% mean Rank IC : {rob['rank_ic_mean']:+.4f} "
              f"± {rob['rank_ic_std']:.4f}  (n_trials={rob['n_trials']})")
        print(f"  Avg stocks per date   : {rob['avg_n_full']:.1f} full → "
              f"{rob['avg_n_dropped']:.1f} after drop")
        print("────────────────────────────────────────────────────────────────────")

    # ── Backtest + comparison ─────────────────────────────────────────────────
    if not args.synthetic and not args.skip_backtest:
        print("\n── Running backtest ─────────────────────────────────────────────────")
        from scripts.backtest import run_backtest_from_predictions
        out_df_parsed = out_df.copy()
        out_df_parsed["date"] = pd.to_datetime(out_df_parsed["date"])
        run_backtest_from_predictions(
            factorvaepreds=out_df_parsed,
            config=config,
            root=ROOT,
            benchmark_path=Path(args.benchmark),
        )


if __name__ == "__main__":
    main()
