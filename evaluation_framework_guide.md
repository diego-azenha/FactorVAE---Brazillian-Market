# FactorVAE — Expanded Evaluation Framework

Guide to upgrade the repository from "training runs, metrics print" to a full comparative evaluation: benchmark models, proper backtest charts with dates, richer performance metrics, training diagnostics, and logging at session start. Everything below is additive — no existing file is rewritten in spirit, only extended.

---

## 1. What gets added, at a glance

| Area | Before | After |
|------|--------|-------|
| Baselines | None | 2 benchmark models in `benchmarks/` (Momentum, Linear/Ridge) |
| Backtest chart | No dates, mean(y) as benchmark | Real dates on axis, IBOV/IBX index as benchmark, three strategies plotted |
| Backtest metrics | AR, Sharpe, MDD | + Information Ratio, Calmar, avg turnover, hit rate |
| Training logging | metrics.csv only | + printed split summary at start, + loss curves saved as PNG |
| Rolling diagnostics | None | Rolling 60-day Rank IC plot |
| Comparison table | None | Single CSV + stdout table comparing FactorVAE vs benchmarks on all metrics |

---

## 2. New folder structure

```
factorvae-br/
├── benchmarks/                       # NEW top-level folder
│   ├── __init__.py
│   ├── momentum.py                   # Zero-learning signal: ret_20d
│   ├── linear_model.py               # Ridge regression trained the same way
│   └── run_benchmarks.py             # Generates benchmarks/predictions/*.parquet
│
├── benchmarks/predictions/           # NEW: outputs of benchmark runs
│   ├── momentum_predictions.parquet
│   └── linear_predictions.parquet
│
├── scripts/
│   ├── train.py                      # MODIFIED: print split summary
│   ├── evaluate.py                   # unchanged
│   ├── backtest.py                   # MODIFIED: dated chart, index benchmark, more metrics
│   └── plot_training_curves.py       # NEW: post-training diagnostics
│
├── src/factorvae/evaluation/
│   ├── metrics.py                    # MODIFIED: add rolling_rank_ic
│   ├── backtest.py                   # MODIFIED: add info_ratio, calmar, hit_rate
│   └── comparison.py                 # NEW: load + compare FactorVAE vs benchmarks
│
└── results/
    ├── predictions/predictions.parquet
    └── figures/
        ├── cumulative_return.png
        ├── cumulative_excess_return.png
        ├── rolling_rank_ic.png
        ├── training_curves.png
        └── comparison_table.csv
```

---

## 3. Benchmark models

Two benchmarks give the right contrast for the TCC story: a *zero-learning* signal that tests whether the model beats a single feature read off the data, and a *simple trained baseline* that tests whether the nonlinear probabilistic machinery earns its complexity over a linear fit of the same inputs.

### 3.1 Why these two, and not more

The paper compares against 8 baselines (GRU, ALSTM, GAT, Trans, SFM, Linear, CA, FactorVAE-prior). For a TCC the interesting comparison is bounded by *what story you want to tell*, not by matching the paper's baseline count. Two benchmarks suffice when chosen right:

**Momentum** (signal = `ret_20d` from the feature set) is the null hypothesis: if FactorVAE barely beats a single feature read from the data, the whole probabilistic framework is not earning its keep. This is a hard floor — any respectable model must clear it.

**Ridge regression** on the same 20 features is the linear ablation: if FactorVAE barely beats a ridge fit of the same inputs, then the GRU + VAE + attention machinery is not extracting nonlinear structure. This is the interesting ceiling of the comparison.

Adding a third (say, a plain GRU head) would be nice for the TCC but is not strictly necessary — the two above span the "how much does FactorVAE actually buy you" question.

### 3.2 `benchmarks/momentum.py`

Pure signal, no training. For each date in the test set, read the `ret_20d` feature for each valid ticker and use it as `mu_pred`. Set `sigma_pred = 0` (no risk estimate — TDrisk collapses to the same ranking).

```python
"""
Momentum benchmark: mu_pred = ret_20d (a feature already in the dataset).

No training, no parameters. Produces benchmarks/predictions/momentum_predictions.parquet
with the same schema as results/predictions/predictions.parquet so that backtest
and comparison code can consume all prediction sources uniformly.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import yaml

from factorvae.data.dataset import RealDataset

ROOT = Path(__file__).resolve().parents[1]


def generate_predictions(config: dict) -> pd.DataFrame:
    dc = config["data"]
    dataset = RealDataset(
        processed_dir=dc["processed_dir"],
        start_date=dc["test_start"],
        end_date=dc["test_end"],
        sequence_length=dc["sequence_length"],
    )
    # Raw forward returns for y_true
    raw_returns = (
        pd.read_parquet(Path(dc["processed_dir"]) / "returns.parquet")
        .assign(date=lambda df: pd.to_datetime(df["date"]))
        .set_index(["date", "ticker"])["forward_return"]
    )

    # Find the index of ret_20d in feature_cols (hardcode-safe via list lookup)
    ret_20d_idx = dataset.feature_cols.index("ret_20d")

    records = []
    for idx in range(len(dataset)):
        date_ts = dataset.trading_dates[idx]
        tickers = dataset.universe_by_date[date_ts]
        # x is already z-scored per timestep per feature — grab the LAST timestep
        x, _, _ = dataset[idx]           # (N, T, C)
        ret_20d_last = x[:, -1, ret_20d_idx].numpy()  # (N,)

        for i, ticker in enumerate(tickers):
            try:
                y_true = float(raw_returns.loc[(date_ts, ticker)])
            except KeyError:
                y_true = float("nan")
            records.append({
                "date":       date_ts.strftime("%Y-%m-%d"),
                "ticker":     ticker,
                "mu_pred":    float(ret_20d_last[i]),
                "sigma_pred": 0.0,
                "y_true":     y_true,
            })
    return pd.DataFrame(records)
```

### 3.3 `benchmarks/linear_model.py`

Ridge regression on the flattened feature vector of the last timestep. Training uses the same train/val/test split as FactorVAE, so the comparison is fair.

```python
"""
Linear benchmark: Ridge regression on the last timestep's features.

For each training date, stack x[:, -1, :] into (N_s, C) and y into (N_s,),
concatenate across all train dates, fit Ridge(alpha=1.0). At test time,
apply the same linear model per date.

Uses ONLY the last timestep of x, which is the standard convention for
linear factor models (no temporal structure assumed).
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge

from factorvae.data.dataset import RealDataset

ROOT = Path(__file__).resolve().parents[1]


def _stack_dataset(dataset: RealDataset) -> tuple[np.ndarray, np.ndarray, list, list]:
    """Flatten a RealDataset into (X, y, dates, tickers) arrays."""
    X_all, y_all, date_labels, ticker_labels = [], [], [], []
    for idx in range(len(dataset)):
        x, y, _ = dataset[idx]           # (N, T, C), (N,)
        date_ts = dataset.trading_dates[idx]
        tickers = dataset.universe_by_date[date_ts]
        X_last = x[:, -1, :].numpy()     # (N, C) — last-timestep features
        X_all.append(X_last)
        y_all.append(y.numpy())
        date_labels.extend([date_ts] * len(tickers))
        ticker_labels.extend(tickers)
    return (
        np.concatenate(X_all, axis=0),
        np.concatenate(y_all, axis=0),
        date_labels,
        ticker_labels,
    )


def train_and_predict(config: dict, alpha: float = 1.0) -> pd.DataFrame:
    dc = config["data"]
    train_ds = RealDataset(dc["processed_dir"], dc["train_start"], dc["train_end"], dc["sequence_length"])
    test_ds  = RealDataset(dc["processed_dir"], dc["test_start"],  dc["test_end"],  dc["sequence_length"])

    X_train, y_train, _, _ = _stack_dataset(train_ds)
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    # Predict on test
    X_test, _, dates, tickers = _stack_dataset(test_ds)
    mu_pred = model.predict(X_test)

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
```

### 3.4 `benchmarks/run_benchmarks.py`

Single entry point. Runs both models, writes both parquets.

```python
"""
Generate prediction files for all benchmark models.

Run once. Output goes to benchmarks/predictions/ and is consumed by
scripts/backtest.py and scripts/evaluate.py for comparison.

Usage:
    python benchmarks/run_benchmarks.py
"""

from __future__ import annotations
from pathlib import Path
import yaml

from benchmarks.momentum import generate_predictions as momentum_predict
from benchmarks.linear_model import train_and_predict as linear_predict

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)

    out_dir = ROOT / "benchmarks" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running Momentum benchmark...")
    mom = momentum_predict(config)
    mom.to_parquet(out_dir / "momentum_predictions.parquet", index=False)
    print(f"  saved {len(mom)} rows")

    print("Running Ridge Linear benchmark...")
    lin = linear_predict(config, alpha=1.0)
    lin.to_parquet(out_dir / "linear_predictions.parquet", index=False)
    print(f"  saved {len(lin)} rows")


if __name__ == "__main__":
    main()
```

### 3.5 Running

Run once, immediately after `build_features.py` completes. Predictions are deterministic (no random init), so they never need to be regenerated unless the processed data changes.

```
python benchmarks/run_benchmarks.py
```

---

## 4. Richer backtest metrics

Extend `src/factorvae/evaluation/backtest.py` with four metrics that are standard in quant practice and match the TCC's level.

```python
def compute_performance_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    turnover: pd.Series | None = None,
) -> dict:
    """
    Extended performance metrics.

    Args:
        portfolio_returns: daily net returns of the strategy
        benchmark_returns: daily returns of the reference (e.g. IBOV equal-weight)
        turnover:          daily turnover series from topk_drop_strategy (optional)

    Returns dict with:
        annualized_return    : strategy AR (absolute)
        annualized_excess    : AR over benchmark
        volatility           : annualized std of strategy returns
        sharpe               : SR on excess returns (paper convention)
        information_ratio    : excess AR / tracking error (= sharpe on excess; reported
                               separately to match industry convention)
        max_drawdown         : max peak-to-trough on cumulative excess returns
        calmar               : annualized_excess / max_drawdown
        hit_rate             : fraction of days where excess return > 0
        avg_turnover         : mean daily turnover (only if turnover passed)
    """
    import numpy as np

    bench = benchmark_returns.reindex(portfolio_returns.index).fillna(0.0).values
    port = portfolio_returns.values
    excess = port - bench

    days = 252
    ann_return = float(np.mean(port) * days)
    ann_excess = float(np.mean(excess) * days)
    vol = float(np.std(port, ddof=1) * np.sqrt(days))

    excess_vol = float(np.std(excess, ddof=1) * np.sqrt(days))
    sharpe = ann_excess / excess_vol if excess_vol > 1e-9 else 0.0
    info_ratio = sharpe  # same under this convention; documented for clarity

    cum_excess = np.cumprod(1.0 + excess)
    running_max = np.maximum.accumulate(cum_excess)
    drawdown = (running_max - cum_excess) / running_max
    mdd = float(drawdown.max()) if len(drawdown) > 0 else 0.0

    calmar = ann_excess / mdd if mdd > 1e-9 else 0.0
    hit_rate = float(np.mean(excess > 0))

    out = {
        "annualized_return":  ann_return,
        "annualized_excess":  ann_excess,
        "volatility":         vol,
        "sharpe":             sharpe,
        "information_ratio":  info_ratio,
        "max_drawdown":       mdd,
        "calmar":             calmar,
        "hit_rate":           hit_rate,
    }
    if turnover is not None:
        out["avg_turnover"] = float(turnover.reindex(portfolio_returns.index).mean())
    return out
```

Implementation notes:

The `information_ratio` and `sharpe` are numerically identical under this definition. They are reported separately because quant reports typically list both and readers expect to see them; the distinction matters in presentations even when the formula collapses.

`max_drawdown` is computed on cumulative *excess* returns, matching the paper's Table 3 footnote that AR/SR/MDD are measured on excess returns against the benchmark.

`avg_turnover` documents strategy aggressiveness — important because TopK-Drop with $n=5$, $k=50$ yields a maximum turnover of $10\%$/day, which compounds to a lot of fee drag over a 5-year test period.

---

## 5. Benchmark index as real comparison

The current `backtest.py` uses `predictions.groupby("date")["y_true"].mean()` as benchmark — that is an *equal-weight cross-section average* of whatever tickers happened to be in the universe each day, not a real index. For the TCC, use the actual IBOV or IBX daily return series.

### 5.1 Where the index data comes from

Options, in order of preference:

1. **Use an index series already in `data/raw/`** if prices for IBOV/IBX are included. Most brokers provide this, and it is the cleanest reference.
2. **Compute an equal-weight market return from the full universe** (what the code does now, but rename it honestly to `equal_weight_market`).
3. **Use a cap-weighted approximation** if market-cap data is available.

For the TCC, either option 1 or 3 is publishable. Option 2 is fine as a fallback but should be labeled "equal-weight market benchmark", not "IBOV".

### 5.2 `scripts/backtest.py` modification

Add a command-line argument to point to the benchmark parquet, with a sensible default:

```python
parser.add_argument(
    "--benchmark",
    default=str(ROOT / "data" / "processed" / "benchmark.parquet"),
    help="Parquet with columns [date, return] for the benchmark index",
)
```

Load with fallback to equal-weight if the file is absent:

```python
def load_benchmark(path: Path, predictions: pd.DataFrame) -> pd.Series:
    if path.exists():
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")["return"]
    # Fallback: equal-weight cross-section
    predictions = predictions.copy()
    predictions["date"] = pd.to_datetime(predictions["date"])
    return predictions.groupby("date")["y_true"].mean().rename("benchmark")
```

For the TCC, create `data/processed/benchmark.parquet` with IBOV daily returns once, then every backtest compares against the same reference.

---

## 6. Charts with dates

Three fixes to the figures, all in `scripts/backtest.py`:

### 6.1 Real dates on the x-axis

Current code plots `cum_mu.values` which forces matplotlib to use integer positional indices. Plot against the DatetimeIndex instead:

```python
cum_mu = (1.0 + port_mu).cumprod()
ax.plot(cum_mu.index, cum_mu.values, label="FactorVAE (TopK-Drop)")
```

Set up date formatting:

```python
import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate()
```

### 6.2 Second chart: cumulative excess return

The paper shows both absolute and excess curves (Figure 6a and 6b). Add the excess curve as a second figure:

```python
excess_mu = port_mu - benchmark.reindex(port_mu.index).fillna(0)
cum_excess_mu = (1.0 + excess_mu).cumprod() - 1.0  # starts at 0 for excess
ax.plot(cum_excess_mu.index, cum_excess_mu.values * 100, label="FactorVAE")
ax.set_ylabel("Cumulative excess return (%)")
```

The `- 1.0` and `* 100` conventions convert the compounded excess into percentage points, matching Table 3 and Figure 6a.

### 6.3 Include benchmark and baselines in the same chart

```python
# Load all prediction sources
sources = {
    "FactorVAE":        ROOT / "results" / "predictions" / "predictions.parquet",
    "Momentum":         ROOT / "benchmarks" / "predictions" / "momentum_predictions.parquet",
    "Linear (Ridge)":   ROOT / "benchmarks" / "predictions" / "linear_predictions.parquet",
}

results = {}
for name, path in sources.items():
    if not path.exists():
        continue
    preds = pd.read_parquet(path)
    preds["date"] = pd.to_datetime(preds["date"])
    port = topk_drop_strategy(preds, k=k, n=n, eta=0.0).set_index("date")
    results[name] = port["portfolio_return"]
```

Then plot all three lines plus the benchmark index on both charts.

---

## 7. Rolling Rank IC diagnostic

One of the most useful visualizations for a probabilistic factor model paper is how the signal quality evolves over the test window. A stable Rank IC around some positive mean is a much stronger result than a mean Rank IC achieved by a few lucky weeks.

Add to `src/factorvae/evaluation/metrics.py`:

```python
def rolling_rank_ic(
    predictions: pd.DataFrame,
    window: int = 60,
) -> pd.Series:
    """
    Rolling mean of per-date Rank IC over a window of trading days.

    Args:
        predictions: DataFrame with [date, ticker, mu_pred, y_true]
        window:      rolling window in trading days (60 ≈ 3 months)

    Returns:
        pd.Series indexed by date with rolling Rank IC.
    """
    import torch
    ics = []
    dates = []
    predictions = predictions.copy()
    predictions["date"] = pd.to_datetime(predictions["date"])
    for date, grp in predictions.groupby("date"):
        y_true = torch.tensor(grp["y_true"].values, dtype=torch.float32)
        mu = torch.tensor(grp["mu_pred"].values, dtype=torch.float32)
        ics.append(compute_rank_ic(y_true, mu))
        dates.append(date)
    series = pd.Series(ics, index=dates).sort_index()
    return series.rolling(window, min_periods=window // 2).mean()
```

Plot in a new section of `scripts/backtest.py`:

```python
for name, preds in all_preds.items():
    r = rolling_rank_ic(preds, window=60)
    ax.plot(r.index, r.values, label=name)
ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
ax.set_title("60-day rolling Rank IC")
fig.savefig(fig_dir / "rolling_rank_ic.png", dpi=150, bbox_inches="tight")
```

This plot tells the reader at a glance whether FactorVAE's advantage is concentrated in specific regimes or distributed across the sample. Both are publishable stories, but they are different stories.

---

## 8. Training diagnostics

### 8.1 Print split summary at training start

In `scripts/train.py`, right after `datamodule.setup()`:

```python
datamodule.setup()

# Informative split banner — prints regardless of dataset type
def _summarize(name: str, ds) -> None:
    if hasattr(ds, "trading_dates") and len(ds.trading_dates) > 0:
        first = ds.trading_dates[0].strftime("%Y-%m-%d")
        last  = ds.trading_dates[-1].strftime("%Y-%m-%d")
        n_dates = len(ds.trading_dates)
        mean_N = (
            sum(len(v) for v in ds.universe_by_date.values()) / n_dates
        )
        print(f"  {name:6s}: {n_dates:4d} dates  [{first} → {last}]  mean N_s = {mean_N:.1f}")
    else:
        print(f"  {name:6s}: {len(ds):4d} synthetic samples")

print("\n── Dataset splits ─────────────────────────────────────────")
_summarize("Train", datamodule._train)
_summarize("Val",   datamodule._val)
_summarize("Test",  datamodule._test)
print("───────────────────────────────────────────────────────────\n")
```

Called before `trainer.fit(...)` so it appears in the log before the Lightning progress bar takes over.

### 8.2 Training curves plot

Lightning already writes `lightning_logs/version_X/metrics.csv` with `train_loss`, `train_loss_recon`, `train_loss_kl`, and `val_rank_ic` per epoch. A new standalone script reads the latest version folder and produces a 2x2 panel.

New file: `scripts/plot_training_curves.py`

```python
"""
Post-training diagnostic plots from Lightning's CSV logger.

Reads lightning_logs/version_*/metrics.csv (most recent), plots:
  - train_loss, train_loss_recon, train_loss_kl over epochs
  - val_rank_ic over epochs

Usage:
    python scripts/plot_training_curves.py
    python scripts/plot_training_curves.py --version 7
"""

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "lightning_logs"


def latest_version(logs: Path) -> Path:
    versions = sorted([p for p in logs.glob("version_*") if p.is_dir()],
                      key=lambda p: int(p.name.split("_")[1]))
    if not versions:
        raise FileNotFoundError(f"No lightning_logs/version_* found in {logs}")
    return versions[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=None,
                        help="specific version number; defaults to latest")
    args = parser.parse_args()

    version_dir = LOGS / f"version_{args.version}" if args.version is not None else latest_version(LOGS)
    metrics = pd.read_csv(version_dir / "metrics.csv")

    # Collapse per-epoch values (CSV has separate rows for train/val per epoch)
    train = metrics.dropna(subset=["train_loss"]).set_index("epoch")
    val   = metrics.dropna(subset=["val_rank_ic"]).set_index("epoch")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(train.index, train["train_loss"], "o-")
    axes[0, 0].set_title("Total training loss")
    axes[0, 0].set_xlabel("Epoch")

    axes[0, 1].plot(train.index, train["train_loss_recon"], "o-", color="tab:blue")
    axes[0, 1].set_title("Reconstruction loss (NLL)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].axhline(
        0.5 * (1 + 1.8379),  # 0.5 * (log(2*pi) + 1) ≈ 1.419 — NLL floor for N(0,1)
        color="grey", linestyle="--", linewidth=0.8, label="N(0,1) baseline",
    )
    axes[0, 1].legend()

    axes[1, 0].plot(train.index, train["train_loss_kl"], "o-", color="tab:orange")
    axes[1, 0].set_title("KL divergence loss")
    axes[1, 0].set_xlabel("Epoch")

    axes[1, 1].plot(val.index, val["val_rank_ic"], "o-", color="tab:green")
    axes[1, 1].axhline(0, color="grey", linestyle="--", linewidth=0.5)
    axes[1, 1].set_title("Validation Rank IC")
    axes[1, 1].set_xlabel("Epoch")

    fig.suptitle(f"Training diagnostics ({version_dir.name})")
    fig.tight_layout()

    out = ROOT / "results" / "figures" / "training_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
```

The reference line on the reconstruction panel (the NLL of a unit-variance Gaussian at the mean) is diagnostic: if `train_loss_recon` stays above or near that line, the model is not learning more than the marginal.

---

## 9. Comparison framework

A single module that, given a set of prediction parquets, produces the full comparative table. Used by both `evaluate.py` (for Rank IC) and `backtest.py` (for AR/Sharpe/etc).

New file: `src/factorvae/evaluation/comparison.py`

```python
"""
Compare FactorVAE against benchmark models on Rank IC, Rank ICIR,
and portfolio metrics. Produces a CSV + printed table.
"""

from __future__ import annotations
from pathlib import Path

import pandas as pd
import torch

from factorvae.evaluation.backtest import compute_performance_metrics, topk_drop_strategy
from factorvae.evaluation.metrics import compute_rank_ic, compute_rank_icir


def load_all_predictions(root: Path) -> dict[str, pd.DataFrame]:
    sources = {
        "FactorVAE":       root / "results" / "predictions" / "predictions.parquet",
        "Momentum":        root / "benchmarks" / "predictions" / "momentum_predictions.parquet",
        "Linear (Ridge)":  root / "benchmarks" / "predictions" / "linear_predictions.parquet",
    }
    out: dict[str, pd.DataFrame] = {}
    for name, path in sources.items():
        if path.exists():
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["date"])
            out[name] = df
    return out


def compute_ic_summary(predictions: pd.DataFrame) -> dict:
    """Rank IC and Rank ICIR aggregated over the test period."""
    ics = []
    for _, grp in predictions.groupby("date"):
        y_true = torch.tensor(grp["y_true"].dropna().values, dtype=torch.float32)
        mu     = torch.tensor(grp.loc[grp["y_true"].notna(), "mu_pred"].values, dtype=torch.float32)
        if len(y_true) < 5:
            continue
        ics.append(compute_rank_ic(y_true, mu))
    return {"rank_ic": sum(ics) / len(ics), "rank_icir": compute_rank_icir(ics)}


def build_comparison_table(
    root: Path,
    benchmark_returns: pd.Series,
    k: int = 50,
    n: int = 5,
    eta: float = 1.0,
) -> pd.DataFrame:
    preds_by_model = load_all_predictions(root)
    rows = []
    for name, preds in preds_by_model.items():
        ic = compute_ic_summary(preds)
        port = topk_drop_strategy(preds, k=k, n=n, eta=0.0)
        port_ret = port.set_index("date")["portfolio_return"]
        turn     = port.set_index("date")["turnover"]
        perf = compute_performance_metrics(port_ret, benchmark_returns, turnover=turn)
        rows.append({"model": name, **ic, **perf})
    return pd.DataFrame(rows).set_index("model")
```

And the print helper:

```python
def print_comparison(df: pd.DataFrame) -> None:
    pct_cols  = ["annualized_return", "annualized_excess", "volatility",
                 "max_drawdown", "hit_rate", "avg_turnover"]
    flt_cols  = ["rank_ic", "rank_icir", "sharpe", "information_ratio", "calmar"]
    formatted = df.copy()
    for c in pct_cols:
        if c in formatted.columns:
            formatted[c] = formatted[c].map(lambda v: f"{v*100:+.2f}%")
    for c in flt_cols:
        if c in formatted.columns:
            formatted[c] = formatted[c].map(lambda v: f"{v:+.3f}")
    print(formatted.to_string())
```

Call from `scripts/backtest.py`:

```python
table = build_comparison_table(ROOT, benchmark, k=k, n=n, eta=eta)
table.to_csv(ROOT / "results" / "figures" / "comparison_table.csv")
print("\n── Comparison: FactorVAE vs Benchmarks ──")
print_comparison(table)
```

---

## 10. Modified `scripts/backtest.py` — full flow

The final `backtest.py` brings everything together. High-level shape:

```python
def main() -> None:
    # 1. Load config and arguments
    # 2. Load predictions (FactorVAE + benchmarks, dict name → df)
    # 3. Load benchmark return series
    # 4. For each source, run TopK-Drop (eta=0) and TDrisk (eta>0)
    # 5. Compute performance metrics via compute_performance_metrics
    # 6. Build and print comparison table
    # 7. Plot three figures with real dates:
    #    - cumulative absolute return
    #    - cumulative excess return (FactorVAE + benchmarks vs index)
    #    - rolling 60-day Rank IC
    # 8. Save table CSV and figures to results/figures/
```

Key behavioral differences from current implementation:
- Always produces `comparison_table.csv` regardless of whether benchmarks are present (falls back gracefully if `benchmarks/predictions/` is empty).
- Prints a summary line warning when FactorVAE underperforms any benchmark on Rank IC — this avoids accidentally skipping over a negative result.
- Figures always have proper date axes, model labels, and units.

---

## 11. Execution order

After all files are in place, the pipeline is:

```
# 0. Data (only when raw data changes)
python scripts/build_features.py

# 1. Benchmarks (once; deterministic)
python benchmarks/run_benchmarks.py

# 2. Train FactorVAE
python scripts/train.py

# 3. Evaluate and generate predictions parquet
python scripts/evaluate.py

# 4. Full backtest and comparison
python scripts/backtest.py

# 5. Training diagnostics (reads latest lightning_logs version)
python scripts/plot_training_curves.py
```

Steps 1–3 are independent of step 5; step 4 depends on 1, 2, and 3 all having completed.

---

## 12. What gets reported in the TCC

With this framework in place, the evaluation section of the thesis has:

A table comparing FactorVAE, Momentum, and Ridge on eight numbers: Rank IC, Rank ICIR, annualized return, annualized excess, Sharpe, information ratio, max drawdown, Calmar, hit rate, average turnover. Nine columns, three rows. Clean and honest.

Three figures: cumulative absolute return, cumulative excess return, rolling Rank IC. All three have real dates and clear legends. The rolling Rank IC chart, in particular, is the one that distinguishes a paper that just cherry-picked a period from one that shows regime-robustness.

A training diagnostics figure showing loss decomposition and validation IC per epoch. This is for the methodology section — it documents that the model actually converged and does not hide a failed training run.

A single CSV (`results/figures/comparison_table.csv`) that ships with the thesis supplementary materials, containing the exact numbers reported in the tables.

---

## 13. What is deliberately not included

Two things the paper does that we skip, with rationale:

The **robustness experiment (missing stocks)** works well in China A-shares with ~3500 tickers because removing 50-200 leaves thousands. With ~130 tickers/day in the Brazilian universe, removing 50 leaves too few for a meaningful cross-section — Rank IC becomes unstable on the reduced set. Running this anyway would produce noisy numbers that don't strengthen the thesis.

The **five-seed variance reporting** in Tables 1–3 requires training the model five times. At ~3.5 minutes per epoch × 50 epochs × 5 seeds, that's ≈15 hours of training for one configuration. Acceptable for a journal submission, probably not for a TCC on a single-GPU workstation. Report the single-seed numbers and note the caveat; if time permits, run 3 seeds at the end and report mean ± std in the final table.
