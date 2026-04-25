"""
Portfolio backtest from saved predictions.

Reads results/predictions/predictions.parquet plus any available benchmark
model predictions, applies TopK-Drop strategy, computes extended performance
metrics, prints a comparison table, and saves three figures to results/figures/.

The core logic is exposed as `run_backtest_from_predictions()` so that
scripts/evaluate.py can call it inline (no subprocess needed).

Usage:
    python scripts/backtest.py
    python scripts/backtest.py --benchmark data/processed/benchmark.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from factorvae.evaluation.backtest  import compute_performance_metrics, topk_drop_strategy
from factorvae.evaluation.comparison import (
    build_comparison_table,
    format_for_display,
    load_all_predictions,
    load_benchmark,
    print_comparison,
)
from factorvae.evaluation.metrics import rolling_rank_ic
from factorvae.evaluation.plot_style import (
    PALETTE, TEXT_SECONDARY,
    add_brand_bar, add_footer, add_title, apply_style, finalize_axes, label_lines,
)
from factorvae.evaluation.plot_table import render_comparison_table

ROOT = Path(__file__).resolve().parents[1]


# ── Figure helpers ────────────────────────────────────────────────────────────

def _date_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


COLOR_MAP = {
    "FactorVAE":      PALETTE[0],      # brand red
    "Momentum":       "#003f88",       # deep navy
    "Linear (Ridge)": "#1a6eb5",       # medium blue
    "MLP":            "#5b9fd4",       # light blue
    "GRU":            "#9ec5e8",       # pale blue
}


# ── Core backtest logic (importable by evaluate.py) ───────────────────────────

def run_backtest_from_predictions(
    factorvaepreds: pd.DataFrame,
    config: dict,
    root: Path,
    benchmark_path: "Path | None" = None,
) -> None:
    """
    Full backtest: metrics table + three figures.

    Args:
        factorvaepreds: FactorVAE predictions (already loaded and date-parsed).
        config:         parsed config.yaml dict.
        root:           workspace root for locating benchmark parquets + output dirs.
        benchmark_path: path to benchmark return parquet; falls back to EW market.
    """
    apply_style()

    k   = config["evaluation"]["top_k"]
    n   = config["evaluation"]["drop_n"]
    eta = config["evaluation"]["risk_aversion_eta"]

    if benchmark_path is None:
        benchmark_path = root / "data" / "processed" / "benchmark.parquet"

    benchmark = load_benchmark(benchmark_path, factorvaepreds)

    # ── Load all prediction sources (FactorVAE always present; benchmarks optional)
    all_preds = load_all_predictions(root)
    if "FactorVAE" not in all_preds:
        # Use the passed-in predictions directly (evaluate.py path before file is saved)
        all_preds["FactorVAE"] = factorvaepreds

    # ── Comparison table ──────────────────────────────────────────────────────
    table = build_comparison_table(root, benchmark, k=k, n=n, eta=0.0)

    fig_dir = root / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    csv_path = fig_dir / "comparison_table.csv"
    table.to_csv(csv_path)
    print(f"\n── Comparison: FactorVAE vs Benchmarks {'─' * 20}")
    print_comparison(table)
    print(f"\nTable saved → {csv_path.relative_to(root)}")

    # ── Render three styled comparison table PNGs ─────────────────────────────
    # Add EW Market row to performance table (buy-and-hold baseline)
    _ew_turn  = pd.Series(0.0, index=benchmark.index)
    _ew_perf  = compute_performance_metrics(benchmark, benchmark, turnover=_ew_turn)
    _ew_row   = pd.DataFrame(
        {**{"rank_ic": float("nan"), "rank_icir": float("nan")}, **_ew_perf},
        index=["EW Market"],
    )
    _fv_pos   = (list(table.index).index("FactorVAE") + 1
                 if "FactorVAE" in table.index else len(table))
    perf_table = pd.concat([table.iloc[:_fv_pos], _ew_row, table.iloc[_fv_pos:]])

    formatted      = format_for_display(table)
    formatted_perf = format_for_display(perf_table)

    _PRETTY = {
        "rank_ic":            "Rank IC",
        "rank_icir":          "Rank ICIR",
        "annualized_return":  "Ret. Anual",
        "annualized_excess":  "Retorno Exc.",
        "volatility":         "Volatil.",
        "sharpe":             "Sharpe",
        "information_ratio":  "IR",
        "calmar":             "Calmar",
        "max_drawdown":       "Max DD",
        "hit_rate":           "Hit Rate",
        "avg_turnover":       "Turnover",
    }

    def _sub(cols: list[str], fmt: "pd.DataFrame | None" = None) -> "pd.DataFrame":
        src     = fmt if fmt is not None else formatted
        present = [c for c in cols if c in src.columns]
        return src[present].rename(columns=_PRETTY)

    _IC    = ["rank_ic", "rank_icir"]
    _PERF  = ["annualized_return", "annualized_excess", "volatility",
               "sharpe", "information_ratio", "calmar", "max_drawdown"]
    _STRAT = ["hit_rate", "avg_turnover"]

    render_comparison_table(
        _sub(_IC),
        out_path=fig_dir / "RIC_comparison_ic.png",
        title="Qualidade do sinal preditivo",
        subtitle="Rank IC e Rank ICIR médios · período de teste",
        figsize=(7, 3.5),
    )
    print("Figure saved → results/figures/RIC_comparison_ic.png")

    render_comparison_table(
        _sub(_PERF, formatted_perf),
        out_path=fig_dir / "BKT_comparison_performance.png",
        title="Performance ajustada ao risco — TopK-Drop",
        subtitle=f"k={k} ações, n={n}/dia, taxa 10 bps · período de teste",
        figsize=(11, 5.0),
    )
    print("Figure saved → results/figures/BKT_comparison_performance.png")

    render_comparison_table(
        _sub(_STRAT),
        out_path=fig_dir / "BKT_comparison_strategy.png",
        title="Métricas da estratégia TopK-Drop",
        subtitle="Hit rate e turnover médio · período de teste",
        figsize=(7, 3.5),
    )
    print("Figure saved → results/figures/BKT_comparison_strategy.png")

    # ── Warn if FactorVAE is dominated on Rank IC ─────────────────────────────
    if "rank_ic" in table.columns and "FactorVAE" in table.index:
        fv_ic = table.loc["FactorVAE", "rank_ic"]
        for other in table.index:
            if other != "FactorVAE":
                other_ic = table.loc[other, "rank_ic"]
                try:
                    if float(other_ic) > float(fv_ic):
                        print(
                            f"  WARNING: {other} has higher Rank IC "
                            f"({float(other_ic):.4f}) than FactorVAE ({float(fv_ic):.4f})"
                        )
                except (TypeError, ValueError):
                    pass

    # ── Build per-model portfolio return series ───────────────────────────────
    port_series: dict[str, pd.Series] = {}
    for name, preds in all_preds.items():
        port = topk_drop_strategy(preds, k=k, n=n, eta=0.0).set_index("date")
        port_series[name] = port["portfolio_return"]

    all_dates = sorted({d for r in port_series.values() for d in r.index})
    bm_aligned = benchmark.reindex(all_dates).fillna(0.0)
    label_bm = benchmark.name if hasattr(benchmark, "name") and benchmark.name else "Benchmark"

    # ── Figure 1: Retorno acumulado ───────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(11, 5.5))
    fig1.subplots_adjust(top=0.84, bottom=0.12, left=0.06, right=0.90)

    cum_series: dict[str, pd.Series] = {}
    for name, ret in port_series.items():
        color = COLOR_MAP.get(name, PALETTE[-1])
        cum = (1.0 + ret).cumprod()
        ax1.plot(cum.index, cum.values, color=color)
        cum_series[name] = cum

    cum_bm = (1.0 + bm_aligned).cumprod()
    ax1.plot(cum_bm.index, cum_bm.values, color=TEXT_SECONDARY, linestyle="-", linewidth=0.7)
    cum_series[label_bm] = cum_bm

    label_lines(ax1, cum_series, color_map={**COLOR_MAP, label_bm: TEXT_SECONDARY})
    finalize_axes(ax1, y_right=False)
    _date_axis(ax1)
    ax1.set_ylabel("Retorno acumulado (1 = início)")

    add_brand_bar(fig1)
    add_title(fig1, "Retorno acumulado — estratégia TopK-Drop",
              subtitle=f"k={k} ações, turnover máx. n={n}/dia, taxa 10 bps · universo B3")
    add_footer(fig1, source="Economatica. Cálculos do autor")
    fig1.savefig(fig_dir / "BKT_cumulative_return.png")
    plt.close(fig1)
    print("Figure saved → results/figures/BKT_cumulative_return.png")

    # ── Figure 2: Retorno acumulado em excesso ────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(11, 5.5))
    fig2.subplots_adjust(top=0.84, bottom=0.12, left=0.06, right=0.90)

    excess_series: dict[str, pd.Series] = {}
    for name, ret in port_series.items():
        color = COLOR_MAP.get(name, PALETTE[-1])
        excess = ret - benchmark.reindex(ret.index).fillna(0.0)
        cum_ex = (1.0 + excess).cumprod() - 1.0
        ax2.plot(cum_ex.index, cum_ex.values * 100.0, color=color)
        excess_series[name] = cum_ex * 100.0

    ax2.axhline(0, color=TEXT_SECONDARY, linewidth=0.6, linestyle="--")
    label_lines(ax2, excess_series, color_map=COLOR_MAP)
    finalize_axes(ax2, y_right=False)
    _date_axis(ax2)
    ax2.set_ylabel("Retorno acumulado em excesso (%)")

    add_brand_bar(fig2)
    add_title(fig2, "Retorno acumulado em excesso vs benchmark",
              subtitle=f"TopK-Drop k={k}, n={n} · universo B3")
    add_footer(fig2, source="Economatica. Cálculos do autor")
    fig2.savefig(fig_dir / "BKT_cumulative_excess_return.png")
    plt.close(fig2)
    print("Figure saved → results/figures/BKT_cumulative_excess_return.png")

    # ── Figure 3: Rolling 60-day Rank IC ──────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(11, 4.5))
    fig3.subplots_adjust(top=0.84, bottom=0.12, left=0.06, right=0.90)

    ic_series: dict[str, pd.Series] = {}
    for name, preds in all_preds.items():
        color = COLOR_MAP.get(name, PALETTE[-1])
        r = rolling_rank_ic(preds, window=60)
        ax3.plot(r.index, r.values, color=color)
        ic_series[name] = r

    ax3.axhline(0, color=TEXT_SECONDARY, linewidth=0.6, linestyle="--")
    label_lines(ax3, ic_series, color_map=COLOR_MAP)
    finalize_axes(ax3, y_right=False)
    _date_axis(ax3)
    ax3.set_ylabel("IC de Spearman, média 60 dias")

    add_brand_bar(fig3)
    add_title(fig3, "IC de Spearman — rolling 60 dias",
              subtitle="Correlação cross-sectional entre retorno previsto e realizado")
    add_footer(fig3, source="Economatica. Cálculos do autor")
    fig3.savefig(fig_dir / "RIC_rolling_rank_ic.png")
    plt.close(fig3)
    print("Figure saved → results/figures/RIC_rolling_rank_ic.png")


# ── Standalone entry point ────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run full backtest and comparison.")
    parser.add_argument("--config",      default=str(ROOT / "config.yaml"))
    parser.add_argument(
        "--predictions",
        default=str(ROOT / "results" / "predictions" / "predictions.parquet"),
    )
    parser.add_argument(
        "--benchmark",
        default=str(ROOT / "data" / "processed" / "benchmark.parquet"),
        help=(
            "Parquet with columns [date, return] for the index benchmark. "
            "Falls back to equal-weight market if the file does not exist."
        ),
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    preds = pd.read_parquet(args.predictions)
    preds["date"] = pd.to_datetime(preds["date"])

    run_backtest_from_predictions(
        factorvaepreds=preds,
        config=config,
        root=ROOT,
        benchmark_path=Path(args.benchmark),
    )


if __name__ == "__main__":
    main()
