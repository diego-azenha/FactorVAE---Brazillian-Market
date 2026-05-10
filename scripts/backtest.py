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
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yaml

from factorvae.evaluation.backtest  import compute_performance_metrics, topk_drop_strategy
from factorvae.evaluation.comparison import (
    build_comparison_table,
    compute_ic_summary,
    format_for_display,
    load_all_predictions,
    load_benchmark,
    print_comparison,
)
from factorvae.evaluation.metrics import rolling_rank_ic
from factorvae.evaluation.plot_style import (
    PALETTE, TEXT_SECONDARY,
    apply_style, finalize_axes,
)
from factorvae.evaluation.plot_table import render_comparison_table

ROOT = Path(__file__).resolve().parents[1]


# ── Figure helpers ────────────────────────────────────────────────────────────

def _date_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


COLOR_MAP = {
    "FactorVAE":           "#E8192E",  # bright red
    "FactorVAE (TDrisk)": "#8B0000",  # dark burgundy — risk-adjusted variant
    "GRU":                "#003f88",  # deep navy   — temporal baseline
    "IPCA":               "#1a6eb5",  # medium blue — linear factor model
    "CA":                 "#5b9fd4",  # light blue  — non-linear factor model
}


# ── Core backtest logic (importable by evaluate.py) ───────────────────────────

def run_backtest_from_predictions(
    factorvaepreds: pd.DataFrame,
    config: dict,
    root: Path,
    benchmark_path: "Path | None" = None,
    out_dir: "Path | None" = None,
) -> None:
    """
    Full backtest: metrics table + three figures.

    Args:
        factorvaepreds: FactorVAE predictions (already loaded and date-parsed).
        config:         parsed config.yaml dict.
        root:           workspace root for locating benchmark parquets + output dirs.
        benchmark_path: path to benchmark return parquet; falls back to EW market.
        out_dir:        directory for all outputs (figures, CSV). Defaults to
                        results/figures/ for backward compatibility.
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
    rf = config.get("evaluation", {}).get("risk_free_rate", 0.10)
    table = build_comparison_table(root, benchmark, k=k, n=n, eta=0.0, risk_free_rate=rf)

    # ── Add TDrisk row for FactorVAE when eta > 0 (Experiment 3) ─────────────
    if eta > 0.0 and "FactorVAE" in all_preds:
        _fv_preds    = all_preds["FactorVAE"]
        _port_td     = topk_drop_strategy(_fv_preds, k=k, n=n, eta=eta)
        _port_td_ret = _port_td.set_index("date")["portfolio_return"]
        _turn_td     = _port_td.set_index("date")["turnover"]
        _perf_td     = compute_performance_metrics(
            _port_td_ret, benchmark, turnover=_turn_td, risk_free_rate=rf
        )
        _tdrisk_row = pd.DataFrame(
            {**{"rank_ic": float("nan"), "rank_icir": float("nan")}, **_perf_td},
            index=["FactorVAE (TDrisk)"],
        )
        _fv_idx = list(table.index).index("FactorVAE") + 1 if "FactorVAE" in table.index else len(table)
        table = pd.concat([table.iloc[:_fv_idx], _tdrisk_row, table.iloc[_fv_idx:]])

    fig_dir = out_dir if out_dir is not None else root / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    csv_path = fig_dir / "comparison_table.csv"
    table.to_csv(csv_path)
    print(f"\n── Comparison: FactorVAE vs Benchmarks {'─' * 20}")
    print_comparison(table)
    print(f"\nTable saved → {csv_path.relative_to(root)}")

    # ── Render three styled comparison table PNGs ─────────────────────────────
    # Add benchmark row to performance table (buy-and-hold baseline).
    # Restrict benchmark to the test period (dates present in the predictions)
    # so its cumulative return is comparable to the model portfolios.
    _test_dates   = sorted(factorvaepreds["date"].unique())
    _bm_test      = benchmark.reindex(_test_dates).fillna(0.0)
    _ew_turn      = pd.Series(0.0, index=_bm_test.index)
    _ew_perf      = compute_performance_metrics(_bm_test, _bm_test, turnover=_ew_turn,
                                               risk_free_rate=rf)
    _ew_row   = pd.DataFrame(
        {**{"rank_ic": float("nan"), "rank_icir": float("nan")}, **_ew_perf},
        index=[benchmark.name],
    )
    # Place EW Market after all FactorVAE variants (including TDrisk)
    _fv_names = [n for n in table.index if n.startswith("FactorVAE")]
    _fv_pos   = (max(list(table.index).index(n) for n in _fv_names) + 1
                 if _fv_names else len(table))
    perf_table = pd.concat([table.iloc[:_fv_pos], _ew_row, table.iloc[_fv_pos:]])

    formatted      = format_for_display(table)
    formatted_perf = format_for_display(perf_table)

    _PRETTY = {
        "rank_ic":            "Rank IC",
        "rank_icir":          "Rank ICIR",
        "annualized_return":  "CAGR",
        "annualized_excess":  "Retorno Exc.",
        "volatility":         "Volatil.",
        "sharpe":             "Sharpe",
        "information_ratio":  "IR",
        "cumulative_return":  "Ret. Acum.",
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
    _PERF  = ["annualized_return", "cumulative_return", "volatility",
               "sharpe", "information_ratio", "max_drawdown"]
    _STRAT = ["hit_rate", "avg_turnover"]

    # Rank IC: exclude TDrisk (same signal as FactorVAE, only portfolio construction differs)
    ic_table = _sub(_IC)
    ic_table = ic_table[~ic_table.index.str.contains("TDrisk", na=False)]
    render_comparison_table(
        ic_table,
        out_path=fig_dir / "RIC_comparison_ic.png",
        title="Qualidade do sinal preditivo",
        subtitle="Rank IC e Rank ICIR médios · período de teste",
        figsize=(7, 3.5),
    )
    print("Figure saved → results/figures/RIC_comparison_ic.png")

    _strategy_label = "TopK-Drop / TDrisk" if eta > 0.0 else "TopK-Drop"
    _eta_note       = f" · η={eta:.1f} (TDrisk)" if eta > 0.0 else ""
    render_comparison_table(
        _sub(_PERF, formatted_perf),
        out_path=fig_dir / "BKT_comparison_performance.png",
        title=f"Performance ajustada ao risco — {_strategy_label}",
        subtitle=f"k={k} ações, n={n}/dia, taxa 25 bps{_eta_note} · período de teste",
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

    # TDrisk: FactorVAE with risk-adjusted scoring (Experiment 3)
    if eta > 0.0 and "FactorVAE" in all_preds:
        _port_td_s = topk_drop_strategy(all_preds["FactorVAE"], k=k, n=n, eta=eta).set_index("date")
        port_series["FactorVAE (TDrisk)"] = _port_td_s["portfolio_return"]

    all_dates = sorted({d for r in port_series.values() for d in r.index})
    bm_aligned = benchmark.reindex(all_dates).fillna(0.0)
    label_bm = benchmark.name if hasattr(benchmark, "name") and benchmark.name else "Benchmark"

    # ── Figure 1: Retorno acumulado ───────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(11, 5.5))
    fig1.subplots_adjust(top=0.87, bottom=0.12, left=0.08, right=0.97)

    cum_series: dict[str, pd.Series] = {}
    for name, ret in port_series.items():
        color = COLOR_MAP.get(name, PALETTE[-1])
        wealth = (1.0 + ret.fillna(0.0)).cumprod()
        ax1.plot(wealth.index, wealth.values, color=color, label=name)
        cum_series[name] = wealth

    bm_wealth = (1.0 + bm_aligned.fillna(0.0)).cumprod()
    ax1.plot(bm_wealth.index, bm_wealth.values, color=TEXT_SECONDARY,
             linestyle="-", linewidth=0.7, label=label_bm)
    cum_series[label_bm] = bm_wealth

    ax1.set_yscale("log")
    ax1.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=[1.0, 2.0, 3.0, 5.0]))
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax1.yaxis.set_minor_locator(mticker.NullLocator())
    ax1.yaxis.grid(True)
    ax1.xaxis.grid(False)
    ax1.set_ylabel("Retorno acumulado (base 1, escala log)")
    ax1.set_title(
        f"Retorno acumulado — estratégia TopK-Drop\n"
        f"k={k} ações, turnover máx. n={n}/dia, taxa 25 bps · universo B3",
        fontsize=12, fontweight="bold", loc="left", pad=8,
    )
    ax1.legend(frameon=False, fontsize=9)
    finalize_axes(ax1, y_right=False)
    _date_axis(ax1)
    fig1.text(0.08, 0.02, "Fonte: Economatica. Cálculos do autor",
              fontsize=8, color=TEXT_SECONDARY, style="italic")
    fig1.savefig(fig_dir / "BKT_cumulative_return.png")
    plt.close(fig1)
    print("Figure saved → results/figures/BKT_cumulative_return.png")

    # ── Figure 2: Retorno acumulado em excesso ────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(11, 5.5))
    fig2.subplots_adjust(top=0.87, bottom=0.12, left=0.08, right=0.97)

    excess_series: dict[str, pd.Series] = {}
    for name, ret in port_series.items():
        color = COLOR_MAP.get(name, PALETTE[-1])
        bm_ret = benchmark.reindex(ret.index).fillna(0.0)
        log_excess = np.log1p(ret).cumsum() - np.log1p(bm_ret).cumsum()
        ax2.plot(log_excess.index, log_excess.values * 100.0, color=color, label=name)
        excess_series[name] = log_excess * 100.0

    ax2.axhline(0, color=TEXT_SECONDARY, linewidth=0.6, linestyle="--")
    ax2.yaxis.grid(True)
    ax2.xaxis.grid(False)
    ax2.set_ylabel("Log-retorno acumulado em excesso vs benchmark (p.p.)")
    ax2.set_title(
        f"Retorno acumulado em excesso vs benchmark\n"
        f"TopK-Drop k={k}, n={n} · universo B3",
        fontsize=12, fontweight="bold", loc="left", pad=8,
    )
    ax2.legend(frameon=False, fontsize=9)
    finalize_axes(ax2, y_right=False)
    _date_axis(ax2)
    fig2.text(0.08, 0.02, "Fonte: Economatica. Cálculos do autor",
              fontsize=8, color=TEXT_SECONDARY, style="italic")
    fig2.savefig(fig_dir / "BKT_cumulative_excess_return.png")
    plt.close(fig2)
    print("Figure saved → results/figures/BKT_cumulative_excess_return.png")

    # ── Figure 3: Rolling 60-day Rank IC ──────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(11, 4.5))
    fig3.subplots_adjust(top=0.87, bottom=0.12, left=0.08, right=0.97)

    ic_series: dict[str, pd.Series] = {}
    for name, preds in all_preds.items():
        color = COLOR_MAP.get(name, PALETTE[-1])
        r = rolling_rank_ic(preds, window=60)
        ax3.plot(r.index, r.values, color=color, label=name)
        ic_series[name] = r

    ax3.axhline(0, color=TEXT_SECONDARY, linewidth=0.6, linestyle="--")
    ax3.yaxis.grid(True)
    ax3.xaxis.grid(False)
    ax3.set_ylabel("IC de Spearman, média 60 dias")
    ax3.set_title(
        "IC de Spearman — rolling 60 dias\n"
        "Correlação cross-sectional entre retorno previsto e realizado",
        fontsize=12, fontweight="bold", loc="left", pad=8,
    )
    ax3.legend(frameon=False, fontsize=9)
    finalize_axes(ax3, y_right=False)
    _date_axis(ax3)
    fig3.text(0.08, 0.02, "Fonte: Economatica. Cálculos do autor",
              fontsize=8, color=TEXT_SECONDARY, style="italic")
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
