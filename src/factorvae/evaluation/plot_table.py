"""
Renderizador de tabelas comparativas em matplotlib,
com estilo alinhado ao plot_style.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from factorvae.evaluation.plot_style import (
    BRAND_RED, TEXT_PRIMARY, TEXT_SECONDARY, BG_COLOR,
    apply_style, add_title, add_footer, add_brand_bar,
)


def render_comparison_table(
    df: pd.DataFrame,
    out_path,
    title: str = "Comparação entre modelos",
    subtitle: str = "Métricas de IC e performance, período de teste",
    source: str = "Economatica. Cálculos do autor",
    highlight_row: str = "FactorVAE",
    figsize: tuple = (11, 4.5),
) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(top=0.72, bottom=0.12, left=0.04, right=0.96)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    cols      = list(df.columns)
    n_data    = len(cols)
    n_rows    = len(df)
    row_h     = 0.62 / (n_rows + 1)  # body occupies 0.62 of axes height
    y_header  = 0.88

    # Model column: fixed 0.24; remaining split equally among data cols
    model_x   = 0.01
    model_w   = 0.24
    data_x0   = model_w + 0.02
    col_w     = (1.0 - data_x0) / n_data if n_data else 0.1

    # ── Header ──
    ax.text(model_x, y_header, "Modelo", fontweight="bold",
            fontsize=10.5, color=TEXT_PRIMARY, va="center", ha="left",
            fontfamily="serif")
    for j, c in enumerate(cols):
        ax.text(data_x0 + j * col_w + col_w / 2, y_header, c,
                fontweight="bold", fontsize=10.5, color=TEXT_PRIMARY,
                va="center", ha="center", fontfamily="serif")

    ax.axhline(y_header - row_h * 0.6, color=TEXT_PRIMARY, linewidth=0.9,
               xmin=0.01, xmax=0.99)

    # ── Body ──
    for i, (model_name, row) in enumerate(df.iterrows()):
        y          = y_header - (i + 1) * row_h
        highlight  = (model_name == highlight_row)
        row_color  = BRAND_RED if highlight else TEXT_PRIMARY
        row_weight = "bold" if highlight else "normal"

        ax.text(model_x, y, str(model_name), fontweight=row_weight,
                fontsize=10, color=row_color, va="center", ha="left",
                fontfamily="serif")
        for j, c in enumerate(cols):
            ax.text(data_x0 + j * col_w + col_w / 2, y, str(row[c]),
                    fontsize=10, color=row_color, fontweight=row_weight,
                    va="center", ha="center", fontfamily="serif")

        ax.axhline(y - row_h * 0.5, color="#E5E5E5", linewidth=0.4,
                   xmin=0.01, xmax=0.99)

    add_brand_bar(fig)
    add_title(fig, title, subtitle=subtitle)
    add_footer(fig, source=source)
    fig.savefig(out_path)
    plt.close(fig)
