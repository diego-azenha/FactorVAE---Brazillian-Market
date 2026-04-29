"""
Estilo visual para gráficos do TCC — inspirado em The Economist / OWID.

Uso:
    from factorvae.evaluation.plot_style import (
        apply_style, PALETTE, TEXT_PRIMARY, TEXT_SECONDARY,
        add_title, add_footer, add_brand_bar, label_lines, finalize_axes,
    )

    apply_style()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.06, right=0.90)
    # ... plotagem ...
    add_brand_bar(fig)
    add_title(fig, "Título", subtitle="Subtítulo descritivo")
    add_footer(fig, source="Economatica. Cálculos do autor")
    finalize_axes(ax)
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

BRAND_RED      = "#C8102E"   # Insper lava-red (serve como "Economist red")
TEXT_PRIMARY   = "#121212"
TEXT_SECONDARY = "#6F6F6F"
GRID_COLOR     = "#E5E5E5"
BG_COLOR       = "#FFFFFF"   # branco puro

PALETTE = [
    BRAND_RED,    # modelo principal (FactorVAE)
    "#1F4E79",    # azul petróleo (benchmark principal)
    "#5A8F29",    # verde oliva
    "#C77D02",    # âmbar
    "#6B4E71",    # violeta apagado
    "#4A4A4A",    # grafite
]


def apply_style() -> None:
    """Aplica rcParams globais. Chame uma vez antes de criar figuras."""
    mpl.rcParams.update({
        "font.family":      "sans-serif",
        "font.sans-serif":  ["DejaVu Sans", "Helvetica Neue", "Helvetica", "Arial", "Liberation Sans"],
        "font.size":        10,
        "axes.titlesize":   11,
        "axes.labelsize":   10,
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
        "legend.fontsize":  9,

        "text.color":      TEXT_PRIMARY,
        "axes.labelcolor": "#2D2D2D",
        "xtick.color":     "#2D2D2D",
        "ytick.color":     "#2D2D2D",

        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.spines.left":   False,
        "axes.spines.bottom": True,
        "axes.edgecolor":     "#2D2D2D",
        "axes.linewidth":     0.8,

        "axes.grid":       True,
        "axes.grid.axis":  "y",
        "grid.color":      GRID_COLOR,
        "grid.linewidth":  0.5,
        "grid.linestyle":  "-",
        "axes.axisbelow":  True,

        "lines.linewidth":      1.0,
        "lines.solid_capstyle": "round",

        "figure.facecolor":  BG_COLOR,
        "axes.facecolor":    BG_COLOR,
        "savefig.facecolor": BG_COLOR,
        "savefig.dpi":       220,
        "savefig.bbox":      "tight",
    })


def add_brand_bar(fig: plt.Figure, x: float = 0.06, y: float = 0.965,
                  width: float = 0.05, height: float = 0.012) -> None:
    """Barrinha vermelha no topo esquerdo — marca visual estilo Economist."""
    fig.add_artist(plt.Rectangle(
        (x, y), width, height, color=BRAND_RED, clip_on=False,
        transform=fig.transFigure,
    ))


def add_title(fig: plt.Figure, title: str, subtitle: str | None = None,
              x: float = 0.06, y_title: float = 0.905, y_sub: float = 0.855,
              fontsize_sub: float = 10.5, wrap_sub: bool = False) -> None:
    fig.text(x, y_title, title,
             fontsize=16, fontweight="bold", color=TEXT_PRIMARY, ha="left")
    if subtitle:
        fig.text(x, y_sub, subtitle,
                 fontsize=fontsize_sub, color=TEXT_SECONDARY, ha="left",
                 wrap=wrap_sub)


def add_footer(fig: plt.Figure, source: str,
               x: float = 0.06, y: float = 0.02) -> None:
    fig.text(x, y, f"Fonte: {source}",
             fontsize=8.5, color=TEXT_SECONDARY, ha="left", style="italic")


def label_lines(ax: plt.Axes, series_map: dict,
                color_map: dict | None = None) -> None:
    """Rótulos no fim de cada linha, com deslocamento vertical para evitar sobreposição."""
    if not series_map:
        return

    entries: list[dict] = []
    for i, (label, values) in enumerate(series_map.items()):
        color = (color_map or {}).get(label, PALETTE[i % len(PALETTE)])
        x_last = values.index[-1] if hasattr(values, "index") else len(values) - 1
        y_last = float(values.iloc[-1] if hasattr(values, "iloc") else values[-1])
        entries.append({"y_data": y_last, "y_adj": y_last,
                        "label": label, "x": x_last, "color": color})

    entries.sort(key=lambda e: e["y_data"])

    # Convert 14 pt gap into data units using axes geometry
    fig = ax.get_figure()
    ax_height_in = ax.get_position().height * fig.get_size_inches()[1]
    ax_height_pts = ax_height_in * 72.0
    y_lo, y_hi = ax.get_ylim()
    data_range = (y_hi - y_lo) if y_hi != y_lo else 1.0
    pts_per_data = ax_height_pts / data_range
    min_sep = 14.0 / pts_per_data  # 14 pt per label slot

    for i in range(1, len(entries)):
        if entries[i]["y_adj"] - entries[i - 1]["y_adj"] < min_sep:
            entries[i]["y_adj"] = entries[i - 1]["y_adj"] + min_sep

    for e in entries:
        dy_pts = (e["y_adj"] - e["y_data"]) * pts_per_data
        ax.annotate(
            e["label"],
            xy=(e["x"], e["y_data"]),
            xytext=(8, dy_pts),
            textcoords="offset points",
            color=e["color"],
            fontsize=9,
            fontweight="bold",
            va="center",
            annotation_clip=False,
        )


def finalize_axes(ax: plt.Axes, y_right: bool = False) -> None:
    """Ticks sem comprimento, margem horizontal mínima. y_right move o eixo para a direita."""
    ax.tick_params(axis="both", which="both", length=0)
    ax.margins(x=0.02)
    if y_right:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
