"""Matplotlib padrão — sem customizações."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Padrão matplotlib
BRAND_RED      = "#1f77b4"
TEXT_PRIMARY   = "black"
TEXT_SECONDARY = "gray"
GRID_COLOR     = "lightgray"
BG_COLOR       = "white"

PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
]


def apply_style() -> None:
    """Sem customizações — matplotlib padrão."""
    pass


def add_brand_bar(fig: plt.Figure, **kwargs) -> None:
    """No-op."""
    pass


def add_title(fig: plt.Figure, title: str, subtitle: str | None = None,
              **kwargs) -> None:
    """Sem customização — matplotlib padrão."""
    pass


def add_footer(fig: plt.Figure, source: str, **kwargs) -> None:
    """No-op."""
    pass


def label_lines(ax: plt.Axes, series_map: dict,
                color_map: dict | None = None) -> None:
    """No-op."""
    pass


def finalize_axes(ax: plt.Axes, y_right: bool = False) -> None:
    """No-op."""
    pass


def finalize_axes(ax: plt.Axes, y_right: bool = False) -> None:
    """Ticks sem comprimento, margem horizontal mínima. y_right move o eixo para a direita."""
    ax.tick_params(axis="both", which="both", length=0)
    ax.margins(x=0.02)
    if y_right:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
