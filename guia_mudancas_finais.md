# Guia de Mudanças Finais — FactorVAE

## 0. Resultado da revisão de look-ahead bias

Varri todo o pipeline de dados e modelagem. **Nenhuma fonte de look-ahead foi encontrada.** Pontos verificados:

- `scripts/build_features.py`: todas as features usam janelas `.rolling(...)` e `.ewm(...)` estritamente backward. `pct_change(k)` olha para `close[t-k]`, nunca à frente. Nenhum uso de `.shift(+n)` aparece no cálculo de features.
- Target em `compute_returns`: `fwd = (close.shift(-1) - close.shift(-2)) / close.shift(-1)` — usa preços futuros **somente para o target** (comportamento correto, não é bias).
- `RealDataset.__init__` e `__getitem__`: janela de lookback construída com `searchsorted(date_ts, side="right")`, que devolve o índice logo após o último elemento `<= date_ts`. A fatia `iloc[loc - T : loc]` inclui datas `≤ date_ts`. OK.
- Z-score cross-sectional em `__getitem__` é computado **dentro da própria amostra** (mean/std sobre os tickers daquela data). Nenhuma estatística de treino vaza para validação/teste.
- `FactorVAEDataModule.setup()` já tem `assert train_end < val_start < val_end < test_start`.
- `benchmarks/linear_model.py`: Ridge ajustado só sobre `train_ds`, previsto sobre `test_ds`. Sem vazamento.
- `benchmarks/momentum.py`: usa `x[:, -1, ret_20d_idx]`, que é feature conhecida no instante `t`. OK.

Nenhuma correção necessária. Só proponho um **reforço opcional** do teste `test_no_future_features_in_lookback` (hoje checa só 10 datas × 3 tickers) para varrer tudo.

---

## 1. Fix: inconsistência em `config.yaml`

**Problema.** `config.yaml` declara `training.batch_size: 16`, mas `FactorVAEDataModule` hardcoda `batch_size=1` em `train_dataloader()` / `val_dataloader()` / `test_dataloader()`. O config engana o leitor.

**Arquivo:** `config.yaml`

**Mudança:** remover a chave `batch_size`. Substituir por comentário explicando que o batch é sempre uma cross-section (lógica do modelo assume isso).

```yaml
training:
  # Nota: o DataLoader sempre usa batch_size=1 (uma cross-section por step).
  # A "batch" efetiva do modelo é N_s (número de tickers na data s), que varia.
  max_epochs: 50
  learning_rate: 1.0e-3
  gamma: 1.0
  seed: 42
  sigma_floor: 1.0e-6
```

**Arquivo:** `src/factorvae/training/lightning_module.py`

Nenhuma mudança necessária — já usa o config corretamente (não lê `batch_size`).

**Arquivo:** `tests/test_evaluate_output_schema.py` e `tests/test_training_step.py`

Remover `"batch_size": 1` dos dicts `_make_config()` (ficou redundante).

---

## 2. Módulo de estilo visual: `plot_style.py`

**Objetivo.** Estética profissional inspirada no estilo The Economist: barra vermelha superior como marca, título em bold forte, subtítulo em cinza, eixo y à direita, grid horizontal suave, rótulos de série no fim da curva (sem legenda lateral).

**Criar:** `src/factorvae/evaluation/plot_style.py`

```python
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
BG_COLOR       = "#FBF8F4"   # off-white levemente creme, à la Economist

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
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Lato", "Inter", "Helvetica Neue", "DejaVu Sans"],
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,

        "text.color":        TEXT_PRIMARY,
        "axes.labelcolor":   TEXT_SECONDARY,
        "xtick.color":       TEXT_SECONDARY,
        "ytick.color":       TEXT_SECONDARY,

        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.spines.left":   False,
        "axes.spines.bottom": True,
        "axes.edgecolor":     TEXT_SECONDARY,
        "axes.linewidth":     0.6,

        "axes.grid":        True,
        "axes.grid.axis":   "y",
        "grid.color":       GRID_COLOR,
        "grid.linewidth":   0.5,
        "grid.linestyle":   "-",
        "axes.axisbelow":   True,

        "lines.linewidth":       2.0,
        "lines.solid_capstyle":  "round",

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
              x: float = 0.06, y_title: float = 0.905, y_sub: float = 0.855) -> None:
    fig.text(x, y_title, title,
             fontsize=16, fontweight="bold", color=TEXT_PRIMARY, ha="left")
    if subtitle:
        fig.text(x, y_sub, subtitle,
                 fontsize=10.5, color=TEXT_SECONDARY, ha="left")


def add_footer(fig: plt.Figure, source: str,
               x: float = 0.06, y: float = 0.02) -> None:
    fig.text(x, y, f"Fonte: {source}",
             fontsize=8.5, color=TEXT_SECONDARY, ha="left", style="italic")


def label_lines(ax: plt.Axes, series_map: dict,
                color_map: dict | None = None) -> None:
    """Rótulos diretamente no fim da linha, em vez de legenda."""
    for i, (label, values) in enumerate(series_map.items()):
        color = (color_map or {}).get(label, PALETTE[i % len(PALETTE)])
        x_last = values.index[-1] if hasattr(values, "index") else len(values) - 1
        y_last = values.iloc[-1] if hasattr(values, "iloc") else values[-1]
        ax.annotate(label, xy=(x_last, y_last),
                    xytext=(6, 0), textcoords="offset points",
                    color=color, fontsize=10, fontweight="semibold", va="center")


def finalize_axes(ax: plt.Axes, y_right: bool = True) -> None:
    """Eixo y à direita (convenção Economist) e ticks discretos."""
    ax.tick_params(axis="both", which="both", length=0)
    ax.margins(x=0.02)
    if y_right:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
```

---

## 3. Refatorar os scripts de plot existentes

### 3.1 `scripts/backtest.py`

Substituir a seção `# ── Figure 1/2/3` pelo template abaixo. Remover `_apply_date_fmt` antigo e usar `plot_style.finalize_axes`.

```python
from factorvae.evaluation.plot_style import (
    apply_style, PALETTE, TEXT_SECONDARY,
    add_title, add_footer, add_brand_bar, label_lines, finalize_axes,
)
import matplotlib.dates as mdates

apply_style()
COLOR_MAP = {
    "FactorVAE":      PALETTE[0],
    "Momentum":       PALETTE[1],
    "Linear (Ridge)": PALETTE[2],
    "MLP":            PALETTE[3],
    "GRU":            PALETTE[4],
}

def _date_axis(ax):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# ─── Figura 1: Retorno acumulado ─────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(11, 5.5))
fig1.subplots_adjust(top=0.80, bottom=0.14, left=0.06, right=0.90)

cum_series = {}
for name, ret in port_series.items():
    color = COLOR_MAP.get(name, PALETTE[-1])
    cum = (1.0 + ret).cumprod()
    ax1.plot(cum.index, cum.values, color=color)
    cum_series[name] = cum

bm_aligned = benchmark.reindex(sorted({d for r in port_series.values() for d in r.index})).fillna(0.0)
cum_bm = (1.0 + bm_aligned).cumprod()
label_bm = benchmark.name if benchmark.name else "Equal-Weight Market"
ax1.plot(cum_bm.index, cum_bm.values, color=TEXT_SECONDARY, linestyle="--", linewidth=1.3)
cum_series[label_bm] = cum_bm

label_lines(ax1, cum_series, color_map={**COLOR_MAP, label_bm: TEXT_SECONDARY})
finalize_axes(ax1)
_date_axis(ax1)
ax1.set_ylabel("Retorno acumulado (1 = início)")

add_brand_bar(fig1)
add_title(fig1, "Retorno acumulado — estratégia TopK-Drop",
          subtitle=f"k={k} ações, turnover máx. n={n}/dia, taxa 10 bps · universo B3")
add_footer(fig1, source="Economatica. Cálculos do autor")
fig1.savefig(fig_dir / "cumulative_return.png")
plt.close(fig1)
```

Aplicar o mesmo padrão a `cumulative_excess_return.png` e `rolling_rank_ic.png`. Para o IC rolling, o ylabel é `"IC de Spearman, média 60 dias"` e o subtítulo algo como `"Correlação cross-sectional entre retorno previsto e realizado"`.

### 3.2 `scripts/plot_training_curves.py`

Mesmo pattern, título bilíngue: `"Diagnóstico de treino"` com subtítulo `"Perda total, componente de reconstrução e KL, Rank IC de validação"`. Cada subplot usa `PALETTE[i]`. Chamar `add_brand_bar` e `add_title` no `fig` nível, e `finalize_axes(ax, y_right=False)` nos subplots (só a tabela comparativa vai ter y à direita).

---

## 4. Novo módulo de tabela renderizada

**Por quê.** O CSV `comparison_table.csv` é útil para inspecionar, mas a defesa precisa de uma tabela com tipografia consistente com os gráficos.

**Criar:** `src/factorvae/evaluation/plot_table.py`

```python
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

    # Header
    cols     = list(df.columns)
    n_rows   = len(df)
    n_cols   = len(cols) + 1  # +1 for row label (model name)
    col_w    = 1.0 / n_cols
    row_h    = 0.8 / (n_rows + 1)
    y_header = 0.85

    ax.text(0.02, y_header, "Modelo", fontweight="bold",
            fontsize=10.5, color=TEXT_PRIMARY, va="center")
    for j, c in enumerate(cols):
        ax.text((j + 1) * col_w + 0.02, y_header, c,
                fontweight="bold", fontsize=10.5, color=TEXT_PRIMARY,
                va="center", ha="left")

    ax.axhline(y_header - row_h * 0.55, color=TEXT_PRIMARY, linewidth=0.9,
               xmin=0.02, xmax=0.98)

    # Body
    for i, (model_name, row) in enumerate(df.iterrows()):
        y = y_header - (i + 1) * row_h
        is_highlight = (model_name == highlight_row)
        row_color = BRAND_RED if is_highlight else TEXT_PRIMARY
        row_weight = "bold" if is_highlight else "normal"

        ax.text(0.02, y, str(model_name), fontweight=row_weight,
                fontsize=10, color=row_color, va="center")
        for j, c in enumerate(cols):
            ax.text((j + 1) * col_w + 0.02, y, str(row[c]),
                    fontsize=10, color=row_color, fontweight=row_weight,
                    va="center", ha="left")

        ax.axhline(y - row_h * 0.5, color="#E5E5E5", linewidth=0.4,
                   xmin=0.02, xmax=0.98)

    add_brand_bar(fig)
    add_title(fig, title, subtitle=subtitle)
    add_footer(fig, source=source)
    fig.savefig(out_path)
    plt.close(fig)
```

**Integrar** em `scripts/backtest.py`, logo após construir o `table`:

```python
from factorvae.evaluation.plot_table import render_comparison_table

# Formatar o DataFrame como string já formatado (%, ±, etc.)
formatted = _format_for_display(table)  # mover a lógica de formatação do print_comparison para uma função
render_comparison_table(
    formatted,
    out_path=fig_dir / "comparison_table.png",
    title="Comparação entre modelos",
    subtitle=f"Métricas IC + performance · TopK-Drop (k={k}, n={n}) · 2019–2025",
)
```

---

## 5. Benchmarks adicionais: MLP e GRU

**Motivação.** O paper original compara o FactorVAE contra modelos sequenciais simples (MLP, GRU, LSTM, Transformer, SFM, ALSTM). Hoje temos só Momentum e Ridge. Adicionar **MLP** e **GRU** cobre as duas ablações mais informativas:

- **MLP** testa se a estrutura temporal do GRU agrega algo.
- **GRU puro** (sem VAE, sem fatores) testa se a estrutura probabilística/fatorial agrega algo — é o benchmark mais crítico conceitualmente, porque isola exatamente o que o FactorVAE adiciona.

Ambos são arquiteturas simples, não requerem pipelines novos, e são os mesmos benchmarks do paper.

### 5.1 `benchmarks/mlp.py`

```python
"""
MLP benchmark: rede feedforward simples sobre features do último timestep.

A cada data, pega x[:, -1, :] (N, C), passa por MLP de 2 camadas, prediz y.
Treinamento: concatena todas as datas de treino, minibatch padrão.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from factorvae.data.dataset import RealDataset

ROOT = Path(__file__).resolve().parents[1]


class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _stack_last_timestep(dataset: RealDataset):
    X_all, y_all, date_labels, ticker_labels = [], [], [], []
    for idx in range(len(dataset)):
        x, y, _ = dataset[idx]
        date_ts = dataset.trading_dates[idx]
        tickers = dataset.universe_by_date[date_ts]
        X_all.append(x[:, -1, :].numpy())
        y_all.append(y.numpy())
        date_labels.extend([date_ts] * len(tickers))
        ticker_labels.extend(tickers)
    return (np.concatenate(X_all), np.concatenate(y_all),
            date_labels, ticker_labels)


def train_and_predict(config: dict,
                      hidden: int = 64,
                      epochs: int = 20,
                      lr: float = 1e-3,
                      batch_size: int = 256,
                      seed: int = 42) -> pd.DataFrame:
    torch.manual_seed(seed)
    dc = config["data"]

    train_ds = RealDataset(dc["processed_dir"], dc["train_start"], dc["train_end"],
                           dc["sequence_length"])
    test_ds  = RealDataset(dc["processed_dir"], dc["test_start"],  dc["test_end"],
                           dc["sequence_length"])

    print(f"  MLP: treinando em {len(train_ds)} datas…")
    X_tr, y_tr, _, _ = _stack_last_timestep(train_ds)
    C = X_tr.shape[1]

    model = SimpleMLP(C, hidden=hidden)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                        batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optim.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optim.step()

    print(f"  MLP: prevendo em {len(test_ds)} datas…")
    X_te, _, dates, tickers = _stack_last_timestep(test_ds)
    model.eval()
    with torch.no_grad():
        mu_pred = model(torch.tensor(X_te, dtype=torch.float32)).numpy()

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

### 5.2 `benchmarks/gru.py`

GRU que prediz `y` diretamente a partir da sequência completa `(N, T, C)`, sem a estrutura VAE. Mesmo contrato de output.

```python
"""
GRU benchmark: GRU que prediz y diretamente a partir de (N, T, C).
Sem VAE, sem estrutura fatorial — isola o ganho do GRU puro.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from factorvae.data.dataset import RealDataset

ROOT = Path(__file__).resolve().parents[1]


class SimpleGRU(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 20):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (N, T, C)
        _, h_n = self.gru(x)          # h_n: (1, N, H)
        return self.head(h_n.squeeze(0)).squeeze(-1)  # (N,)


def train_and_predict(config: dict,
                      hidden: int = 20,
                      epochs: int = 15,
                      lr: float = 1e-3,
                      seed: int = 42) -> pd.DataFrame:
    torch.manual_seed(seed)
    dc = config["data"]

    train_ds = RealDataset(dc["processed_dir"], dc["train_start"], dc["train_end"],
                           dc["sequence_length"])
    test_ds  = RealDataset(dc["processed_dir"], dc["test_start"],  dc["test_end"],
                           dc["sequence_length"])

    C = train_ds.C
    model = SimpleGRU(C, hidden=hidden)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"  GRU: treinando em {len(train_ds)} datas × {epochs} épocas…")
    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(len(train_ds))
        total_loss = 0.0
        for idx in indices:
            x, y, _ = train_ds[idx]
            optim.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        print(f"    epoch {epoch + 1}: loss = {total_loss / len(train_ds):.4f}")

    print(f"  GRU: prevendo em {len(test_ds)} datas…")
    model.eval()
    raw_returns = (
        pd.read_parquet(Path(dc["processed_dir"]) / "returns.parquet")
        .assign(date=lambda df: pd.to_datetime(df["date"]))
        .set_index(["date", "ticker"])["forward_return"]
    )

    records = []
    with torch.no_grad():
        for idx in range(len(test_ds)):
            x, _, _ = test_ds[idx]
            mu_pred = model(x).numpy()
            date_ts = test_ds.trading_dates[idx]
            tickers = test_ds.universe_by_date[date_ts]
            for i, ticker in enumerate(tickers):
                try:
                    y_true = float(raw_returns.loc[(date_ts, ticker)])
                except KeyError:
                    y_true = float("nan")
                records.append({
                    "date":       date_ts.strftime("%Y-%m-%d"),
                    "ticker":     ticker,
                    "mu_pred":    float(mu_pred[i]),
                    "sigma_pred": 0.0,
                    "y_true":     y_true,
                })
    return pd.DataFrame(records)
```

### 5.3 Integrar em `benchmarks/run_benchmarks.py`

```python
from benchmarks.mlp import train_and_predict as mlp_predict
from benchmarks.gru import train_and_predict as gru_predict

print("Running MLP benchmark…")
mlp_df = mlp_predict(config)
mlp_df.to_parquet(out_dir / "mlp_predictions.parquet", index=False)

print("Running GRU benchmark…")
gru_df = gru_predict(config)
gru_df.to_parquet(out_dir / "gru_predictions.parquet", index=False)
```

### 5.4 Integrar em `src/factorvae/evaluation/comparison.py`

No dict `sources` dentro de `load_all_predictions`:

```python
sources = {
    "FactorVAE":      root / "results" / "predictions" / "predictions.parquet",
    "Momentum":       root / "benchmarks" / "predictions" / "momentum_predictions.parquet",
    "Linear (Ridge)": root / "benchmarks" / "predictions" / "linear_predictions.parquet",
    "MLP":            root / "benchmarks" / "predictions" / "mlp_predictions.parquet",
    "GRU":            root / "benchmarks" / "predictions" / "gru_predictions.parquet",
}
```

---

## 6. Extensão com índices macro (contribuição do TCC)

**Ideia.** O predictor original só enxerga o embedding `e` de cada ticker (20 dias de features do próprio ativo). Um argumento natural é que em mercados com troca frequente de regime (Brasil), a distribuição de fatores depende do contexto macro — Selic, câmbio, Treasury, commodities. A extensão injeta um vetor macro $m_t$ no predictor.

**Posição na arquitetura.** O macro entra **só no `FactorPredictor`**, não no encoder nem no decoder. Motivo: o predictor é quem estima a distribuição prior dos fatores para o próximo dia. O encoder vê $y$ (oráculo) e não precisa; o decoder traduz fatores em retornos por ticker, agnostic ao macro.

**Arquivo novo:** `data/processed/macro.parquet` com schema `[date, feature_name, value]`. O agente programador não precisa gerá-lo agora — basta garantir que o pipeline lê quando disponível.

### 6.1 Mudança em `src/factorvae/data/dataset.py`

Adicionar parâmetro opcional `use_macro` e leitura condicional no `RealDataset`:

```python
def __init__(
    self,
    processed_dir: str | Path,
    start_date: str,
    end_date: str,
    sequence_length: int = 20,
    feature_cols: list[str] | None = None,
    use_macro: bool = False,
    macro_normalizer: "MacroNormalizer | None" = None,
):
    # ... código existente ...
    self.use_macro = use_macro
    self._macro_by_date: dict[pd.Timestamp, np.ndarray] | None = None
    self.macro_dim: int = 0

    if use_macro:
        macro_path = Path(processed_dir) / "macro.parquet"
        if not macro_path.exists():
            raise FileNotFoundError(
                f"use_macro=True mas {macro_path} não existe."
            )
        macro_wide = (
            pd.read_parquet(macro_path)
            .assign(date=lambda df: pd.to_datetime(df["date"]))
            .pivot(index="date", columns="feature_name", values="value")
            .sort_index()
            .ffill()   # macro não tem dados nos fins de semana
        )
        if macro_normalizer is None:
            raise ValueError(
                "use_macro=True requer um macro_normalizer (ajustado no treino)."
            )
        macro_norm = macro_normalizer.transform(macro_wide)
        self.macro_dim = macro_normalizer.dim
        self._macro_by_date = {
            ts: macro_norm.loc[ts].values.astype(np.float32)
            for ts in macro_norm.index if ts in macro_norm.index
        }
```

### 6.2 `MacroNormalizer` em `src/factorvae/data/datamodule.py`

Para garantir ausência de vazamento, a normalização do macro usa mean/std calculados **apenas no range de treino**:

```python
# src/factorvae/data/datamodule.py

import pandas as pd
from pathlib import Path


class MacroNormalizer:
    """Normaliza dados macro com estatísticas do range de treino apenas."""

    def __init__(self, macro_wide: pd.DataFrame, train_start: str, train_end: str):
        train_slice = macro_wide.loc[
            pd.Timestamp(train_start):pd.Timestamp(train_end)
        ]
        self.mean = train_slice.mean()
        self.std  = train_slice.std() + 1e-8
        self.columns = macro_wide.columns.tolist()

    def transform(self, macro_wide: pd.DataFrame) -> pd.DataFrame:
        return (macro_wide - self.mean) / self.std

    @property
    def dim(self) -> int:
        return len(self.columns)
```

E o DataModule passa o `MacroNormalizer` ajustado no treino para os três splits:

```python
class FactorVAEDataModule(L.LightningDataModule):
    def setup(self, stage: str | None = None) -> None:
        dc = self.config["data"]
        # ... asserts existentes ...

        use_macro = dc.get("use_macro", False)
        macro_normalizer = None
        if use_macro and not self.use_synthetic:
            macro_path = Path(dc["processed_dir"]) / "macro.parquet"
            macro_wide = (
                pd.read_parquet(macro_path)
                .assign(date=lambda df: pd.to_datetime(df["date"]))
                .pivot(index="date", columns="feature_name", values="value")
                .sort_index()
                .ffill()
            )
            macro_normalizer = MacroNormalizer(
                macro_wide, dc["train_start"], dc["train_end"]
            )

        if self.use_synthetic:
            # ... código synthetic existente, sem macro ...
            pass
        else:
            T = dc["sequence_length"]
            ds_kwargs = {"use_macro": use_macro, "macro_normalizer": macro_normalizer}
            self._train = RealDataset(dc["processed_dir"], dc["train_start"], dc["train_end"], T, **ds_kwargs)
            self._val   = RealDataset(dc["processed_dir"], dc["val_start"],   dc["val_end"],   T, **ds_kwargs)
            self._test  = RealDataset(dc["processed_dir"], dc["test_start"],  dc["test_end"],  T, **ds_kwargs)
```

### 6.3 `__getitem__` devolve tupla extra

```python
def __getitem__(self, idx):
    # ... código existente até construir x, y, mask ...
    if self.use_macro:
        date_ts = self.trading_dates[idx]
        m = torch.from_numpy(self._macro_by_date[date_ts])   # (macro_dim,)
        return x, m, y, mask
    return x, y, mask
```

### 6.4 Mudança em `src/factorvae/models/factor_predictor.py`

Adicionar branch opcional para macro:

```python
class FactorPredictor(nn.Module):
    def __init__(self, hidden_dim: int, num_factors: int,
                 leaky_slope: float = 0.1, macro_dim: int = 0):
        super().__init__()
        self.heads = nn.ModuleList(
            [SingleHeadAttention(hidden_dim) for _ in range(num_factors)]
        )
        self.macro_dim = macro_dim
        if macro_dim > 0:
            self.macro_proj = nn.Sequential(
                nn.Linear(macro_dim, hidden_dim),
                nn.LeakyReLU(leaky_slope),
            )
            dist_in = hidden_dim * 2
        else:
            self.macro_proj = None
            dist_in = hidden_dim

        self.dist_net = DistributionNetwork(dist_in, leaky_slope)

    def forward(self, e: torch.Tensor,
                m: torch.Tensor | None = None) -> tuple:
        h_muti = torch.stack([head(e) for head in self.heads], dim=0)  # (K, H)
        if self.macro_proj is not None:
            assert m is not None, "macro_dim > 0 mas m não foi passado"
            m_proj = self.macro_proj(m)                                  # (H,)
            m_proj_expanded = m_proj.unsqueeze(0).expand(h_muti.size(0), -1)
            h_muti = torch.cat([h_muti, m_proj_expanded], dim=-1)        # (K, 2H)
        return self.dist_net(h_muti)
```

### 6.5 `DistributionNetwork` — aceitar `in_dim != hidden_dim`

```python
class DistributionNetwork(nn.Module):
    def __init__(self, in_dim: int, leaky_slope: float = 0.1):
        super().__init__()
        self.hidden = nn.Linear(in_dim, in_dim)
        self.act = nn.LeakyReLU(negative_slope=leaky_slope)
        self.mu_head = nn.Linear(in_dim, 1)
        self.sigma_head = nn.Linear(in_dim, 1)
    # resto idêntico
```

### 6.6 `FactorVAE` e fluxos

```python
class FactorVAE(nn.Module):
    def __init__(self, config: dict):
        # ... código existente ...
        macro_dim = config["model"].get("macro_dim", 0)
        self.predictor = FactorPredictor(H, K, slope, macro_dim=macro_dim)

    def forward_train(self, x, y, m=None):
        e = self.feature_extractor(x)
        mu_post, sigma_post = self.encoder(y, e)
        mu_prior, sigma_prior = self.predictor(e, m=m)
        mu_y_rec, sigma_y_rec = self.decoder(mu_post, sigma_post, e)
        return {...}  # igual

    def forward_predict(self, x, m=None):
        e = self.feature_extractor(x)
        mu_prior, sigma_prior = self.predictor(e, m=m)
        return self.decoder(mu_prior, sigma_prior, e)
```

### 6.7 `FactorVAELightning` — desempacotar batch condicional

```python
def training_step(self, batch, batch_idx):
    if len(batch) == 4:
        x, m, y, mask = batch
        x, m, y = x.squeeze(0), m.squeeze(0), y.squeeze(0)
    else:
        x, y, mask = batch
        x, y = x.squeeze(0), y.squeeze(0)
        m = None
    out = self.model.forward_train(x, y, m=m)
    # resto igual
```

### 6.8 `config.yaml`

```yaml
model:
  num_features: 20
  hidden_dim: 20
  num_factors: 8
  num_portfolios: 64
  leaky_relu_slope: 0.1
  macro_dim: 0          # preencher com len(macro_cols) quando use_macro=True

data:
  # ... campos existentes ...
  use_macro: false      # flag mestre para ablação
```

### 6.9 Convenção de ablação

Diego deve rodar **dois** experimentos:

1. `use_macro: false` → FactorVAE vanilla (replicação do paper em B3).
2. `use_macro: true`  → FactorVAE + macro (contribuição do TCC).

As predições vão para `results/predictions/predictions_vanilla.parquet` e `results/predictions/predictions_macro.parquet`. Acrescentar ambos em `comparison.load_all_predictions` como `"FactorVAE"` e `"FactorVAE + Macro"`.

---

## 7. Reforço opcional dos testes de look-ahead

`tests/test_real_dataset.py::test_no_future_features_in_lookback` hoje só varre as primeiras 10 datas × 3 tickers. Estender para cobrir todas as datas:

```python
@skip_if_no_data
def test_no_future_features_in_lookback_all(small_dataset):
    """Verifica TODA a cross-section de TODAS as datas do small_dataset."""
    for i in range(len(small_dataset)):
        date_ts = small_dataset.trading_dates[i]
        tickers = small_dataset.universe_by_date[date_ts]
        for ticker in tickers:
            feat_df = small_dataset._features_by_ticker[ticker]
            loc = feat_df.index.searchsorted(date_ts, side="right")
            window_dates = feat_df.index[loc - small_dataset.T : loc]
            assert window_dates[-1] <= date_ts, (
                f"Feature futura no lookback: {ticker} em {date_ts} "
                f"tem janela até {window_dates[-1]}"
            )
```

É mais lento que o teste original (por isso foi restrito), mas `small_dataset` cobre só 6 meses, então roda em segundos.

---

## 8. Ordem de implementação sugerida

1. Fix do `config.yaml` (seção 1) — trivial, testa em seguida com `pytest`.
2. Criar `plot_style.py` e `plot_table.py` (seções 2 e 4).
3. Refatorar os três plots existentes (seção 3). Rodar `scripts/backtest.py` e conferir figuras.
4. Adicionar MLP e GRU (seção 5). Rodar `benchmarks/run_benchmarks.py`.
5. Reforço do teste de look-ahead (seção 7).
6. Extensão macro (seção 6) — a maior mudança, feita por último para não bloquear as outras. Testar primeiro com `use_macro: false` para garantir que não quebrou nada, depois com `true`.

Após todas as etapas, rodar `pytest tests/` e conferir que nada regrediu.
