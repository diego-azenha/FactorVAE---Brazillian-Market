# FactorVAE — Guia de Scaffolding para Replicação

Este documento especifica a estrutura, responsabilidades e integração dos componentes de um repositório Python para replicar o FactorVAE (Duan et al., 2022). O guia é destinado a um agente desenvolvedor: assume que ele não conhece o paper a fundo, então cada seção traz não só *o que* implementar mas *por que* existe e *como* se conecta aos demais.

O escopo é uma execução única e organizada, sem suporte a múltiplos experimentos paralelos ou tracking de runs.

---

## 1. Visão do modelo

### 1.1 Problema

Para cada data $s$, dado um tensor de características históricas $x_s \in \mathbb{R}^{N_s \times T \times C}$ cobrindo os $T$ dias anteriores e os $N_s$ tickers válidos, o modelo precisa produzir uma distribuição preditiva do retorno futuro $y_s \in \mathbb{R}^{N_s}$. A formulação segue o Dynamic Factor Model:

$$y_s = \alpha_s + \sum_{k=1}^{K} \beta_s^{(k)} z_s^{(k)} + \epsilon_s$$

onde $\alpha_s \in \mathbb{R}^{N_s}$ é o retorno idiossincrático, $\beta_s \in \mathbb{R}^{N_s \times K}$ é a matriz de exposição, $z_s \in \mathbb{R}^K$ são os $K$ fatores latentes, e $\epsilon_s$ é ruído de média zero. A contribuição do FactorVAE é tratar $z_s$ como variável aleatória gaussiana aprendida via VAE.

### 1.2 Arquitetura em quatro módulos

O modelo completo tem quatro redes, encadeadas em torno de um **embedding compartilhado** $e \in \mathbb{R}^{N \times H}$:

| Módulo | Símbolo | Entrada | Saída | Quando opera |
|--------|---------|---------|-------|--------------|
| Feature Extractor | $\phi_{\text{feat}}$ | $x$ | $e$ | Sempre |
| Factor Encoder | $\phi_{\text{enc}}$ | $(y, e)$ | $(\mu_{\text{post}}, \sigma_{\text{post}})$ | Apenas no treino |
| Factor Predictor | $\phi_{\text{pred}}$ | $e$ | $(\mu_{\text{prior}}, \sigma_{\text{prior}})$ | Treino e inferência |
| Factor Decoder | $\phi_{\text{dec}}$ | $(\mu_z, \sigma_z, e)$ | $(\mu_y, \sigma_y)$ | Treino e inferência |

O embedding $e$ é produzido uma vez por forward pass e consumido por três módulos (encoder, predictor, decoder). Esse compartilhamento é a razão pela qual o feature extractor precisa ser bem treinado: ele é o gargalo por onde toda informação histórica passa.

### 1.3 Os dois modos de operação

**Treino.** Os quatro módulos participam. O encoder enxerga $y$ e produz uma distribuição posterior $q(z|x,y)$. O predictor não enxerga $y$ e produz uma distribuição prior $p(z|x)$. O decoder reconstrói $y$ a partir da posterior (rota de reconstrução). O prior é empurrado para se aproximar da posterior via KL.

**Inferência.** O encoder é descartado. O predictor produz $(\mu_{\text{prior}}, \sigma_{\text{prior}})$ a partir de $e$, e o decoder as propaga para $(\mu_{\text{pred}}, \sigma_{\text{pred}})$. Nenhuma parte do fluxo depende de $y$.

A figura mental correta é: o predictor aprende a imitar o encoder. Depois de treinado, o encoder vira dispensável.

---

## 2. Componentes em detalhe

Esta seção especifica cada módulo por dentro: operações, fórmulas, shapes e contrato de interface. Hiperparâmetros dimensionais seguem notação fixa ao longo do documento:

- $N$: número de tickers na cross-section (varia por data)
- $T$: tamanho da janela temporal
- $C$: número de características por ticker por dia
- $H$: dimensão do hidden state (embedding $e$)
- $K$: número de fatores latentes
- $M$: número de portfólios do encoder

### 2.1 Feature Extractor ($\phi_{\text{feat}}$)

**Papel.** Transformar séries temporais em embeddings que resumem o histórico de cada ticker.

**Operações.** A cada passo $t$, a característica do ticker $i$ passa por uma projeção linear com LeakyReLU, e o resultado alimenta uma GRU que acumula contexto temporal:

$$h_{\text{proj}}^{(i,t)} = \text{LeakyReLU}\left(w_{\text{proj}} x^{(i,t)} + b_{\text{proj}}\right)$$

$$h_{\text{gru}}^{(i,t)} = \text{GRU}\left(h_{\text{proj}}^{(i,t)}, h_{\text{gru}}^{(i,t-1)}\right)$$

O embedding final é o hidden state do último passo: $e^{(i)} = h_{\text{gru}}^{(i,T)}$.

**Parâmetros.** $w_{\text{proj}} \in \mathbb{R}^{C \times H}$, $b_{\text{proj}} \in \mathbb{R}^H$, mais os pesos internos da GRU. A projeção e a GRU são compartilhadas entre tickers — o mesmo conjunto de pesos processa todos os $N$ tickers em paralelo.

**Contrato.**
- Input: `x` de shape `(N, T, C)`.
- Output: `e` de shape `(N, H)`.

### 2.2 Factor Encoder ($\phi_{\text{enc}}$)

**Papel.** Extrair fatores "ótimos" usando informação futura ($y$). Funciona como oracle durante o treino.

O encoder tem dois submódulos: **Portfolio Layer** e **Mapping Layer**. A motivação de começar por portfólios em vez de alimentar $y$ direto em uma rede densa é dupla: reduz dimensionalidade ($N$ variável → $M$ fixo) e protege contra tickers faltantes na cross-section.

#### 2.2.1 Portfolio Layer ($\phi_p$)

Constrói $M$ portfólios dinâmicos cujos pesos são softmax dos embeddings:

$$a_p^{(i,j)} = \frac{\exp\left(w_p e^{(i)} + b_p\right)^{(j)}}{\sum_{i=1}^{N} \exp\left(w_p e^{(i)} + b_p\right)^{(j)}}, \qquad \sum_{i=1}^{N} a_p^{(i,j)} = 1 \;\; \forall j$$

Retornos de portfólio são médias ponderadas dos retornos individuais:

$$y_p^{(j)} = \sum_{i=1}^{N} y^{(i)} \, a_p^{(i,j)}$$

**Atenção à dimensão do softmax.** A normalização é *cross-sectional*: para cada portfólio $j$, somamos sobre os tickers $i$. Implementar como softmax errado na dimensão $j$ ainda compila mas destrói o sinal do modelo.

**Parâmetros.** $w_p \in \mathbb{R}^{H \times M}$, $b_p \in \mathbb{R}^M$.

**Contrato.**
- Input: `y` de shape `(N,)`, `e` de shape `(N, H)`.
- Output intermediário: `y_p` de shape `(M,)`, independente de $N$.

#### 2.2.2 Mapping Layer ($\phi_{\text{map}}$)

Mapeia os $M$ retornos de portfólio para os parâmetros da posterior:

$$\mu_{\text{post}} = w_{\text{post}_\mu} \, y_p + b_{\text{post}_\mu}$$

$$\sigma_{\text{post}} = \text{Softplus}\left(w_{\text{post}_\sigma} \, y_p + b_{\text{post}_\sigma}\right)$$

O Softplus garante $\sigma > 0$ sem descontinuidade.

**Parâmetros.** $w_{\text{post}_\mu}, w_{\text{post}_\sigma} \in \mathbb{R}^{M \times K}$, $b_{\text{post}_\mu}, b_{\text{post}_\sigma} \in \mathbb{R}^K$.

**Contrato.**
- Input: `y_p` de shape `(M,)`.
- Output: `(mu_post, sigma_post)`, ambos de shape `(K,)`.

### 2.3 Factor Predictor ($\phi_{\text{pred}}$)

**Papel.** Produzir a distribuição prior dos fatores usando apenas o embedding $e$. Aprende a imitar o encoder sem ver $y$.

O predictor é composto por **Multi-Head Global Attention** seguida de **Distribution Network**.

#### 2.3.1 Multi-Head Global Attention

Cada uma das $K$ heads produz uma representação global do mercado $h_{\text{muti}}^{(k)} \in \mathbb{R}^H$. Dentro de cada head:

$$k^{(i)} = w_{\text{key}} \, e^{(i)}, \qquad v^{(i)} = w_{\text{value}} \, e^{(i)}$$

$$a_{\text{att}}^{(i)} = \frac{\max\!\left(0, \dfrac{q \, k^{(i)T}}{\|q\|_2 \cdot \|k^{(i)}\|_2}\right)}{\sum_{i=1}^{N} \max\!\left(0, \dfrac{q \, k^{(i)T}}{\|q\|_2 \cdot \|k^{(i)}\|_2}\right)}$$

$$h_{\text{att}} = \sum_{i=1}^{N} a_{\text{att}}^{(i)} \, v^{(i)}$$

Três detalhes não óbvios:

O query $q \in \mathbb{R}^H$ é um **`nn.Parameter` aprendível**, não derivado dos dados. Cada head tem seu próprio $q$, que o treino move no espaço de embeddings para consultar aspectos distintos do mercado.

A atenção usa **similaridade cosseno com ReLU** no numerador, não o softmax padrão da literatura de transformers. Scores negativos são zerados; a normalização é pela soma dos scores positivos.

As $K$ heads são independentes, cada uma com seus próprios $w_{\text{key}}$, $w_{\text{value}}$ e $q$. Concatenam-se em:

$$h_{\text{muti}} = \text{Concat}\left(h_{\text{att}}^{(1)}, \dots, h_{\text{att}}^{(K)}\right) \in \mathbb{R}^{K \times H}$$

**Parâmetros por head.** $w_{\text{key}}, w_{\text{value}} \in \mathbb{R}^{H \times H}$, $q \in \mathbb{R}^H$.

#### 2.3.2 Distribution Network ($\pi_{\text{prior}}$)

Aplica a mesma estrutura da alpha layer (Seção 2.4.1), mas sobre $h_{\text{muti}}^{(k)}$, produzindo um par $(\mu_{\text{prior}}^{(k)}, \sigma_{\text{prior}}^{(k)})$ por head:

$$h_{\text{prior}}^{(k)} = \text{LeakyReLU}\left(w_{\text{pri}} \, h_{\text{muti}}^{(k)} + b_{\text{pri}}\right)$$

$$\mu_{\text{prior}}^{(k)} = w_{\mu} \, h_{\text{prior}}^{(k)} + b_{\mu}$$

$$\sigma_{\text{prior}}^{(k)} = \text{Softplus}\left(w_{\sigma} \, h_{\text{prior}}^{(k)} + b_{\sigma}\right)$$

Os pesos $w_{\text{pri}}, w_\mu, w_\sigma$ são compartilhados entre heads (cada head aplica as mesmas operações ao seu $h_{\text{muti}}^{(k)}$).

**Contrato do predictor como um todo.**
- Input: `e` de shape `(N, H)`.
- Output: `(mu_prior, sigma_prior)`, ambos de shape `(K,)`.

### 2.4 Factor Decoder ($\phi_{\text{dec}}$)

**Papel.** Converter uma distribuição de fatores $(\mu_z, \sigma_z)$ em uma distribuição preditiva de retornos $(\mu_y, \sigma_y)$, usando o embedding $e$ para calcular $\alpha$ e $\beta$ de cada ticker.

O decoder tem dois submódulos: **Alpha Layer** e **Beta Layer**. Ambos operam sobre $e$ — a informação de fatores entra apenas na composição final.

#### 2.4.1 Alpha Layer ($\pi_{\text{alpha}}$)

Produz o retorno idiossincrático como gaussiano $\alpha^{(i)} \sim \mathcal{N}(\mu_\alpha^{(i)}, (\sigma_\alpha^{(i)})^2)$:

$$h_\alpha^{(i)} = \text{LeakyReLU}\left(w_\alpha \, e^{(i)} + b_\alpha\right)$$

$$\mu_\alpha^{(i)} = w_{\alpha_\mu} \, h_\alpha^{(i)} + b_{\alpha_\mu}$$

$$\sigma_\alpha^{(i)} = \text{Softplus}\left(w_{\alpha_\sigma} \, h_\alpha^{(i)} + b_{\alpha_\sigma}\right)$$

**Parâmetros.** $w_\alpha \in \mathbb{R}^{H \times H}$, $w_{\alpha_\mu}, w_{\alpha_\sigma} \in \mathbb{R}^{H \times 1}$, biases correspondentes.

**Contrato.** Input `e` de shape `(N, H)`; output `(mu_alpha, sigma_alpha)` de shape `(N,)`.

#### 2.4.2 Beta Layer ($\phi_\beta$)

Linear sem ativação:

$$\beta^{(i)} = w_\beta \, e^{(i)} + b_\beta$$

**Parâmetros.** $w_\beta \in \mathbb{R}^{H \times K}$, $b_\beta \in \mathbb{R}^K$.

**Contrato.** Input `e` de shape `(N, H)`; output `beta` de shape `(N, K)`.

#### 2.4.3 Composição analítica do retorno

Aqui reside o aspecto mais importante do decoder: **ele não amostra**. Como $\alpha$ e $z$ são gaussianos independentes e $\beta$ é determinístico, a soma $\hat{y}^{(i)} = \alpha^{(i)} + \sum_k \beta^{(i,k)} z^{(k)}$ é também gaussiana, com momentos fechados:

$$\mu_y^{(i)} = \mu_\alpha^{(i)} + \sum_{k=1}^{K} \beta^{(i,k)} \mu_z^{(k)}$$

$$\sigma_y^{(i)} = \sqrt{\left(\sigma_\alpha^{(i)}\right)^2 + \sum_{k=1}^{K} \left(\beta^{(i,k)}\right)^2 \left(\sigma_z^{(k)}\right)^2}$$

O decoder recebe $(\mu_z, \sigma_z)$ como distribuição e devolve $(\mu_y, \sigma_y)$ também como distribuição — sem reparameterization trick, sem Monte Carlo. A log-verossimilhança gaussiana do $y$ observado pode ser calculada em forma fechada a partir de $(\mu_y, \sigma_y)$.

Esse desenho tem duas consequências práticas: o forward pass de treino é determinístico (dado os pesos), e a variância preditiva $\sigma_y$ tem interpretação estrutural (decomposta em parcela idiossincrática e parcela fatorial).

**Contrato do decoder como um todo.**
- Input: `(mu_z, sigma_z)` de shape `(K,)`, `e` de shape `(N, H)`.
- Output: `(mu_y, sigma_y)` de shape `(N,)`.

---

## 3. Fluxos de execução

### 3.1 Fluxo de treino

Dado um batch $(x, y)$ correspondendo a uma data $s$:

1. $e \leftarrow \phi_{\text{feat}}(x)$ — embedding compartilhado, shape `(N, H)`.
2. $(\mu_{\text{post}}, \sigma_{\text{post}}) \leftarrow \phi_{\text{enc}}(y, e)$ — posterior dos fatores, shape `(K,)`.
3. $(\mu_{\text{prior}}, \sigma_{\text{prior}}) \leftarrow \phi_{\text{pred}}(e)$ — prior dos fatores, shape `(K,)`.
4. $(\mu_y, \sigma_y) \leftarrow \phi_{\text{dec}}(\mu_{\text{post}}, \sigma_{\text{post}}, e)$ — distribuição reconstruída via posterior.
5. Loss de reconstrução: NLL gaussiana de $y$ sob $\mathcal{N}(\mu_y, \sigma_y^2)$.
6. Loss KL: $\text{KL}(q_{\text{post}} \,\|\, p_{\text{prior}})$ em forma fechada.
7. Loss total: $\mathcal{L}_{\text{recon}} + \gamma \cdot \mathcal{L}_{\text{KL}}$.

A **reconstrução sempre usa a posterior**, nunca a prior. O papel do KL é empurrar a prior para perto da posterior; a prior nunca toca o decoder durante o treino.

### 3.2 Fluxo de inferência

Dado apenas $x$:

1. $e \leftarrow \phi_{\text{feat}}(x)$.
2. $(\mu_{\text{prior}}, \sigma_{\text{prior}}) \leftarrow \phi_{\text{pred}}(e)$.
3. $(\mu_{\text{pred}}, \sigma_{\text{pred}}) \leftarrow \phi_{\text{dec}}(\mu_{\text{prior}}, \sigma_{\text{prior}}, e)$.

A média $\mu_{\text{pred}}^{(i)}$ é usada para rankear ações; o desvio $\sigma_{\text{pred}}^{(i)}$ alimenta estratégias risk-adjusted (TDrisk).

---

## 4. Função objetivo

A loss tem forma:

$$\mathcal{L}(x, y) = -\frac{1}{N} \sum_{i=1}^{N} \log P_{\phi_{\text{dec}}}\!\left(\hat{y}_{\text{rec}}^{(i)} = y^{(i)} \,\Big|\, x, z_{\text{post}}\right) + \gamma \cdot \text{KL}\!\left(P_{\phi_{\text{enc}}}(z | x, y) \;\Big\|\; P_{\phi_{\text{pred}}}(z | x)\right)$$

### 4.1 Reconstrução

Com $(\mu_y, \sigma_y)$ produzidos pelo decoder via composição analítica (Seção 2.4.3), a NLL gaussiana por ticker é:

$$-\log P(y^{(i)} | \mu_y^{(i)}, \sigma_y^{(i)}) = \frac{1}{2} \log(2\pi (\sigma_y^{(i)})^2) + \frac{(y^{(i)} - \mu_y^{(i)})^2}{2 (\sigma_y^{(i)})^2}$$

Agrega-se com média sobre $i$.

### 4.2 KL analítica

Como posterior e prior são gaussianas diagonais, o KL é fechado. Para cada fator $k$ independente:

$$\text{KL}\!\left(\mathcal{N}(\mu_q, \sigma_q^2) \,\|\, \mathcal{N}(\mu_p, \sigma_p^2)\right) = \log\frac{\sigma_p}{\sigma_q} + \frac{\sigma_q^2 + (\mu_q - \mu_p)^2}{2 \sigma_p^2} - \frac{1}{2}$$

Com $\mu_q = \mu_{\text{post}}$, $\sigma_q = \sigma_{\text{post}}$, $\mu_p = \mu_{\text{prior}}$, $\sigma_p = \sigma_{\text{prior}}$. Soma-se sobre os $K$ fatores.

**Atenção à ordem dos argumentos.** No FactorVAE o posterior é $q$ (do encoder) e o prior é $p$ (do predictor). Inverter destrói o gradiente do predictor.

---

## 5. Formato esperado dos dados

Os dados serão fornecidos depois. O desenvolvedor deve tratar a camada de dados como um contrato: o que importa para o modelo é o tensor final que chega no `forward`.

### 5.1 Contrato de entrada do modelo

Para cada data $s$ nos conjuntos de treino/validação/teste, o `Dataset` devolve:

| Nome | Shape | Dtype | Descrição |
|------|-------|-------|-----------|
| `x` | `(N_s, T, C)` | `float32` | Características históricas dos $N_s$ tickers válidos, cobrindo os $T$ dias anteriores a $s$ |
| `y` | `(N_s,)` | `float32` | Retorno futuro de cada ticker |
| `mask` | `(N_s,)` | `bool` | `True` para tickers com dados completos |

**Retorno target.** $y^{(i)} = (p_{t+2}^{(i)} - p_{t+1}^{(i)}) / p_{t+1}^{(i)}$, onde $t$ é a data de predição. O lag de um dia previne look-ahead: em $t$, o modelo prevê o retorno de $t+1$ para $t+2$, usando características até $t$.

### 5.2 Contrato do arquivo processado

O desenvolvedor deve assumir que haverá, em `data/processed/`, arquivos parquet com layout mínimo:

- `features.parquet`: colunas `[date, ticker, feature_name, value]`.
- `returns.parquet`: colunas `[date, ticker, forward_return]`.
- `universe.parquet`: colunas `[date, ticker, is_valid]`.

A transformação `long → tensor (N, T, C)` acontece em `src/factorvae/data/dataset.py`. Enquanto os dados reais não chegam, esse módulo deve funcionar com um **stub sintético** que gere tensores com as shapes corretas e estrutura de fator conhecida ($y = \alpha + \beta z + \epsilon$), permitindo testar se o modelo consegue aprender algo antes dos dados reais chegarem.

### 5.3 Hiperparâmetros dimensionais

Valores de referência do paper, ajustáveis no config:

- $T = 20$
- $C$: depende das features finais
- $N_s$: variável por data
- $H = 20$
- $K = 8$ (testar 4, 8, 16)
- $M = 64$ (testar 32, 64)

---

## 6. Estrutura do repositório

```
factorvae-br/
│
├── README.md
├── pyproject.toml
├── .gitignore
│
├── config.yaml
│
├── src/
│   └── factorvae/
│       ├── __init__.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py
│       │   └── datamodule.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── feature_extractor.py
│       │   ├── factor_encoder.py
│       │   ├── factor_decoder.py
│       │   ├── factor_predictor.py
│       │   ├── factorvae.py
│       │   └── distributions.py
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   ├── lightning_module.py
│       │   └── losses.py
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   └── backtest.py
│       │
│       └── utils/
│           ├── __init__.py
│           └── seeding.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── backtest.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── results/
│   ├── checkpoints/
│   ├── predictions/
│   └── figures/
│
└── tests/
    ├── conftest.py
    ├── test_feature_extractor.py
    ├── test_distributions.py
    ├── test_portfolio_layer.py
    ├── test_factor_encoder.py
    ├── test_factor_predictor.py
    ├── test_factor_decoder.py
    ├── test_factorvae_integration.py
    ├── test_losses.py
    ├── test_training_step.py
    ├── test_inference.py
    ├── test_metrics.py
    └── test_no_lookahead.py
```

---

## 7. Conteúdo de cada arquivo

### 7.1 `config.yaml`

Arquivo único, YAML plano. Carregado em `scripts/train.py` via `yaml.safe_load`.

```yaml
data:
  processed_dir: data/processed
  train_start: "2010-01-01"
  train_end: "2017-12-31"
  val_start: "2018-01-01"
  val_end: "2018-12-31"
  test_start: "2019-01-01"
  test_end: "2024-12-31"
  sequence_length: 20

model:
  num_features: null           # preenchido após inspeção dos dados
  hidden_dim: 20
  num_factors: 8
  num_portfolios: 64
  leaky_relu_slope: 0.1

training:
  batch_size: 1                # uma cross-section por batch
  max_epochs: 50
  learning_rate: 1e-3
  gamma: 1.0                   # peso do KL
  seed: 42
  sigma_floor: 1e-6            # clamp mínimo para estabilidade

evaluation:
  top_k: 50
  drop_n: 5
  risk_aversion_eta: 1.0
```

### 7.2 `src/factorvae/data/`

**`dataset.py`**

Duas classes. `RealDataset` lê os parquets em `data/processed/` e constrói os tensores; pode ficar como stub com `NotImplementedError` até os dados reais chegarem. `SyntheticDataset` gera dados com estrutura de fator conhecida — fixar $K_{\text{true}}$, $\alpha$, $\beta$, amostrar $z \sim \mathcal{N}(0, I)$ e produzir $y = \alpha + \beta z + \epsilon$ com ruído controlado. Ambas devolvem a tupla `(x, y, mask)` conforme o contrato da Seção 5.1.

**`datamodule.py`**

`FactorVAEDataModule(pl.LightningDataModule)` que instancia os três splits temporais. `DataLoader` com `batch_size=1` (cada batch é uma cross-section inteira). No `setup()` inclui `assert train_end < val_start < val_end < test_start` para prevenir vazamento.

### 7.3 `src/factorvae/models/`

**`feature_extractor.py`**

Implementa as fórmulas da Seção 2.1.

```python
class FeatureExtractor(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int, leaky_slope: float):
        super().__init__()
        self.proj = nn.Linear(num_features, hidden_dim)
        self.act = nn.LeakyReLU(leaky_slope)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, T, C)
        h_proj = self.act(self.proj(x))        # (N, T, H)
        _, h_final = self.gru(h_proj)          # h_final: (1, N, H)
        return h_final.squeeze(0)              # (N, H)
```

**`factor_encoder.py`**

Dois submódulos internos: `PortfolioLayer` e `MappingLayer`. A ordem de dimensão do softmax é crítica:

```python
class PortfolioLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_portfolios: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_portfolios)

    def forward(self, y: Tensor, e: Tensor) -> Tensor:
        # y: (N,), e: (N, H)
        logits = self.linear(e)                # (N, M)
        weights = F.softmax(logits, dim=0)     # softmax cross-sectional!
        y_p = (y.unsqueeze(1) * weights).sum(dim=0)  # (M,)
        return y_p


class MappingLayer(nn.Module):
    def __init__(self, num_portfolios: int, num_factors: int):
        super().__init__()
        self.mu_head = nn.Linear(num_portfolios, num_factors)
        self.sigma_head = nn.Linear(num_portfolios, num_factors)

    def forward(self, y_p: Tensor) -> tuple[Tensor, Tensor]:
        mu = self.mu_head(y_p)                 # (K,)
        sigma = F.softplus(self.sigma_head(y_p))  # (K,)
        return mu, sigma


class FactorEncoder(nn.Module):
    def __init__(self, hidden_dim, num_portfolios, num_factors):
        super().__init__()
        self.portfolio = PortfolioLayer(hidden_dim, num_portfolios)
        self.mapping = MappingLayer(num_portfolios, num_factors)

    def forward(self, y: Tensor, e: Tensor) -> tuple[Tensor, Tensor]:
        y_p = self.portfolio(y, e)
        return self.mapping(y_p)
```

**`factor_decoder.py`**

Alpha e Beta layers, e a composição analítica via função pura em `distributions.py`.

```python
class AlphaLayer(nn.Module):
    def __init__(self, hidden_dim: int, leaky_slope: float):
        super().__init__()
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.LeakyReLU(leaky_slope)
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.sigma_head = nn.Linear(hidden_dim, 1)

    def forward(self, e: Tensor) -> tuple[Tensor, Tensor]:
        h = self.act(self.hidden(e))           # (N, H)
        mu = self.mu_head(h).squeeze(-1)       # (N,)
        sigma = F.softplus(self.sigma_head(h)).squeeze(-1)  # (N,)
        return mu, sigma


class BetaLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_factors: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_factors)

    def forward(self, e: Tensor) -> Tensor:
        return self.linear(e)                  # (N, K)


class FactorDecoder(nn.Module):
    def __init__(self, hidden_dim, num_factors, leaky_slope):
        super().__init__()
        self.alpha = AlphaLayer(hidden_dim, leaky_slope)
        self.beta = BetaLayer(hidden_dim, num_factors)

    def forward(
        self, mu_z: Tensor, sigma_z: Tensor, e: Tensor
    ) -> tuple[Tensor, Tensor]:
        mu_alpha, sigma_alpha = self.alpha(e)
        beta = self.beta(e)
        return compose_return(mu_alpha, sigma_alpha, beta, mu_z, sigma_z)
```

**`factor_predictor.py`**

Multi-head attention com query aprendível e distribution network head-wise.

```python
class SingleHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.q = nn.Parameter(torch.randn(hidden_dim) * 0.1)

    def forward(self, e: Tensor) -> Tensor:
        # e: (N, H)
        k = self.w_key(e)                      # (N, H)
        v = self.w_value(e)                    # (N, H)
        q_norm = self.q / (self.q.norm() + 1e-8)
        k_norm = k / (k.norm(dim=1, keepdim=True) + 1e-8)
        scores = F.relu((k_norm * q_norm).sum(dim=1))  # (N,), cos sim com ReLU
        scores = scores / (scores.sum() + 1e-8)        # normalização
        h_att = (scores.unsqueeze(1) * v).sum(dim=0)   # (H,)
        return h_att


class DistributionNetwork(nn.Module):
    """Compartilhada entre as K heads do predictor."""
    def __init__(self, hidden_dim: int, leaky_slope: float):
        super().__init__()
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.LeakyReLU(leaky_slope)
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.sigma_head = nn.Linear(hidden_dim, 1)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        # h: (K, H)
        hidden = self.act(self.hidden(h))      # (K, H)
        mu = self.mu_head(hidden).squeeze(-1)       # (K,)
        sigma = F.softplus(self.sigma_head(hidden)).squeeze(-1)  # (K,)
        return mu, sigma


class FactorPredictor(nn.Module):
    def __init__(self, hidden_dim, num_factors, leaky_slope):
        super().__init__()
        self.heads = nn.ModuleList([
            SingleHeadAttention(hidden_dim) for _ in range(num_factors)
        ])
        self.dist_net = DistributionNetwork(hidden_dim, leaky_slope)

    def forward(self, e: Tensor) -> tuple[Tensor, Tensor]:
        h_muti = torch.stack([head(e) for head in self.heads], dim=0)  # (K, H)
        return self.dist_net(h_muti)
```

**`factorvae.py`**

Orquestra os quatro módulos e expõe os dois modos de operação.

```python
class FactorVAE(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        m = config["model"]
        self.feature_extractor = FeatureExtractor(
            m["num_features"], m["hidden_dim"], m["leaky_relu_slope"])
        self.encoder = FactorEncoder(
            m["hidden_dim"], m["num_portfolios"], m["num_factors"])
        self.predictor = FactorPredictor(
            m["hidden_dim"], m["num_factors"], m["leaky_relu_slope"])
        self.decoder = FactorDecoder(
            m["hidden_dim"], m["num_factors"], m["leaky_relu_slope"])

    def forward_train(self, x: Tensor, y: Tensor) -> dict:
        e = self.feature_extractor(x)
        mu_post, sigma_post = self.encoder(y, e)
        mu_prior, sigma_prior = self.predictor(e)
        mu_y_rec, sigma_y_rec = self.decoder(mu_post, sigma_post, e)
        return {
            "mu_post": mu_post, "sigma_post": sigma_post,
            "mu_prior": mu_prior, "sigma_prior": sigma_prior,
            "mu_y_rec": mu_y_rec, "sigma_y_rec": sigma_y_rec,
        }

    def forward_predict(self, x: Tensor) -> tuple[Tensor, Tensor]:
        e = self.feature_extractor(x)
        mu_prior, sigma_prior = self.predictor(e)
        return self.decoder(mu_prior, sigma_prior, e)
```

Repare que `forward_predict` nunca chama `self.encoder` — este é um invariante testável.

**`distributions.py`**

Funções puras, sem `nn.Module`, testáveis isoladamente.

```python
def compose_return(
    mu_alpha: Tensor, sigma_alpha: Tensor,
    beta: Tensor, mu_z: Tensor, sigma_z: Tensor,
) -> tuple[Tensor, Tensor]:
    """Composição analítica das fórmulas da Eq. 12.
    Shapes: mu_alpha, sigma_alpha: (N,); beta: (N, K); mu_z, sigma_z: (K,).
    Retorna: (mu_y, sigma_y), ambos (N,).
    """
    mu_y = mu_alpha + beta @ mu_z                               # (N,)
    var_factor = (beta ** 2) @ (sigma_z ** 2)                   # (N,)
    sigma_y = torch.sqrt(sigma_alpha ** 2 + var_factor)
    return mu_y, sigma_y


def gaussian_nll(y: Tensor, mu: Tensor, sigma: Tensor, floor: float = 1e-6) -> Tensor:
    """NLL gaussiana, média sobre os elementos."""
    sigma = sigma.clamp(min=floor)
    log_term = torch.log(2 * math.pi * sigma ** 2)
    sq_term = (y - mu) ** 2 / (sigma ** 2)
    return 0.5 * (log_term + sq_term).mean()


def kl_gaussian_diagonal(
    mu_q: Tensor, sigma_q: Tensor,
    mu_p: Tensor, sigma_p: Tensor, floor: float = 1e-6,
) -> Tensor:
    """KL(N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2)), somado sobre dimensões."""
    sigma_q = sigma_q.clamp(min=floor)
    sigma_p = sigma_p.clamp(min=floor)
    term1 = torch.log(sigma_p / sigma_q)
    term2 = (sigma_q ** 2 + (mu_q - mu_p) ** 2) / (2 * sigma_p ** 2)
    return (term1 + term2 - 0.5).sum()
```

### 7.4 `src/factorvae/training/`

**`losses.py`**

Wrappers finos sobre `distributions.py` para clareza no training step.

```python
def reconstruction_loss(y, mu_y_rec, sigma_y_rec, floor):
    return gaussian_nll(y, mu_y_rec, sigma_y_rec, floor)

def kl_loss(mu_post, sigma_post, mu_prior, sigma_prior, floor):
    return kl_gaussian_diagonal(mu_post, sigma_post, mu_prior, sigma_prior, floor)
```

**`lightning_module.py`**

Loss decomposta, log separado de `loss_recon` e `loss_kl` para diagnóstico.

```python
class FactorVAELightning(pl.LightningModule):
    def __init__(self, model: FactorVAE, config: dict):
        super().__init__()
        self.model = model
        self.gamma = config["training"]["gamma"]
        self.lr = config["training"]["learning_rate"]
        self.sigma_floor = config["training"]["sigma_floor"]
        self.save_hyperparameters(config)

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        x, y = x.squeeze(0), y.squeeze(0)      # desempacota batch_size=1
        out = self.model.forward_train(x, y)
        l_recon = reconstruction_loss(y, out["mu_y_rec"], out["sigma_y_rec"],
                                       self.sigma_floor)
        l_kl = kl_loss(out["mu_post"], out["sigma_post"],
                       out["mu_prior"], out["sigma_prior"], self.sigma_floor)
        loss = l_recon + self.gamma * l_kl
        self.log_dict({"loss": loss, "loss_recon": l_recon, "loss_kl": l_kl})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        x, y = x.squeeze(0), y.squeeze(0)
        mu_pred, _ = self.model.forward_predict(x)
        rank_ic = compute_rank_ic(y, mu_pred)
        self.log("val_rank_ic", rank_ic)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
```

### 7.5 `src/factorvae/evaluation/`

**`metrics.py`**

```python
def compute_rank_ic(y_true: Tensor, y_pred: Tensor) -> float:
    """Spearman entre y_true e y_pred na cross-section."""
    rank_true = y_true.argsort().argsort().float()
    rank_pred = y_pred.argsort().argsort().float()
    return pearson_corr(rank_true, rank_pred).item()

def compute_rank_icir(rank_ics: list[float]) -> float:
    arr = np.array(rank_ics)
    return arr.mean() / arr.std()
```

**`backtest.py`**

Implementa TopK-Drop com parâmetros $k$, $n$ e taxa de corretagem. Variante TDrisk usa $\mu - \eta \sigma$ no ranking.

```python
def topk_drop_strategy(
    predictions: pd.DataFrame,  # [date, ticker, mu_pred, sigma_pred, y_true]
    k: int, n: int, eta: float = 0.0,
) -> pd.DataFrame:
    # score = mu - eta * sigma se eta > 0, senão só mu
    # Em cada data, seleciona top k; mantém k-n da anterior, troca os piores
    ...

def compute_performance_metrics(
    portfolio_returns: pd.Series, benchmark_returns: pd.Series,
) -> dict:
    # Retorna {annualized_return, sharpe, max_drawdown}
    ...
```

### 7.6 `scripts/`

**`train.py`**

Linear, ~40 linhas. Lê config, instancia datamodule, model, lightning module, trainer. Callbacks: `ModelCheckpoint(monitor="val_rank_ic", mode="max")`, `EarlyStopping(monitor="val_rank_ic", mode="max", patience=10)`.

**`evaluate.py`**

Carrega o melhor checkpoint, roda sobre test set, salva `results/predictions/predictions.parquet` com `[date, ticker, mu_pred, sigma_pred, y_true]`. Reporta Rank IC e Rank ICIR agregados.

**`backtest.py`**

Lê `predictions.parquet`, aplica `topk_drop_strategy` duas vezes (com $\eta=0$ e $\eta>0$), calcula métricas contra benchmark, salva tabela e curva em `results/figures/`.

---

## 8. Testes

Três objetivos: correção matemática dos componentes isolados, integridade do fluxo ponta-a-ponta, e detecção de vazamento temporal. Todos com `pytest`.

### 8.1 `conftest.py`

Fixtures sintéticas reutilizáveis:

```python
@pytest.fixture
def dims():
    return {"N": 100, "T": 20, "C": 10, "H": 16, "K": 4, "M": 32}

@pytest.fixture
def random_x(dims):
    torch.manual_seed(0)
    return torch.randn(dims["N"], dims["T"], dims["C"])

@pytest.fixture
def random_e(dims):
    return torch.randn(dims["N"], dims["H"])

@pytest.fixture
def synthetic_factor_batch(dims):
    """Dados com estrutura de fator conhecida, para teste de overfit."""
    torch.manual_seed(0)
    K_true = 3
    beta_true = torch.randn(dims["N"], K_true)
    z_true = torch.randn(K_true)
    alpha_true = 0.1 * torch.randn(dims["N"])
    y = alpha_true + beta_true @ z_true + 0.05 * torch.randn(dims["N"])
    x = torch.randn(dims["N"], dims["T"], dims["C"])
    return x, y
```

### 8.2 Testes de componentes isolados

**`test_feature_extractor.py`**
- Forward produz output de shape `(N, H)`.
- Gradiente flui para todos os parâmetros após `loss.backward()`.
- Invariância parcial: rodar com $N=50$ e $N=500$ não quebra.

**`test_distributions.py`** — matemática fundamental, cobre `distributions.py`.
- `gaussian_nll` bate com `-scipy.stats.norm.logpdf` para valores fixos (tolerância `1e-5`).
- `kl_gaussian_diagonal` bate com estimativa Monte Carlo (100k amostras, tolerância `1e-2`).
- `kl_gaussian_diagonal(p, p)` $\approx 0$.
- `kl_gaussian_diagonal` é não-negativo para distribuições aleatórias.
- `compose_return` produz shapes corretas.
- `compose_return`: variância composta bate com variância empírica de $\alpha + \beta z$ amostrados (100k amostras, tolerância `1e-2`).

**`test_portfolio_layer.py`** — o ponto mais sensível da arquitetura.
- Pesos somam exatamente 1 para cada portfólio $j$: `weights.sum(dim=0)` $\approx 1$.
- Pesos são não-negativos.
- $y_p$ tem shape `(M,)` independente de $N$ (testar com $N=50$ e $N=500$).
- `test_softmax_dim_correctness`: verifica explicitamente que o softmax é sobre stocks, não portfolios — construir um caso patológico onde errar a dimensão produz outputs visivelmente diferentes.

**`test_factor_encoder.py`**
- Output shapes: $(\mu_{\text{post}}, \sigma_{\text{post}})$ ambos `(K,)`.
- $\sigma_{\text{post}} > 0$ estritamente.
- Encoder é invariante ao número de stocks: $N=50$ e $N=500$ produzem shapes iguais na saída.

**`test_factor_predictor.py`**
- Output shapes: $(\mu_{\text{prior}}, \sigma_{\text{prior}})$ ambos `(K,)`.
- $\sigma_{\text{prior}} > 0$.
- Pesos de atenção somam 1 dentro de cada head.
- Query $q$ de cada head recebe gradiente após backward.

**`test_factor_decoder.py`**
- Output shapes: $(\mu_y, \sigma_y)$ ambos `(N,)`.
- $\sigma_y > 0$ estritamente.
- Para $(\mu_z, \sigma_z)$ e $(\mu_\alpha, \sigma_\alpha)$ conhecidos e $\beta$ fixo, $\mu_y$ e $\sigma_y$ batem com momentos empíricos de muitas amostras de $\alpha + \beta z$.

### 8.3 Testes de integração

**`test_factorvae_integration.py`**
- `forward_train`: dict tem as 6 chaves esperadas com shapes corretas.
- `forward_train`: gradiente flui para parâmetros de todos os quatro módulos após `loss.backward()`.
- `forward_predict`: roda sem erro, output `(mu_pred, sigma_pred)` tem shape `(N,)`.
- `test_encoder_unused_in_predict`: após `forward_predict(x).sum().backward()`, nenhum parâmetro de `model.encoder` tem gradiente (todos com `.grad is None` ou `.grad == 0`).

**`test_losses.py`**
- `reconstruction_loss` e `kl_loss` são finitas para inputs válidos.
- `test_overfit_single_batch`: treinar por 200 steps em um batch sintético da fixture `synthetic_factor_batch` reduz a loss total em $\geq 50\%$. Gate crítico.

**`test_training_step.py`**
- `trainer.fit(..., fast_dev_run=True)` completa sem erro com `SyntheticDataset`.
- Após um epoch, `val_rank_ic` aparece nos logs.
- Salvar e recarregar checkpoint produz `forward_predict` idêntico.

### 8.4 Testes de invariantes operacionais

**`test_inference.py`**
- `test_predict_robust_to_y_corruption`: perturbar $y$ no dataloader com ruído puro durante inferência não altera `mu_pred` nem `sigma_pred`. Se alterar, há vazamento.
- `test_deterministic_inference`: com seeds fixos, `forward_predict(x)` é reprodutível.

**`test_metrics.py`**
- `rank_ic(y, y) == 1.0`.
- `rank_ic(y, -y) == -1.0`.
- Para $y$ aleatório e grande $N$, `rank_ic` $\approx 0$.

**`test_no_lookahead.py`**
- Nenhuma data do split de treino é $\geq$ `train_end`.
- Nenhuma feature em $x_s$ tem timestamp posterior a $s$.
- O target $y_s$ usa preços em $s+1$ e $s+2$ exclusivamente.

---

## 9. Ordem de implementação

A ordem abaixo minimiza retrabalho. Cada etapa só avança após seus testes passarem.

1. **`distributions.py` + `test_distributions.py`** — matemática base. Sem isso, tudo que vier depois propaga bugs silenciosos.
2. **`feature_extractor.py` + `test_feature_extractor.py`** — trivial, valida setup Lightning.
3. **`factor_encoder.py` + `test_portfolio_layer.py` + `test_factor_encoder.py`** — risco alto de bug no softmax.
4. **`factor_decoder.py` + `test_factor_decoder.py`**.
5. **`factor_predictor.py` + `test_factor_predictor.py`**.
6. **`factorvae.py` + `test_factorvae_integration.py`** — integração dos quatro módulos.
7. **`losses.py` + `test_losses.py`** — crítico: `test_overfit_single_batch` é o gate antes de qualquer coisa com dados reais.
8. **`dataset.py` com `SyntheticDataset` funcional** — desbloqueia treino end-to-end.
9. **`lightning_module.py` + `datamodule.py` + `test_training_step.py` + `test_inference.py`**.
10. **`scripts/train.py`** — primeiro run real em dados sintéticos.
11. **`metrics.py` + `backtest.py` + testes associados**.
12. **Integração com dados reais** — `RealDataset`, `scripts/evaluate.py`, `scripts/backtest.py`, `test_no_lookahead.py`.

---

## 10. Pontos de atenção

Três armadilhas específicas do FactorVAE que o agente pode introduzir sem perceber:

O **softmax do portfolio layer** deve operar sobre a dimensão dos stocks (`dim=0` no layout `(N, M)`), não sobre a dos portfólios. O teste `test_softmax_dim_correctness` existe especificamente para isso. Softmax na dimensão errada ainda compila e produz números, mas o sinal do modelo vira ruído.

O **KL tem posterior e prior com papéis específicos**: $q$ é a posterior do encoder, $p$ é a prior *aprendida* do predictor. No VAE padrão, $p$ é fixo em $\mathcal{N}(0, I)$; aqui não. A forma fechada é a mesma, mas passar os argumentos trocados inverte o gradiente do predictor — ele deixa de receber sinal para se aproximar da posterior.

A **composição de variância no decoder usa $\beta^2$, não $|\beta|$**. Parece óbvio, mas é comum perder o quadrado na hora de escrever. O teste `test_decoder_matches_sampled_moments` captura essa classe de erro comparando com amostragem.

Dois pontos estruturais menos óbvios que também merecem atenção:

O **decoder é analítico, não amostral**. Alguns desenvolvedores, por hábito de VAE padrão, implementam o decoder com reparameterization trick em $z_{\text{post}}$. Isso funciona mas é mais ruidoso e desnecessário: a Eq. 12 do paper dá os momentos fechados. A implementação analítica é mais rápida e produz gradientes mais limpos.

O **embedding $e$ é compartilhado entre encoder, predictor e decoder**. Se o feature extractor for instanciado separadamente em cada módulo, os três terão representações independentes e o modelo não treinará como pretendido. `FactorVAE.__init__` instancia um único `FeatureExtractor` e os forward passes reutilizam o mesmo `e`.
