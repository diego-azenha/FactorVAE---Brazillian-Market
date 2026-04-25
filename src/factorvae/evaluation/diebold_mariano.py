"""
Diebold-Mariano test for comparing predictive accuracy of two models.

Reference: Diebold & Mariano (1995), "Comparing Predictive Accuracy",
           JBES 13(3), pp. 253-263.

Implementation details:
  - Loss differential: d_t = IC_A(t) - IC_B(t)  (or any per-date scalar metric)
  - HAC variance via Newey-West (bandwidth h = floor(T^(1/3)), recommended
    for T ≈ 1000 at block sizes consistent with daily IC autocorrelation)
  - Block bootstrap as a nonparametric cross-check (relaxes stationarity /
    short-memory assumptions implicit in the asymptotic DM)

Methodological note (for dissertation defence):
  DM (1995) derives the statistic for point forecasts under MSE or MAE.
  Applying it to a per-date skill metric (Rank IC) is a common adaptation
  in empirical asset pricing (see Harvey, Liu & Zhu 2016 for the broader
  debate on significance testing in finance). The valid interpretation is:
  "Assuming d_t = IC_A(t) - IC_B(t) is covariance-stationary with short
  memory, we test H0: E[d_t] = 0 vs. the stated alternative."
  The block bootstrap serves as a nonparametric cross-check.

Usage:
    from factorvae.evaluation.diebold_mariano import diebold_mariano, block_bootstrap_dm

    # Load per-date Rank IC series (e.g. from a CSV)
    ic_a = pd.Series(...)  # model A IC per date
    ic_b = pd.Series(...)  # model B IC per date

    result = diebold_mariano(ic_a, ic_b, alternative="greater")
    print(result)

    bs_pval = block_bootstrap_dm(ic_a, ic_b, alternative="greater")
    print(f"Bootstrap p-value: {bs_pval:.4f}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DMResult:
    """Result of a Diebold-Mariano test."""

    dm_stat: float
    """DM test statistic (t-distributed asymptotically under H0)."""

    pvalue: float
    """Asymptotic one- or two-sided p-value."""

    alternative: str
    """Alternative hypothesis: 'greater', 'less', or 'two-sided'."""

    n: int
    """Number of forecast periods used."""

    mean_diff: float
    """Sample mean of the loss differential d_t = IC_A - IC_B."""

    bandwidth: int
    """Newey-West bandwidth used for HAC variance estimation."""

    def __str__(self) -> str:
        return (
            f"DM test: stat={self.dm_stat:+.4f}, p-value={self.pvalue:.4f}, "
            f"alternative='{self.alternative}', n={self.n}, "
            f"mean(d)={self.mean_diff:+.4f}, h={self.bandwidth}"
        )


def _newey_west_variance(d: np.ndarray, h: int) -> float:
    """
    Newey-West (1987) HAC variance estimator for the sample mean of d.

    V_NW = gamma_0 + 2 * sum_{j=1}^{h} (1 - j/(h+1)) * gamma_j

    where gamma_j = (1/T) * sum_{t=j+1}^{T} d_t * d_{t-j}
    (without mean-centering — DM recommend using raw d, not d-dbar).

    Args:
        d: Array of loss differentials, length T.
        h: Bandwidth (number of lags). Recommended: floor(T^(1/3)).

    Returns:
        Estimated long-run variance V_NW (variance of the sample mean × T).
        Caller should divide by T to get variance of dbar.
    """
    T = len(d)
    gamma_0 = float(np.mean(d ** 2))
    lrv = gamma_0
    for j in range(1, h + 1):
        gamma_j = float(np.mean(d[j:] * d[:-j]))
        weight   = 1.0 - j / (h + 1)
        lrv     += 2.0 * weight * gamma_j
    # LRV is the long-run variance; variance of dbar = LRV / T
    # but DM stat uses sqrt(LRV / T) in the denominator
    return lrv


def diebold_mariano(
    ic_a: pd.Series | np.ndarray,
    ic_b: pd.Series | np.ndarray,
    alternative: str = "greater",
    bandwidth: int | None = None,
) -> DMResult:
    """
    Diebold-Mariano test on per-date loss differentials.

    Tests H0: E[IC_A(t) - IC_B(t)] = 0 against the stated alternative.

    Args:
        ic_a:        Per-date Rank IC series for model A (or any scalar metric).
        ic_b:        Per-date Rank IC series for model B.
        alternative: One of 'greater'   (H1: A is better than B, i.e. mean(d) > 0)
                              'less'     (H1: A is worse than B)
                              'two-sided' (H1: A ≠ B)
        bandwidth:   Newey-West lag truncation h. Defaults to floor(T^(1/3)).

    Returns:
        DMResult dataclass.

    Raises:
        ValueError: If alternative is invalid or arrays have different lengths.
    """
    valid_alts = {"greater", "less", "two-sided"}
    if alternative not in valid_alts:
        raise ValueError(f"alternative must be one of {valid_alts}, got '{alternative}'")

    d = np.asarray(ic_a, dtype=float) - np.asarray(ic_b, dtype=float)
    d = d[~np.isnan(d)]
    T = len(d)

    if T < 10:
        raise ValueError(f"Too few observations ({T}) for DM test (need ≥ 10)")

    h = bandwidth if bandwidth is not None else math.floor(T ** (1.0 / 3.0))
    lrv = _newey_west_variance(d, h)

    dbar = float(np.mean(d))
    se   = math.sqrt(max(lrv, 1e-16) / T)
    stat = dbar / se

    # Asymptotic N(0,1) critical values (DM 1995 use standard normal)
    from scipy import stats as sp_stats
    if alternative == "greater":
        pvalue = float(sp_stats.norm.sf(stat))     # P(Z > stat)
    elif alternative == "less":
        pvalue = float(sp_stats.norm.cdf(stat))    # P(Z < stat)
    else:  # two-sided
        pvalue = float(2.0 * sp_stats.norm.sf(abs(stat)))

    return DMResult(
        dm_stat=stat,
        pvalue=pvalue,
        alternative=alternative,
        n=T,
        mean_diff=dbar,
        bandwidth=h,
    )


def block_bootstrap_dm(
    ic_a: pd.Series | np.ndarray,
    ic_b: pd.Series | np.ndarray,
    alternative: str = "greater",
    block_size: int | None = None,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> float:
    """
    Block bootstrap p-value for the DM test.

    Resamples the loss differential series d_t in contiguous blocks
    (preserving short-range autocorrelation), computes the mean under each
    resample, and estimates the p-value nonparametrically.

    Serves as a nonparametric cross-check that relaxes the stationarity and
    short-memory assumptions of the asymptotic DM statistic.

    Args:
        ic_a:        Per-date Rank IC series for model A.
        ic_b:        Per-date Rank IC series for model B.
        alternative: 'greater', 'less', or 'two-sided'.
        block_size:  Block length. Defaults to floor(T^(1/3)) (same as NW bandwidth).
        n_bootstrap: Number of bootstrap replications.
        seed:        RNG seed for reproducibility.

    Returns:
        Bootstrap p-value as a float in [0, 1].
    """
    d = np.asarray(ic_a, dtype=float) - np.asarray(ic_b, dtype=float)
    d = d[~np.isnan(d)]
    T = len(d)

    if T < 10:
        raise ValueError(f"Too few observations ({T}) for block bootstrap")

    b = block_size if block_size is not None else max(1, math.floor(T ** (1.0 / 3.0)))
    rng = np.random.default_rng(seed)
    dbar_obs = float(np.mean(d))

    # Center the differential under H0 (shift mean to zero)
    d_centered = d - dbar_obs

    # Generate bootstrap statistics
    boot_means = np.empty(n_bootstrap)
    n_blocks = math.ceil(T / b)
    for i in range(n_bootstrap):
        starts   = rng.integers(0, T, size=n_blocks)
        blocks   = [d_centered[s : s + b] for s in starts]
        resample = np.concatenate(blocks)[:T]
        boot_means[i] = float(np.mean(resample))

    # P-value
    if alternative == "greater":
        pvalue = float(np.mean(boot_means >= dbar_obs))
    elif alternative == "less":
        pvalue = float(np.mean(boot_means <= dbar_obs))
    else:  # two-sided
        pvalue = float(np.mean(np.abs(boot_means) >= abs(dbar_obs)))

    return pvalue


def dm_summary(
    ic_a: pd.Series | np.ndarray,
    ic_b: pd.Series | np.ndarray,
    name_a: str = "Model A",
    name_b: str = "Model B",
    alternative: str = "greater",
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run both the asymptotic DM test and block bootstrap, return a summary DataFrame.

    Args:
        ic_a, ic_b:    Per-date Rank IC series.
        name_a, name_b: Names for display.
        alternative:   'greater', 'less', or 'two-sided'.
        n_bootstrap:   Bootstrap replications.
        seed:          RNG seed.

    Returns:
        Single-row DataFrame with columns:
          model_a, model_b, alternative, n, mean_ic_a, mean_ic_b, mean_diff,
          dm_stat, dm_pvalue, bootstrap_pvalue, bandwidth
    """
    d     = np.asarray(ic_a, dtype=float) - np.asarray(ic_b, dtype=float)
    valid = ~np.isnan(d)
    result = diebold_mariano(ic_a, ic_b, alternative=alternative)
    bs_pv  = block_bootstrap_dm(
        ic_a, ic_b, alternative=alternative, n_bootstrap=n_bootstrap, seed=seed
    )

    ic_a_arr = np.asarray(ic_a, dtype=float)[valid]
    ic_b_arr = np.asarray(ic_b, dtype=float)[valid]

    return pd.DataFrame([{
        "model_a":           name_a,
        "model_b":           name_b,
        "alternative":       alternative,
        "n":                 result.n,
        "mean_ic_a":         float(np.nanmean(ic_a_arr)),
        "mean_ic_b":         float(np.nanmean(ic_b_arr)),
        "mean_diff":         result.mean_diff,
        "dm_stat":           result.dm_stat,
        "dm_pvalue":         result.pvalue,
        "bootstrap_pvalue":  bs_pv,
        "bandwidth":         result.bandwidth,
    }])
