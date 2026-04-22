"""
Shared fixtures for all FactorVAE tests.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture
def dims() -> dict[str, int]:
    return {"N": 100, "T": 20, "C": 10, "H": 16, "K": 4, "M": 32}


@pytest.fixture
def random_x(dims: dict[str, int]) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(dims["N"], dims["T"], dims["C"])


@pytest.fixture
def random_e(dims: dict[str, int]) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(dims["N"], dims["H"])


@pytest.fixture
def synthetic_factor_batch(dims: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Synthetic data with known factor structure: y = alpha + beta @ z + noise.
    Used in overfit tests to verify the model can recover signal.
    """
    torch.manual_seed(42)
    N, T, C, K = dims["N"], dims["T"], dims["C"], dims["K"]

    alpha_true = torch.randn(N) * 0.02
    beta_true = torch.randn(N, K)
    z_true = torch.randn(K)
    noise = torch.randn(N) * 0.01

    y = alpha_true + beta_true @ z_true + noise  # (N,)

    # x is a simple pattern correlated with the factors (makes overfitting possible)
    x = torch.randn(N, T, C)
    # Inject factor signal into the first K features
    for k in range(K):
        x[:, -1, k] += beta_true[:, k] * z_true[k]

    return x, y
