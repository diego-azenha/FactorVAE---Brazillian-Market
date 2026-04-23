"""
FactorVAE — main model class.

Orchestrates the four modules around a SINGLE shared FeatureExtractor instance.
The embedding e is computed once per forward pass and reused by all downstream modules.

Two modes:
  - forward_train(x, y): uses all four modules; returns a dict with
      e, mu_post, sigma_post, mu_prior, sigma_prior, mu_y_rec, sigma_y_rec
  - forward_predict(x): encoder is completely skipped; returns (mu_y, sigma_y)
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from factorvae.models.factor_decoder import FactorDecoder
from factorvae.models.factor_encoder import FactorEncoder
from factorvae.models.factor_predictor import FactorPredictor
from factorvae.models.feature_extractor import FeatureExtractor


class FactorVAE(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        C = config["model"]["num_features"]
        H = config["model"]["hidden_dim"]
        K = config["model"]["num_factors"]
        M = config["model"]["num_portfolios"]
        slope = config["model"]["leaky_relu_slope"]
        macro_dim = config["model"].get("macro_dim", 0)

        # Single shared feature extractor — DO NOT instantiate one per module
        self.feature_extractor = FeatureExtractor(C, H, slope)
        self.encoder = FactorEncoder(H, M, K)
        self.predictor = FactorPredictor(H, K, slope, macro_dim=macro_dim)
        self.decoder = FactorDecoder(H, K, slope)

    def forward_train(self, x: Tensor, y: Tensor, m: Tensor | None = None) -> dict:
        """
        Training forward pass.

        Args:
            x: (N, T, C)
            y: (N,)  future returns (oracle signal)
            m: (macro_dim,) optional macro vector

        Returns dict with keys:
            e            : (N, H)
            mu_post      : (K,)
            sigma_post   : (K,)
            mu_prior     : (K,)
            sigma_prior  : (K,)
            mu_y_rec     : (N,)  reconstructed return mean via posterior
            sigma_y_rec  : (N,)  reconstructed return std  via posterior
        """
        e = self.feature_extractor(x)                          # (N, H)
        mu_post, sigma_post = self.encoder(y, e)               # (K,), (K,)
        mu_prior, sigma_prior = self.predictor(e, m=m)         # (K,), (K,)
        # Reconstruction uses POSTERIOR — never the prior
        mu_y_rec, sigma_y_rec = self.decoder(mu_post, sigma_post, e)  # (N,), (N,)
        return {
            "e": e,
            "mu_post": mu_post,
            "sigma_post": sigma_post,
            "mu_prior": mu_prior,
            "sigma_prior": sigma_prior,
            "mu_y_rec": mu_y_rec,
            "sigma_y_rec": sigma_y_rec,
        }

    def forward_predict(self, x: Tensor, m: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """
        Inference forward pass. Encoder is NOT called.

        Args:
            x: (N, T, C)
            m: (macro_dim,) optional macro vector

        Returns:
            mu_pred:    (N,)
            sigma_pred: (N,)
        """
        e = self.feature_extractor(x)
        mu_prior, sigma_prior = self.predictor(e, m=m)
        return self.decoder(mu_prior, sigma_prior, e)
