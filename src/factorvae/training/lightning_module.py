"""
FactorVAELightning — PyTorch Lightning training module.

Logs loss_recon and loss_kl separately in addition to total loss.
Validation logs rank_ic per step and computes mean over the epoch.
"""

from __future__ import annotations

import lightning as L
import torch
import torch.optim as optim
from torch import Tensor

from factorvae.evaluation.metrics import compute_rank_ic
from factorvae.models.factorvae import FactorVAE
from factorvae.training.losses import kl_loss, reconstruction_loss


class FactorVAELightning(L.LightningModule):
    def __init__(self, model: FactorVAE, config: dict):
        super().__init__()
        self.model = model
        self.lr = config["training"]["learning_rate"]
        self.gamma = config["training"]["gamma"]
        self.floor = config["training"]["sigma_floor"]
        self.save_hyperparameters(config)
        self._val_rank_ics: list[float] = []

    # ─── Training ───────────────────────────────────────────

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        # DataLoader wraps each sample in a batch dim → squeeze it
        x, y, mask = batch
        x = x.squeeze(0)   # (N, T, C)
        y = y.squeeze(0)   # (N,)

        out = self.model.forward_train(x, y)
        loss_r = reconstruction_loss(y, out["mu_y_rec"], out["sigma_y_rec"], self.floor)
        loss_k = kl_loss(out["mu_post"], out["sigma_post"], out["mu_prior"], out["sigma_prior"], self.floor)
        loss = loss_r + self.gamma * loss_k

        self.log("train_loss_recon", loss_r, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_loss_kl",    loss_k, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_loss",       loss,   on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # ─── Validation ─────────────────────────────────────────

    def validation_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int) -> None:
        x, y, mask = batch
        x = x.squeeze(0)
        y = y.squeeze(0)

        mu_pred, _ = self.model.forward_predict(x)
        rank_ic = compute_rank_ic(y, mu_pred)
        self._val_rank_ics.append(rank_ic)

    def on_validation_epoch_end(self) -> None:
        if self._val_rank_ics:
            mean_rank_ic = sum(self._val_rank_ics) / len(self._val_rank_ics)
            self.log("val_rank_ic", mean_rank_ic, prog_bar=True)
            self._val_rank_ics.clear()

    # ─── Optimizer ──────────────────────────────────────────

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
