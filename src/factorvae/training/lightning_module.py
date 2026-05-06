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
        self.kl_warmup_steps = config["training"].get("kl_warmup_steps", 0)
        self.kl_free_bits = config["training"].get("kl_free_bits", 0.0)
        self.weight_decay = config["training"].get("weight_decay", 0.0)
        self.save_hyperparameters(config)
        self._val_rank_ics: list[float] = []
        # Buffers for per-factor distribution snapshot (filled in training_step)
        self._prior_mu_buf:  list[Tensor] = []
        self._prior_sig_buf: list[Tensor] = []
        self._post_mu_buf:   list[Tensor] = []
        self._post_sig_buf:  list[Tensor] = []

    # ─── Training ───────────────────────────────────────────

    def training_step(self, batch, batch_idx: int) -> Tensor:
        # DataLoader wraps each sample in a batch dim → squeeze it
        x, y, mask = batch
        x = x.squeeze(0)   # (N, T, C)
        y = y.squeeze(0)   # (N,)

        out = self.model.forward_train(x, y)
        loss_r = reconstruction_loss(y, out["mu_y_rec"], out["sigma_y_rec"], self.floor)
        loss_k = kl_loss(out["mu_post"], out["sigma_post"], out["mu_prior"], out["sigma_prior"],
                         self.floor, self.kl_free_bits)

        # Linear KL warmup: ramp gamma from 0 → gamma over kl_warmup_steps
        if self.kl_warmup_steps > 0:
            gamma_eff = self.gamma * min(1.0, self.global_step / self.kl_warmup_steps)
        else:
            gamma_eff = self.gamma

        loss = loss_r + gamma_eff * loss_k

        self.log("train_loss_recon", loss_r,    on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_loss_kl",    loss_k,    on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_gamma_eff",  gamma_eff, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_loss",       loss,      on_step=False, on_epoch=True, prog_bar=True)

        # Accumulate per-factor distribution params for snapshot visualization
        self._prior_mu_buf.append(out["mu_prior"].detach().cpu())
        self._prior_sig_buf.append(out["sigma_prior"].detach().cpu())
        self._post_mu_buf.append(out["mu_post"].detach().cpu())
        self._post_sig_buf.append(out["sigma_post"].detach().cpu())

        return loss

    def on_train_epoch_end(self) -> None:
        """Average per-factor distribution params over the epoch and log to CSV."""
        if not self._prior_mu_buf:
            return
        # Stack: (n_batches, K) → mean over batches → (K,)
        prior_mu  = torch.stack(self._prior_mu_buf).mean(0)   # (K,)
        prior_sig = torch.stack(self._prior_sig_buf).mean(0)  # (K,)
        post_mu   = torch.stack(self._post_mu_buf).mean(0)    # (K,)
        post_sig  = torch.stack(self._post_sig_buf).mean(0)   # (K,)

        K = prior_mu.shape[0]
        for k in range(K):
            self.log(f"train_prior_mu_{k}",  prior_mu[k].item(),  on_step=False, on_epoch=True)
            self.log(f"train_prior_sig_{k}", prior_sig[k].item(), on_step=False, on_epoch=True)
            self.log(f"train_post_mu_{k}",   post_mu[k].item(),   on_step=False, on_epoch=True)
            self.log(f"train_post_sig_{k}",  post_sig[k].item(),  on_step=False, on_epoch=True)

        self._prior_mu_buf.clear()
        self._prior_sig_buf.clear()
        self._post_mu_buf.clear()
        self._post_sig_buf.clear()

    # ─── Validation ─────────────────────────────────────────

    def validation_step(self, batch, batch_idx: int) -> None:
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
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
