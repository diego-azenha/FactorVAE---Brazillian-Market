"""
FactorVAEDataModule — PyTorch Lightning DataModule.

Wraps three temporal splits (train / val / test) with leak-prevention assertions.
Each batch is a single cross-section (batch_size=1 in the DataLoader, squeezed in
the Lightning module).
"""

from __future__ import annotations

import lightning as L
from torch.utils.data import DataLoader

from factorvae.data.dataset import RealDataset, SyntheticDataset


class FactorVAEDataModule(L.LightningDataModule):
    def __init__(self, config: dict, use_synthetic: bool = False):
        super().__init__()
        self.config = config
        self.use_synthetic = use_synthetic
        self._train: SyntheticDataset | RealDataset | None = None
        self._val:   SyntheticDataset | RealDataset | None = None
        self._test:  SyntheticDataset | RealDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        dc = self.config["data"]

        if not self.use_synthetic:
            # Temporal leak guard
            assert dc["train_end"] < dc["val_start"], (
                f"Leak: train_end={dc['train_end']} >= val_start={dc['val_start']}"
            )
            assert dc["val_end"] < dc["test_start"], (
                f"Leak: val_end={dc['val_end']} >= test_start={dc['test_start']}"
            )

        if self.use_synthetic:
            C = self.config["model"]["num_features"]
            T = dc["sequence_length"]
            self._train = SyntheticDataset(num_dates=200, T=T, C=C, seed=42)
            self._val   = SyntheticDataset(num_dates=50,  T=T, C=C, seed=100)
            self._test  = SyntheticDataset(num_dates=50,  T=T, C=C, seed=200)
        else:
            processed_dir = dc["processed_dir"]
            T = dc["sequence_length"]
            self._train = RealDataset(processed_dir, dc["train_start"], dc["train_end"], T)
            self._val   = RealDataset(processed_dir, dc["val_start"],   dc["val_end"],   T)
            self._test  = RealDataset(processed_dir, dc["test_start"],  dc["test_end"],  T)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._train, batch_size=1, shuffle=True, num_workers=0)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._val, batch_size=1, shuffle=False, num_workers=0)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._test, batch_size=1, shuffle=False, num_workers=0)
