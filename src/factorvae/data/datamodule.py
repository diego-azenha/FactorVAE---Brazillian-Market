"""
FactorVAEDataModule — PyTorch Lightning DataModule.

Wraps three temporal splits (train / val / test) with leak-prevention assertions.
Each batch is a single cross-section (batch_size=1 in the DataLoader, squeezed in
the Lightning module).
"""

from __future__ import annotations

from pathlib import Path

import lightning as L
import pandas as pd
from torch.utils.data import DataLoader

from factorvae.data.dataset import RealDataset, SyntheticDataset


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
            use_macro = dc.get("use_macro", False)
            macro_normalizer = None
            if use_macro:
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

            T = dc["sequence_length"]
            ds_kwargs = {"use_macro": use_macro, "macro_normalizer": macro_normalizer}
            self._train = RealDataset(dc["processed_dir"], dc["train_start"], dc["train_end"], T, **ds_kwargs)
            self._val   = RealDataset(dc["processed_dir"], dc["val_start"],   dc["val_end"],   T, **ds_kwargs)
            self._test  = RealDataset(dc["processed_dir"], dc["test_start"],  dc["test_end"],  T, **ds_kwargs)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._train, batch_size=1, shuffle=True, num_workers=0)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._val, batch_size=1, shuffle=False, num_workers=0)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._test, batch_size=1, shuffle=False, num_workers=0)
