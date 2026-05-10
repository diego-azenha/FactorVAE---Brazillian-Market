"""
Run management and versioning system.

Manages timestamped output directories for experiments, ensuring reproducibility
and preventing results from overwriting previous runs.

Usage:
    manager = RunManager(root_dir)
    manager.save_comparison_table(table_df)  # Saves to run dir with timestamp
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


class RunManager:
    """
    Manage versioned output directories for experiment runs.

    Each run gets a unique timestamped directory with metadata.

    Attributes:
        run_dir : Path to this run's output directory
        run_id  : Unique run identifier (timestamp + short UUID)
        created_at : ISO timestamp of run creation
    """

    def __init__(
        self,
        root_dir: Path | str,
        run_name: str = "run",
        auto_create: bool = True,
    ):
        """
        Initialize RunManager.

        Args:
            root_dir     : Root directory for all runs (e.g., results/)
            run_name     : Prefix for run directory (e.g., "exp1", "backtest")
            auto_create  : Create run_dir if it doesn't exist
        """
        self.root_dir = Path(root_dir)
        self.run_name = run_name

        # Generate unique run ID: timestamp + short UUID
        now = datetime.now()
        self.created_at = now.isoformat()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        self.run_id = f"{timestamp}_{short_uuid}"

        # Create run directory
        self.run_dir = self.root_dir / self.run_id
        if auto_create:
            self.run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.figures_dir = self.run_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)

        self.predictions_dir = self.run_dir / "predictions"
        self.predictions_dir.mkdir(exist_ok=True)

        # Save run metadata
        self._save_metadata()

    def _save_metadata(self) -> None:
        """Save run metadata to JSON."""
        metadata = {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "run_name": self.run_name,
        }
        meta_path = self.run_dir / "run_info.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def save_figure(self, fig_path: Path, name: str | None = None) -> Path:
        """
        Copy figure to run directory.

        Args:
            fig_path : Path to figure file
            name     : Target filename (defaults to fig_path.name)

        Returns:
            Path to saved figure in run directory
        """
        src = Path(fig_path)
        if not src.exists():
            raise FileNotFoundError(f"Figure not found: {src}")

        dst_name = name or src.name
        dst = self.figures_dir / dst_name
        dst.write_bytes(src.read_bytes())
        return dst

    def save_comparison_table(self, df: pd.DataFrame, name: str = "comparison_table.csv") -> Path:
        """
        Save comparison table to run directory.

        Args:
            df   : DataFrame to save
            name : Filename (default: comparison_table.csv)

        Returns:
            Path to saved CSV
        """
        out_path = self.run_dir / name
        df.to_csv(out_path, index=False)
        return out_path

    def save_predictions(self, df: pd.DataFrame, name: str, model_name: str | None = None) -> Path:
        """
        Save model predictions to run directory.

        Args:
            df         : DataFrame with predictions
            name       : Filename (e.g., gru_predictions.parquet)
            model_name : Optional model name (metadata only)

        Returns:
            Path to saved parquet
        """
        out_path = self.predictions_dir / name
        df.to_parquet(out_path, index=False)
        return out_path

    def summary(self) -> dict[str, Any]:
        """Return summary of this run."""
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "run_dir": str(self.run_dir),
            "figures_dir": str(self.figures_dir),
            "predictions_dir": str(self.predictions_dir),
        }

    def __repr__(self) -> str:
        return f"RunManager(run_id='{self.run_id}', root='{self.root_dir}')"


def get_latest_run(root_dir: Path | str) -> RunManager | None:
    """
    Load the most recent run (by directory name).

    Args:
        root_dir : Root directory containing run subdirectories

    Returns:
        RunManager for latest run, or None if no runs exist
    """
    root = Path(root_dir)
    if not root.exists():
        return None

    # List all subdirectories, find most recent by name (timestamp-based)
    run_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not run_dirs:
        return None

    latest = run_dirs[-1]
    manager = RunManager(root, auto_create=False)
    manager.run_id = latest.name
    manager.run_dir = latest
    manager.figures_dir = latest / "figures"
    manager.predictions_dir = latest / "predictions"
    return manager
