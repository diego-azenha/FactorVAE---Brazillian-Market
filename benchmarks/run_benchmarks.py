"""
Generate prediction files for all benchmark models.

Run once immediately after build_features.py. Predictions are deterministic
(Momentum has no random init; Ridge has no random init), so they never need
to be regenerated unless the processed data or config splits change.

Usage:
    python benchmarks/run_benchmarks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Ensure the project root is on sys.path so `benchmarks.*` and `factorvae.*` are importable
# regardless of the working directory from which this script is invoked.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml


def main() -> None:
    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)

    out_dir = ROOT / "benchmarks" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import here so the module-level ROOT in each benchmark resolves correctly
    from benchmarks.momentum    import generate_predictions as momentum_predict
    from benchmarks.linear_model import train_and_predict   as linear_predict

    print("─" * 60)
    print("Running Momentum benchmark…")
    mom = momentum_predict(config)
    out = out_dir / "momentum_predictions.parquet"
    mom.to_parquet(out, index=False)
    print(f"  Saved {len(mom):,} rows → {out.relative_to(ROOT)}")

    print()
    print("Running Ridge Linear benchmark…")
    lin = linear_predict(config, alpha=1.0)
    out = out_dir / "linear_predictions.parquet"
    lin.to_parquet(out, index=False)
    print(f"  Saved {len(lin):,} rows → {out.relative_to(ROOT)}")
    print("─" * 60)
    print("Done. Re-run only if processed data or config splits change.")


if __name__ == "__main__":
    main()
