"""
Generate prediction files for all benchmark models.

Benchmarks (in narrative order):
  GRU  — temporal model, no factor structure (same features as FactorVAE)
  IPCA — latent-factor model with linear conditional loadings (fundamentals)
  CA   — latent-factor model with non-linear conditional loadings (fundamentals)

Usage:
    python benchmarks/run_benchmarks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml


def main() -> None:
    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)

    out_dir = ROOT / "benchmarks" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    from benchmarks.gru  import train_and_predict  as gru_predict
    from benchmarks.ipca import train_and_predict  as ipca_predict
    from benchmarks.ca   import train_and_predict  as ca_predict

    print("-" * 60)
    print("Running GRU benchmark…")
    gru_df = gru_predict(config)
    out = out_dir / "gru_predictions.parquet"
    gru_df.to_parquet(out, index=False)
    print(f"  Saved {len(gru_df):,} rows → {out.relative_to(ROOT)}")

    print()
    print("Running IPCA benchmark…")
    ipca_df = ipca_predict(config, K=3)
    out = out_dir / "ipca_predictions.parquet"
    ipca_df.to_parquet(out, index=False)
    print(f"  Saved {len(ipca_df):,} rows → {out.relative_to(ROOT)}")

    print()
    print("Running CA benchmark…")
    ca_df = ca_predict(config, K=3)
    out = out_dir / "ca_predictions.parquet"
    ca_df.to_parquet(out, index=False)
    print(f"  Saved {len(ca_df):,} rows → {out.relative_to(ROOT)}")

    print("-" * 60)
    print("Done. Re-run only if processed data or config splits change.")


if __name__ == "__main__":
    main()
