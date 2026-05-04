"""
Day 0 master runner — executes all three diagnostics and writes a
single consolidated report to results/diagnostics/day0_report.txt.

Usage:
    python scripts/run_day0_diagnostics.py
    python scripts/run_day0_diagnostics.py --checkpoint results/checkpoints/best-v1.ckpt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAG_DIR = ROOT / "results" / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)


def _run(script: str, extra_args: list[str] = []) -> tuple[str, int]:
    """Run a script and return (stdout+stderr, returncode)."""
    cmd = [sys.executable, str(ROOT / "scripts" / script)] + extra_args
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    output = result.stdout
    if result.stderr.strip():
        output += "\n[STDERR]\n" + result.stderr
    return output, result.returncode


def _section(title: str, output: str, rc: int) -> str:
    status = "OK" if rc == 0 else f"FAILED (rc={rc})"
    bar = "-" * 65
    return f"\n{bar}\n{title}  [{status}]\n{bar}\n{output.strip()}\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "results" / "checkpoints" / "best.ckpt"),
        help="Checkpoint to use for posterior IC and metrics diagnostics.",
    )
    parser.add_argument(
        "--predictions",
        default=str(ROOT / "results" / "predictions" / "predictions.parquet"),
    )
    args = parser.parse_args()

    ckpt  = args.checkpoint
    preds = args.predictions

    report_lines: list[str] = []
    report_lines.append("=" * 65)
    report_lines.append("DAY 0 DIAGNOSTIC REPORT")
    report_lines.append(f"Run at  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Checkpoint : {Path(ckpt).name}")
    report_lines.append(f"Predictions: {Path(preds).name}")
    report_lines.append("=" * 65)

    # ── 1. Universe / survivorship ────────────────────────────────────────
    print("Running [1/3] Universe survivorship audit ...", flush=True)
    out, rc = _run("diagnose_universe.py")
    report_lines.append(_section("1. UNIVERSE SURVIVORSHIP & LOOK-AHEAD", out, rc))

    # ── 2. Posterior IC leakage — test split ─────────────────────────────
    print("Running [2/3] Posterior IC leakage check (test) ...", flush=True)
    out, rc = _run("diagnose_posterior_ic.py", ["--checkpoint", ckpt, "--split", "test"])
    report_lines.append(_section("2. POSTERIOR IC LEAKAGE CHECK (TEST)", out, rc))

    # ── 2b. Posterior IC leakage — val split ─────────────────────────────
    print("Running [2b/3] Posterior IC leakage check (val) ...", flush=True)
    out, rc = _run("diagnose_posterior_ic.py", ["--checkpoint", ckpt, "--split", "val"])
    report_lines.append(_section("2b. POSTERIOR IC LEAKAGE CHECK (VAL)", out, rc))

    # ── 3. Metric arithmetic ──────────────────────────────────────────────
    print("Running [3/3] Metric arithmetic breakdown ...", flush=True)
    out, rc = _run("diagnose_metrics.py", ["--predictions", preds])
    report_lines.append(_section("3. METRIC ARITHMETIC BREAKDOWN", out, rc))

    # ── Write report ──────────────────────────────────────────────────────
    report_path = DIAG_DIR / "day0_report.txt"
    full_report = "\n".join(report_lines)
    report_path.write_text(full_report, encoding="utf-8")

    print("\n" + "=" * 65)
    print("ALL DIAGNOSTICS COMPLETE")
    print(f"Report saved to: results/diagnostics/day0_report.txt")
    print("=" * 65)
    print()
    # Print a concise summary to stdout (ASCII-safe)
    print("\nSUMMARY OF RESULTS:")
    for line in full_report.splitlines():
        try:
            line.encode('cp1252')
            print(line)
        except (UnicodeEncodeError, AttributeError):
            print(line.encode('ascii', errors='replace').decode('ascii'))


if __name__ == "__main__":
    main()
