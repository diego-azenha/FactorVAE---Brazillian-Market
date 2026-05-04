"""
Day 0.5 master runner — executes all three pre-flight diagnostics and writes
a consolidated report to results/diagnostics/day05_report.txt.

Day 0.5 checklist (no model training required):
  [1] Cross-fit posterior IC  — falsifies the leakage hypothesis
  [2] GBM baseline            — establishes the feature-only upper-bound IC
  [3] KL scale fix            — already applied in code (sum → mean over K)
                                Verified here via test suite.

Usage
-----
    python scripts/run_day05_diagnostics.py
    python scripts/run_day05_diagnostics.py --checkpoint results/checkpoints/best.ckpt
    python scripts/run_day05_diagnostics.py --skip-gbm    # faster, skip GBM training
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[1]
DIAG_DIR = ROOT / "results" / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)


def _run(script: str, extra_args: list[str] = []) -> tuple[str, int]:
    """Run a scripts/ script and return (stdout+stderr, returncode)."""
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


def _run_tests() -> tuple[str, int]:
    """Run only the distributions and losses unit tests to verify KL fix."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_distributions.py",
        "tests/test_losses.py",
        "-v", "--tb=short",
    ]
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


def _decision_summary(crossfit_out: str, gbm_out: str) -> str:
    """Extract key numbers and print an actionable summary."""
    lines = []
    lines.append("=" * 65)
    lines.append("DAY 0.5 DECISION SUMMARY")
    lines.append("=" * 65)

    # Parse cross-fit IC from output
    cf_post = cf_prior = None
    for line in crossfit_out.splitlines():
        if "Cross-fit posterior IC" in line and "Mean" not in line:
            try:
                cf_post = float(line.split()[-4])   # Mean column
            except (ValueError, IndexError):
                pass
        if "Cross-fit prior IC" in line and "Mean" not in line:
            try:
                cf_prior = float(line.split()[-4])
            except (ValueError, IndexError):
                pass

    # Forward the DECISION block verbatim
    in_decision = False
    for line in crossfit_out.splitlines():
        if line.strip().startswith("DECISION:"):
            in_decision = True
        if in_decision:
            lines.append(line)
        if in_decision and line.strip() == "":
            break

    lines.append("")
    lines.append("From GBM baseline:")
    in_interp = False
    for line in gbm_out.splitlines():
        if line.strip().startswith("INTERPRETATION:"):
            in_interp = True
        if in_interp:
            lines.append(line)
        if in_interp and line.strip() == "":
            break

    lines.append("")
    lines.append("KL scale fix: applied in src/factorvae/models/distributions.py")
    lines.append("  kl_gaussian_diagonal now returns .mean() over K factors (was .sum())")
    lines.append("  Re-train from scratch before any new experiment.")

    lines.append("")
    lines.append("NEXT STEPS:")
    lines.append("  If cross-fit posterior IC survived (> 0.10, retention > 50%):")
    lines.append("    Mon-Tue  Option 7: supervised regression predictor → frozen mu_post")
    lines.append("    Wed      Option 5: architecture variant in parallel")
    lines.append("    Thu-Fri  Combine winner with Option 1")
    lines.append("    Week 2   Macro features only if Week 1 closes < 50 % of gap")
    lines.append("    Decision gate Friday Week 1: if no model reaches IC > 0.08, write paper.")
    lines.append("")
    lines.append("  If cross-fit posterior IC degraded (≤ 0.10 or retention ≤ 50%):")
    lines.append("    Ship GRU for production (Sharpe +0.45, IR neutral).")
    lines.append("    Write honest paper: FactorVAE teacher is a retrospective reconstructor.")
    lines.append("=" * 65)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "results" / "checkpoints" / "best.ckpt"),
    )
    parser.add_argument(
        "--skip-gbm",
        action="store_true",
        help="Skip GBM training (faster; use when checkpoint is not yet trained).",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["val", "test"],
        help="Dataset split for posterior IC cross-fitting (default: test).",
    )
    args = parser.parse_args()

    ckpt  = args.checkpoint
    split = args.split

    report_lines: list[str] = []
    report_lines.append("=" * 65)
    report_lines.append("DAY 0.5 PRE-FLIGHT DIAGNOSTIC REPORT")
    report_lines.append(f"Run at     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Checkpoint : {Path(ckpt).name}")
    report_lines.append(f"Split      : {split.upper()}")
    report_lines.append("=" * 65)
    report_lines.append(textwrap.dedent("""
    Purpose: Verify that the posterior IC target is a genuine predictive signal
    (not a reconstruction artifact) before committing to the 2-week plan.

    Three items:
      [1] Cross-fit posterior IC  — leakage falsification
      [2] GBM baseline IC         — feature-only upper bound
      [3] KL scale fix            — sum→mean, verified via unit tests
    """))

    # ── [1] Cross-fit posterior IC ────────────────────────────────────────
    print("Running [1/3] Cross-fit posterior IC ...", flush=True)
    crossfit_out, crossfit_rc = _run(
        "crossfit_posterior_ic.py",
        ["--checkpoint", ckpt, "--split", split],
    )
    report_lines.append(_section("1. CROSS-FIT POSTERIOR IC (leakage test)", crossfit_out, crossfit_rc))

    # ── [2] GBM baseline ─────────────────────────────────────────────────
    if args.skip_gbm:
        print("Skipping [2/3] GBM baseline (--skip-gbm).", flush=True)
        gbm_out = "(skipped via --skip-gbm)"
        gbm_rc  = 0
    else:
        print("Running [2/3] GBM baseline (training on full train panel) ...", flush=True)
        gbm_out, gbm_rc = _run(
            "gbm_baseline.py",
            ["--checkpoint", ckpt],
        )
    report_lines.append(_section("2. GBM FEATURE BASELINE IC", gbm_out, gbm_rc))

    # ── [3] KL scale fix — verify via unit tests ─────────────────────────
    print("Running [3/3] Unit tests for KL scale fix ...", flush=True)
    test_out, test_rc = _run_tests()
    report_lines.append(_section("3. KL SCALE FIX VERIFICATION (unit tests)", test_out, test_rc))

    # ── Decision summary ──────────────────────────────────────────────────
    summary = _decision_summary(crossfit_out, gbm_out)
    report_lines.append("\n" + summary)

    # ── Write report ──────────────────────────────────────────────────────
    report_path = DIAG_DIR / "day05_report.txt"
    full_report = "\n".join(report_lines)
    report_path.write_text(full_report, encoding="utf-8")

    print("\n" + "=" * 65)
    print("ALL DAY 0.5 DIAGNOSTICS COMPLETE")
    print(f"Report saved to: results/diagnostics/day05_report.txt")
    print("=" * 65)
    print()

    for line in full_report.splitlines():
        try:
            print(line.encode("cp1252").decode("cp1252"))
        except (UnicodeEncodeError, UnicodeDecodeError):
            print(line.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
