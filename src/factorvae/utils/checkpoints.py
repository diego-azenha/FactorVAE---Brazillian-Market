"""
Checkpoint resolution utility.
"""

from __future__ import annotations

from pathlib import Path


def resolve_checkpoint(checkpoint_arg: str, dirpath: "Path | str | None" = None) -> str:
    """
    Resolve a checkpoint path.

    If *checkpoint_arg* points to an existing file, return it unchanged.
    Otherwise, find the most recently modified .ckpt file in *dirpath*
    (defaults to the same directory as *checkpoint_arg*).

    Raises FileNotFoundError if no .ckpt file is found.
    """
    p = Path(checkpoint_arg)
    if p.exists():
        return str(p)

    search_dir = Path(dirpath) if dirpath is not None else p.parent
    candidates = sorted(search_dir.glob("*.ckpt"), key=lambda f: f.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No .ckpt file found in {search_dir}. "
            "Run scripts/train.py first."
        )
    latest = candidates[-1]
    print(f"[checkpoint] '{p.name}' not found — using latest: {latest.name}")
    return str(latest)
