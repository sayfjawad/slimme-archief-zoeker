#!/usr/bin/env python3
"""Manual rollback: restore <data>/index/{index.sqlite,embeddings.npy} for
a person from one of build_index.py's own daily backups
(<data>/index/backups/<YYYY-MM-DD>/, 14-day retention) and restart their
search service.

Usage:
    python3 rollback_index.py <slug>              # restore the most recent backup
    python3 rollback_index.py <slug> 2026-07-24   # restore a specific date
    python3 rollback_index.py <slug> --list       # list available backups, do nothing

The CURRENT index.sqlite/embeddings.npy are saved first (to
backups/pre-rollback-<timestamp>/) so a rollback is itself never destructive
-- you can always undo an undo.
"""
import shutil
import sys
from datetime import datetime
from pathlib import Path

from pipeline_config import load_config


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    slug = sys.argv[1]
    cfg = load_config(slug)
    index_dir = cfg["_paths"]["index"]
    backups_dir = index_dir / "backups"

    available = sorted(
        (d.name for d in backups_dir.iterdir() if d.is_dir() and (d / "index.sqlite").exists())
    ) if backups_dir.exists() else []

    if "--list" in sys.argv or not available:
        print(f"available backups for {slug} in {backups_dir}:")
        for d in available:
            print(f"  {d}")
        if not available:
            print("  (none found)")
        return

    target = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else available[-1]
    if target not in available:
        print(f"no backup for {target}; available: {available}")
        sys.exit(1)

    src_dir = backups_dir / target
    current_db = index_dir / "index.sqlite"
    current_emb = index_dir / "embeddings.npy"

    # save whatever is live right now before overwriting it -- undo-an-undo safety
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    pre_dir = backups_dir / f"pre-rollback-{ts}"
    pre_dir.mkdir(parents=True, exist_ok=True)
    for f in (current_db, current_emb):
        if f.exists():
            shutil.copy2(f, pre_dir / f.name)
    print(f"current index saved to {pre_dir} before rollback")

    for name in ("index.sqlite", "embeddings.npy"):
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, index_dir / name)
            print(f"restored {name} from {target}")

    import subprocess
    r = subprocess.run(["./ship_index.sh", slug], capture_output=True, text=True)
    print(f"ship+restart {slug}-search@c4130: {'OK' if r.returncode == 0 else 'FAILED: ' + r.stderr}")


if __name__ == "__main__":
    main()
