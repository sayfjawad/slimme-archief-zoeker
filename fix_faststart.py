#!/usr/bin/env python3
"""One-off + repeatable remux of debate mp4s so `moov` sits before `mdat`
("faststart"). ffmpeg's stream-copy download in dg_sync.py writes moov at
the end by default, which forces browsers to fetch the whole (often
1-2GB) file before playback/seeking can start. This walks a directory,
skips files that are already faststart, and remuxes (lossless, -c copy)
the rest in place.

Usage: python3 fix_faststart.py [directory ...]
Defaults to $SHARED_DIR/debatgemist if no directory is given.
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pipeline_config as pc


def is_faststart(path: Path) -> bool:
    """True if a top-level 'moov' atom appears before the first 'mdat'."""
    with open(path, "rb") as f:
        size = path.stat().st_size
        offset = 0
        while offset < size:
            f.seek(offset)
            header = f.read(8)
            if len(header) < 8:
                return False
            box_size = int.from_bytes(header[0:4], "big")
            box_type = header[4:8].decode("ascii", errors="replace")
            if box_type == "moov":
                return True
            if box_type == "mdat":
                return False
            if box_size == 1:
                box_size = int.from_bytes(f.read(8), "big")
            elif box_size == 0:
                return False
            offset += box_size
    return False


def remux(path: Path) -> bool:
    tmp = path.with_suffix(".faststart.tmp.mp4")
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(path),
           "-c", "copy", "-movflags", "+faststart", str(tmp)]
    rc = subprocess.run(cmd).returncode
    if rc != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(path)
    return True


def _process(args):
    i, total, path = args
    try:
        if is_faststart(path):
            return "skipped", path
        ok = remux(path)
        if ok:
            print(f"[{i}/{total}] fixed  {path.name}", flush=True)
            return "fixed", path
        print(f"[{i}/{total}] FAILED {path.name}", file=sys.stderr, flush=True)
        return "failed", path
    except Exception as e:
        print(f"[{i}/{total}] ERROR {path.name}: {e}", file=sys.stderr, flush=True)
        return "failed", path


def main():
    workers = 4
    args = sys.argv[1:]
    if args and args[0] == "--jobs":
        workers = int(args[1])
        args = args[2:]

    dirs = [Path(a) for a in args] or [pc.load_config()["_paths"]["debatgemist"]]
    files = sorted(f for d in dirs for f in Path(d).glob("*.mp4"))
    print(f"{len(files)} mp4 files to check ({workers} parallel)")

    counts = {"skipped": 0, "fixed": 0, "failed": 0}
    work = [(i, len(files), f) for i, f in enumerate(files, 1)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for status, _ in pool.map(_process, work):
            counts[status] += 1

    print(f"done: {counts['fixed']} fixed, {counts['skipped']} already ok, {counts['failed']} failed")


if __name__ == "__main__":
    main()
