#!/usr/bin/env python3
"""Daily incremental sync for every tracked person (config/<slug>.json):
pulls new Kamerverslagen, Handelingen, YouTube uploads and Debat Direct video
since the last run, transcribes+diarizes any new YouTube audio, rebuilds the
index and restarts the app if anything changed, then mails a report.

Replaces the previous setup where only `@reboot resume.sh` or a manual run
ever re-triggered the sync scripts (last one before this: 2026-07-24) and
where the daily mail (progress_report.py --mail) was leftover ETA-on-the-
original-backlog math, sent for wilders only, never wired up for yesilgoz.

Video (dg_sync.py) runs single-process, no --shard: the 8-way distributed
run across gx10/PRD/edge-VPS was built to parallelize the one-time multi-
hundred-GB initial backlog; a daily delta of a handful of new debate days
finishes in minutes locally and doesn't need remote hosts involved.

Run daily via cron (flock-guarded so an unusually slow day can't overlap
with the next run): see crontab. Safe to also run manually; every step is
independently idempotent (same guarantee resume.sh relies on).
"""
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from pipeline_config import load_config

VENV_PY = "/data/git/scrib-r/venv/bin/python3"  # proven torch/torchcodec/pyannote combo
SHRINK_GUARD = 0.9  # refuse to go live if the new index has <90% of the old video count


def video_count(db_path: Path) -> int | None:
    if not db_path.exists():
        return None
    try:
        db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        n = db.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
        db.close()
        return n
    except sqlite3.Error:
        return None


def run(cmd, env, timeout):
    t0 = time.time()
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        out = (r.stdout + r.stderr).strip().splitlines()
        last = next((l for l in reversed(out) if l.strip()), "")
        return r.returncode == 0, last, time.time() - t0, out[-8:]
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout}s", time.time() - t0, []
    except Exception as e:
        return False, f"EXCEPTION: {e}", time.time() - t0, []


def count(glob_dir: Path, pattern: str) -> int:
    return sum(1 for _ in glob_dir.glob(pattern)) if glob_dir.exists() else 0


def sync_person(slug: str) -> tuple[str, bool, int]:
    """Returns (report text, any_failure, total_new_items)."""
    env = {**os.environ, "PERSON": slug}
    cfg = load_config(slug)
    p = cfg["_paths"]
    lines = [f"=== {slug} ==="]
    any_fail = False
    total_new = 0

    steps = [
        ("tk_sync", ["python3", "tk_sync.py"], 1800),
        ("tk_parse", ["python3", "tk_parse.py"], 900),
        ("ob_sync", ["python3", "ob_sync.py"], 1800),
        ("ob_parse", ["python3", "ob_parse.py"], 900),
    ]
    for name, cmd, timeout in steps:
        ok, last, dt, _ = run(cmd, env, timeout)
        lines.append(f"[{'OK' if ok else 'FAILED'}] {name} ({dt:.0f}s): {last}")
        if not ok:
            any_fail = True
        elif "done: 0 new" not in last and name in ("tk_parse", "ob_parse"):
            # tk_parse/ob_parse "done: N new, ..." -- N>0 means new content
            # relevant to this person specifically (already speaker-filtered)
            try:
                n = int(last.split("done: ")[1].split(" new")[0])
                total_new += n
            except (IndexError, ValueError):
                pass

    # YouTube: yt_sync.py has no summary line, so measure the opus-file delta
    before_opus = count(p["youtube"], "*.opus")
    ok, last, dt, tail = run(["python3", "yt_sync.py"], env, 3600)
    after_opus = count(p["youtube"], "*.opus")
    new_audio = after_opus - before_opus
    lines.append(f"[{'OK' if ok else 'FAILED'}] yt_sync ({dt:.0f}s): {new_audio} new audio file(s)")
    if not ok:
        any_fail = True
        lines.append(f"    {tail[-1] if tail else ''}")

    if new_audio > 0:
        diarize_env = {**env, "HF_HOME": "/data/huggingface"}
        ok, last, dt, tail = run(
            [VENV_PY, "transcribe_batch.py", "--diarize"], diarize_env, 7200
        )
        lines.append(f"[{'OK' if ok else 'FAILED'}] transcribe_batch --diarize ({dt:.0f}s): {last}")
        if not ok:
            any_fail = True
            lines.append(f"    {tail[-1] if tail else ''}")
        else:
            total_new += new_audio

    # Debate video: single-process (no --shard) -- see module docstring.
    # dg_sync.py's own internal ffmpeg backstop is 4h *per debate*
    # (PROC_TIMEOUT_S, for genuine multi-hour marathon debates like an APB)
    # -- this outer timeout must stay above that or we race-kill a legitimate
    # download (confirmed live 2026-07-29: killed dg_sync.py mid-download of
    # a 2022 APB session at the old 5400s/90min value; the orphaned ffmpeg
    # child survived the kill -- subprocess.run only signals its direct
    # child -- and had to be cleaned up by hand). 5h gives headroom above
    # one marathon debate plus the rest of that day's smaller dates.
    ok, last, dt, tail = run(["python3", "dg_sync.py", slug], env, 5 * 3600)
    lines.append(f"[{'OK' if ok else 'FAILED'}] dg_sync ({dt:.0f}s): {last}")
    if not ok:
        any_fail = True
        lines.append(f"    {tail[-1] if tail else ''}")
    else:
        try:
            n_new_video = int(last.split("done: ")[1].split(" downloaded")[0])
            total_new += n_new_video
        except (IndexError, ValueError):
            pass

    if total_new > 0:
        db_path = p["index"] / "index.sqlite"
        before_count = video_count(db_path)  # app.py loads this fully into memory
        # at startup and never re-reads it (see app.py:45-46) -- the running
        # service is unaffected by whatever build_index.py does to the file
        # on disk until we explicitly `systemctl restart` below. That restart
        # is the ONLY moment a bad rebuild could reach real traffic, so it's
        # the one place this script must refuse to proceed on its own.
        ok, last, dt, tail = run(["python3", "build_index.py"], env, 3600)
        lines.append(f"[{'OK' if ok else 'FAILED'}] build_index ({dt:.0f}s): {last}")
        if not ok:
            any_fail = True
            lines.append(f"    {tail[-1] if tail else ''}")
        else:
            after_count = video_count(db_path)
            regressed = (
                before_count is not None and before_count > 0
                and (after_count is None or after_count < before_count * SHRINK_GUARD)
            )
            if regressed:
                any_fail = True
                lines.append(
                    f"WAARSCHUWING: index kromp van {before_count} naar {after_count} "
                    f"video's (>{int((1 - SHRINK_GUARD) * 100)}% daling) -- app NIET herstart, "
                    f"blijft op de vorige (goede) index draaien. Vorige index staat veilig in "
                    f"{p['index']}/backups/<datum>/. Handmatig onderzoeken, dan "
                    f"rollback_index.py {slug} als de nieuwe build echt fout is."
                )
            else:
                # c4130 serves the index (CPU-only); ship the fresh index across
                # and bounce its system service there. The Z8 --user unit is gone
                # (moved to .moved-20260823) -- see ship_index.sh.
                r = subprocess.run(["./ship_index.sh", slug], capture_output=True, text=True)
                lines.append(f"[{'OK' if r.returncode == 0 else 'FAILED'}] ship+restart {slug}-search@c4130 "
                             f"({before_count} -> {after_count} video's)")
                if r.returncode != 0:
                    any_fail = True
                    lines.append(f"    {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ''}")
    else:
        lines.append("(no new content -> index not rebuilt)")

    return "\n".join(lines), any_fail, total_new


def main():
    now = datetime.now()
    config_dir = Path(__file__).parent / "config"
    slugs = sorted(p.stem for p in config_dir.glob("*.json"))

    # transcripts + index.sqlite/embeddings.npy only (NOT the video/audio raw
    # pools -- those are large and meant to accumulate, not be duplicated);
    # same call resume.sh makes first, kept idempotent (one backup/day).
    subprocess.run(["./backup_daily.sh"], cwd=Path(__file__).parent)

    sections = []
    any_fail = False
    totals = {}
    for slug in slugs:
        text, fail, n = sync_person(slug)
        sections.append(text)
        any_fail = any_fail or fail
        totals[slug] = n

    summary = ", ".join(f"{s} {n} nieuw" for s, n in totals.items())
    status = "FOUT" if any_fail else ("OK" if any(totals.values()) else "OK, niets nieuws")
    subject = f"[archief-sync] {status} {now:%Y-%m-%d}: {summary}"
    body = f"Dagelijkse sync {now:%F %H:%M}\n\n" + "\n\n".join(sections)

    print(body)
    if "--mail" in sys.argv:
        from notify import send_mail
        send_mail(subject, body)


if __name__ == "__main__":
    main()
