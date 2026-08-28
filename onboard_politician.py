"""Onboard one Tweede Kamer politician into the combined search app -- TK
parliamentary text only (no Handelingen, no YouTube, no video download).

  python3 onboard_politician.py <slug>
      -- looks the row up in data/onboarding_queue.csv

  python3 onboard_politician.py <slug> --person "Mona Keijzer" \
      --voornaam Mona --achternaam Keijzer [--verslagnaam Keijzer]
      -- explicit (for politicians not in the queue yet, e.g. the first picks)

Idempotent. Runs under flock(/tmp/daily_sync.lock) so it can't race the 03:00
daily_sync cron or a second onboarding. Steps:
  1. pre-flight -- today's backup_daily.sh ran (else run it); /data/<SLUG>
     absent or index-less; slug not already a live config.
  2. write config/<slug>.json  (tk section only)
  3. PERSON=<slug> tk_parse.py <slug>    -> grows the SHARED transcript pool
  4. PERSON=<slug> build_index.py <slug> -> /data/<SLUG>/index/ only
  5. assert the fresh index has videos>0 and chunks>0

Does NOT restart politicus-search -- onboard_batch.sh does that once per batch.
Exit non-zero on any failure, leaving artifacts in place for inspection.
"""
import csv
import fcntl
import json
import os
import shutil
import subprocess
import sqlite3
import sys
import time
from datetime import date
from pathlib import Path

BASE = Path(__file__).parent
CONFIG_DIR = BASE / "config"
QUEUE = BASE / "data" / "onboarding_queue.csv"
LOCK = "/tmp/daily_sync.lock"
ODATA = "https://gegevensmagazijn.tweedekamer.nl/OData/v4/2.0"
PY = sys.executable


def die(msg: str, code: int = 1):
    print(f"ABORT: {msg}", file=sys.stderr)
    sys.exit(code)


def arg(name: str) -> str | None:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None


def queue_row(slug: str) -> dict | None:
    if not QUEUE.exists():
        return None
    with QUEUE.open() as fh:
        for row in csv.DictReader(fh):
            if row["slug"] == slug:
                return row
    return None


def run(cmd: list[str], env: dict) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, env=env, cwd=BASE)
    if r.returncode != 0:
        die(f"{cmd[1] if cmd[0] == PY else cmd[0]} exited {r.returncode}")


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not pos:
        die("usage: onboard_politician.py <slug> [--person .. --voornaam .. --achternaam ..]")
    slug = pos[0]

    row = queue_row(slug)
    voornaam = arg("--voornaam") or (row or {}).get("voornaam")
    achternaam = arg("--achternaam") or (row or {}).get("match_achternaam")
    verslagnaam = arg("--verslagnaam") or (row or {}).get("verslagnaam") or achternaam
    person = arg("--person") or (row or {}).get("person") or (
        f"{voornaam} {verslagnaam}".strip() if voornaam and verslagnaam else None)
    if not (voornaam and achternaam and person):
        die(f"{slug} not in {QUEUE.name} and --voornaam/--achternaam/--person not all given")

    cfg_path = CONFIG_DIR / f"{slug}.json"
    data_dir = Path(f"/data/{slug.upper().replace('-', '_')}")

    # ---- 1. pre-flight
    lock_fd = os.open(LOCK, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        die("another daily_sync / onboarding holds /tmp/daily_sync.lock -- retry later")

    idx = data_dir / "index" / "index.sqlite"
    emb = data_dir / "index" / "embeddings.npy"
    if cfg_path.exists() and emb.exists():
        die(f"{slug} already fully onboarded (config + index present)")
    if cfg_path.exists() or data_dir.exists():
        # a previous run got interrupted (config written, or a partial index
        # with no embeddings.npy) -- clear it and start clean
        print(f"clearing partial onboarding for {slug} (config={cfg_path.exists()}, "
              f"data_dir={data_dir.exists()})", flush=True)
        cfg_path.unlink(missing_ok=True)
        if data_dir.exists():
            shutil.rmtree(data_dir)

    backup_marker = Path(f"/data/backups/wilders-search/{date.today().isoformat()}")
    if not backup_marker.exists():
        print("today's backup_daily.sh has not run yet -- running it now", flush=True)
        run(["./backup_daily.sh"], {**os.environ})
        if not backup_marker.exists():
            die("backup_daily.sh did not produce today's dated backup")
    print(f"pre-flight OK -- backup present ({backup_marker.name})", flush=True)

    # ---- 2. config
    cfg = {
        "person": person,
        "slug": slug,
        "data_dir": str(data_dir),
        "tk": {
            "odata_base": ODATA,
            "match": {"achternaam": achternaam, "voornaam": voornaam.split()[0]},
            "since": "2013-01-01",
        },
    }
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {cfg_path}:\n{cfg_path.read_text()}", flush=True)

    # GPU 0 on c4130 is the free one (the other three run gpt-oss-120b); the
    # nightly daily_sync build_index uses it too. Embedding 300k+ chunks on
    # CPU is ~70 min; on the V100 it is a few minutes.
    env = {**os.environ, "PERSON": slug, "HF_HOME": "/data/huggingface", "CUDA_VISIBLE_DEVICES": "0"}
    t0 = time.time()

    # ---- 3. grow the shared transcript pool with this person's debates
    run([PY, "tk_sync.py", slug], env)      # cheap: shared cursor, usually a no-op
    run([PY, "tk_parse.py", slug], env)

    # ---- 4. build only this person's index
    run([PY, "build_index.py", slug], env)

    # ---- 5. verify
    con = sqlite3.connect(idx)
    try:
        n_vid = con.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
        n_chunk = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        n_person = con.execute(
            "SELECT COUNT(*) FROM chunks WHERE lower(speaker) LIKE ?", (f"%{achternaam.lower()}%",)
        ).fetchone()[0]
    finally:
        con.close()
    if n_vid == 0 or n_chunk == 0:
        die(f"index built empty ({n_vid} videos, {n_chunk} chunks) -- config left for inspection")
    if n_person == 0:
        die(f"index has {n_vid} videos but 0 chunks spoken by '{achternaam}' -- check the match name")

    print(f"\nOK  {slug}: {n_vid} debates, {n_chunk} chunks ({n_person} by {achternaam}) "
          f"in {(time.time()-t0)/60:.1f} min. Restart politicus-search to serve it.")


if __name__ == "__main__":
    main()
