#!/usr/bin/env python3
"""Autonomous onboarding orchestrator.

Runs non-stop on hp-z8 as a systemd service (politici-onboard.service,
Restart=always + linger -> survives reboot). Works down
data/onboarding_queue.csv, onboarding every politician into the combined
politicus.zoek-r.nl app, distributing the build across hp-z8 and c4130.

All durable state is one JSON file on the NFS share so a reboot resumes
exactly where it left off and any machine can read progress:
    /data/SHARED/onboarding_state.json

Per politician:
  1. onboard_politician.py <slug>  runs on the assigned worker -- writes
     config/<slug>.json, tk_parse --all (no-op once the pool is complete),
     build_index (100% cache hits after the pre-warm -> no GPU), asserts the
     index is non-empty.
  2. if the worker was hp-z8: rsync /data/<SLUG>/ + the config file to c4130
     (c4130 is where politicus-search reads them).
  3. mark done in the state file.
Every RESTART_EVERY completions: ssh c4130 `systemctl restart politicus-search`,
verify each new slug serves person_chunks>0, then commit_transcripts.sh.

Prereqs it will NOT do (must be true before it starts serving useful work):
  - the shared TK pool is populated (tk_parse.py --all)
  - the embedding cache is warm + present at EMB_CACHE_LOCAL on BOTH machines
It checks both and refuses to start builds until they hold, logging what's
missing, so a human can fix it.

Monitoring:  ./onboard_status.sh    (or: jq . /data/SHARED/onboarding_state.json)
"""
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent
SHARED = Path(os.environ.get("SHARED_DIR", "/data/SHARED"))
STATE_FILE = SHARED / "onboarding_state.json"
QUEUE_CSV = REPO / "data" / "onboarding_queue.csv"
LOG_DIR = REPO / "data" / "onboard_logs"
EMB_CACHE_CANON = SHARED / "emb_cache"
EMB_CACHE_LOCAL = Path(os.environ.get("EMB_CACHE_LOCAL", "/data/emb_cache"))

C4130 = os.environ.get("C4130_HOST", "sayf@100.64.0.13")
C4130_REPO = "/data/politicus-search"
C4130_PY = "/data/abo-ali-search/.venv/bin/python3"
Z8_PY = os.environ.get("Z8_PY", "/data/git/gemeente-search/.venv/bin/python3")

RESTART_EVERY = int(os.environ.get("RESTART_EVERY", "8"))
POLL_SECONDS = 15
STALE_BUILD_MIN = 45          # a "building" entry older than this is presumed dead
BUILD_TIMEOUT_MIN = 25        # kill a tracked build that runs longer than this
MAX_ATTEMPTS = 3

WORKERS = ["z8", "c4130"]     # one concurrent build each


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"{now()}  {msg}", flush=True)


def sh(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# --------------------------------------------------------------- state I/O
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except ValueError:
            log("state file corrupt -- starting fresh")
    return {"started": now(), "politicians": {}, "completed_since_restart": 0,
            "restarts": 0, "commits": 0}


def save_state(st: dict) -> None:
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(st, indent=1))
    os.replace(tmp, STATE_FILE)


def queue_rows() -> list[dict]:
    with QUEUE_CSV.open() as fh:
        return list(csv.DictReader(fh))


def _onboarded_on_c4130() -> set[str]:
    """slugs that already have config/<slug>.json AND /data/<SLUG>/index/embeddings.npy."""
    r = sh(["ssh", C4130,
            f"for f in {C4130_REPO}/config/*.json; do s=$(basename $f .json); "
            f"d=/data/$(echo $s | tr 'a-z-' 'A-Z_')/index/embeddings.npy; "
            f"[ -f \"$d\" ] && echo $s; done"])
    return set(r.stdout.split())


def reconcile(st: dict, done_on_disk: set[str]) -> list[str]:
    """Add any new queue rows as pending; return the ordered slug list."""
    order = []
    for row in queue_rows():
        slug = row["slug"]
        order.append(slug)
        p = st["politicians"].setdefault(slug, {"status": "pending", "attempts": 0})
        p["rank"] = int(row["rank"])
        p["person"] = f"{row['voornaam']} {row['verslagnaam']}".strip()
        # already live on c4130 (hand-onboarded, or a previous orchestrator run)
        if slug in done_on_disk and p["status"] not in ("done", "building"):
            log(f"{slug}: already onboarded on c4130 -> done")
            p.update(status="done", finished=p.get("finished") or now(), served=True)
        # resurrect a build that died (reboot / crash) mid-flight -- only if
        # it's genuinely old AND has no finished index (a fresh build in pass 1
        # legitimately has no embeddings.npy yet), and isn't a live subprocess
        if p["status"] == "building" and slug not in _running:
            started = p.get("started", "")
            age_min = (time.time() - _epoch(started)) / 60 if started else 999
            if age_min > STALE_BUILD_MIN and not _remote_index_exists(slug, p.get("worker")):
                log(f"{slug}: stale 'building' ({age_min:.0f} min, no index) -> back to pending")
                p["status"] = "pending"
    return order


def _epoch(iso: str) -> float:
    try:
        return datetime.fromisoformat(iso).timestamp()
    except ValueError:
        return 0.0


# --------------------------------------------------------------- prereqs
_prereq_ok_until = 0.0


def prereqs_ok() -> bool:
    global _prereq_ok_until
    if time.time() < _prereq_ok_until:
        return True
    ok = True
    if not (EMB_CACHE_LOCAL / "main.keys.npy").exists():
        log(f"MISSING: {EMB_CACHE_LOCAL}/main.keys.npy on this host -- run the "
            f"pre-warm + compact + rsync canon->local first")
        ok = False
    r = sh(["ssh", C4130, f"test -f {EMB_CACHE_LOCAL}/main.keys.npy && echo ok"])
    if "ok" not in r.stdout:
        log(f"MISSING: {EMB_CACHE_LOCAL}/main.keys.npy on c4130")
        ok = False
    pool = len(list((SHARED / "transcripts").glob("*.metadata.json")))
    if pool < 2000:
        log(f"shared pool looks unpopulated ({pool} transcripts) -- run tk_parse.py --all")
        ok = False
    # today's dated backup on the serving host (run it if missing -- this is
    # the safeguard onboard_politician skips via ONBOARD_SKIP_BACKUP)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    r = sh(["ssh", C4130, f"test -d /data/backups/wilders-search/{today} && echo ok"])
    if "ok" not in r.stdout:
        log("running backup_daily.sh on c4130 (no dated backup for today yet)")
        sh(["ssh", C4130, f"cd {C4130_REPO} && ./backup_daily.sh"], timeout=1800)
        r = sh(["ssh", C4130, f"test -d /data/backups/wilders-search/{today} && echo ok"])
        if "ok" not in r.stdout:
            log("backup_daily.sh did not produce today's backup -- refusing to build")
            ok = False
    if ok:
        _prereq_ok_until = time.time() + 1200  # re-verify at most every 20 min
    return ok


# --------------------------------------------------------------- builds
_running: dict[str, dict] = {}   # slug -> {worker, proc, log, started}


def _remote_index_exists(slug: str, worker: str | None) -> bool:
    idx = f"/data/{slug.upper().replace('-', '_')}/index/embeddings.npy"
    if worker == "c4130":
        return "ok" in sh(["ssh", C4130, f"test -f {idx} && echo ok"]).stdout
    return Path(idx).exists()


def start_build(slug: str, worker: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logf = LOG_DIR / f"{slug}.log"
    fh = open(logf, "ab")
    fh.write(f"\n===== {now()}  onboarding {slug} on {worker} =====\n".encode())
    fh.flush()
    gpu = "1" if worker == "z8" else "0"   # z8 GPU0 is often busy; c4130 GPU0 is the free one
    env = (f"EMB_CACHE_DIR={EMB_CACHE_LOCAL} HF_HOME=/data/huggingface "
           f"ONBOARD_SKIP_BACKUP=1 CUDA_VISIBLE_DEVICES={gpu}")
    if worker == "c4130":
        # foreground (no setsid): if the orchestrator dies the ssh dies and the
        # remote onboard_politician dies with it -- reconcile() then resets the
        # slug and it retries clean (onboard_politician clears its own partial).
        cmd = ["ssh", C4130, f"cd {C4130_REPO} && {env} {C4130_PY} onboard_politician.py {slug}"]
    else:
        cmd = ["bash", "-c", f"cd {REPO} && {env} {Z8_PY} onboard_politician.py {slug}"]
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
    _running[slug] = {"worker": worker, "proc": proc, "fh": fh, "log": str(logf),
                      "started": time.time()}
    log(f"{slug}: build started on {worker} (pid {proc.pid})")


def finish_build(slug: str, st: dict) -> None:
    r = _running.pop(slug)
    r["fh"].close()
    rc = r["proc"].returncode
    tail = _tail(r["log"], 12)
    p = st["politicians"][slug]
    ok_line = any(l.startswith(f"OK  {slug}:") for l in tail)
    if rc == 0 and ok_line and _remote_index_exists(slug, r["worker"]):
        if r["worker"] == "z8":
            _sync_to_c4130(slug)
        stats = _index_stats(slug)
        p.update(status="done", finished=now(),
                 dur_min=round((time.time() - r["started"]) / 60, 1), **stats)
        st["completed_since_restart"] += 1
        log(f"{slug}: DONE in {p['dur_min']} min ({stats.get('videos','?')} debates, "
            f"{stats.get('person_chunks','?')} by them)")
    else:
        p["attempts"] = p.get("attempts", 0) + 1
        p.update(status="failed" if p["attempts"] >= MAX_ATTEMPTS else "pending",
                 error=f"rc={rc}: {tail[-1] if tail else ''}"[:300], last_try=now())
        log(f"{slug}: build FAILED (rc={rc}, attempt {p['attempts']}) -- {p['status']}")
    save_state(st)


def _sync_to_c4130(slug: str) -> None:
    d = f"/data/{slug.upper().replace('-', '_')}"
    sh(["rsync", "-a", "--delete", f"{d}/", f"{C4130}:{d}/"], timeout=1200)
    sh(["scp", str(REPO / "config" / f"{slug}.json"),
        f"{C4130}:{C4130_REPO}/config/{slug}.json"])
    log(f"{slug}: synced {d} + config -> c4130")


def _index_stats(slug: str) -> dict:
    r = sh(["ssh", C4130, f"curl -s 'http://localhost:8905/api/stats?person={slug}'"])
    try:
        d = json.loads(r.stdout)
        return {k: d[k] for k in ("videos", "chunks", "person_chunks") if k in d}
    except ValueError:
        return {}


def _tail(path: str, n: int) -> list[str]:
    try:
        return Path(path).read_text(errors="replace").splitlines()[-n:]
    except OSError:
        return []


# --------------------------------------------------------------- serve
def restart_and_commit(st: dict, new_done: list[str]) -> None:
    log(f"restarting politicus-search for {len(new_done)} new: {', '.join(new_done)}")
    sh(["ssh", C4130, f"cd {C4130_REPO} && git pull -q; sudo -n systemctl restart politicus-search"],
       timeout=120)
    for _ in range(40):
        if "200" in sh(["ssh", C4130,
                        "curl -s -o /dev/null -w '%{http_code}' localhost:8905/api/persons"]).stdout:
            break
        time.sleep(3)
    for slug in new_done:
        s = _index_stats(slug)
        pc = s.get("person_chunks", 0)
        st["politicians"][slug]["served"] = True
        st["politicians"][slug]["served_person_chunks"] = pc
        log(f"  {slug}: person_chunks={pc}  {'OK' if pc else 'WARN 0!'}")
    sh(["ssh", C4130, f"cd {C4130_REPO} && ./commit_transcripts.sh"], timeout=1800)
    st["completed_since_restart"] = 0
    st["restarts"] += 1
    st["commits"] += 1
    st["last_restart"] = now()
    save_state(st)


# --------------------------------------------------------------- main loop
def main() -> None:
    log(f"orchestrator up. repo={REPO} state={STATE_FILE}")
    st = load_state()
    # nothing is tracked in _running yet, so any "building" entry is from a
    # previous process -- let reconcile() sort it (done-on-disk -> done, else
    # -> pending for a clean retry)
    for p in st["politicians"].values():
        if p.get("status") == "building":
            p["status"] = "pending"
    warned_prereq = False
    last_fetch = 0.0
    while True:
        if time.time() - last_fetch > 300:   # pick up code pushes every ~5 min
            try:
                sh(["git", "-C", str(REPO), "fetch", "-q", "origin"], timeout=120)
                sh(["git", "-C", str(REPO), "merge", "-q", "--ff-only", "origin/master"], timeout=60)
            except subprocess.SubprocessError:
                pass
            last_fetch = time.time()

        order = reconcile(st, _onboarded_on_c4130())
        save_state(st)

        pending = [s for s in order if st["politicians"][s]["status"] == "pending"
                   and s not in _running]
        done_now = [s for s in order if st["politicians"][s]["status"] == "done"]
        active_workers = {r["worker"] for r in _running.values()}

        if pending and not prereqs_ok():
            if not warned_prereq:
                log("prereqs not met -- idling until they are (see messages above)")
                warned_prereq = True
            time.sleep(POLL_SECONDS * 4)
            continue
        warned_prereq = False

        # launch on any idle worker
        for w in WORKERS:
            if w not in active_workers and pending:
                slug = pending.pop(0)
                p = st["politicians"][slug]
                p.update(status="building", worker=w, started=now())
                save_state(st)
                start_build(slug, w)

        # reap finished; kill a build that has run far past any sane duration
        # (a hung ssh, a wedged NFS read) so its worker slot frees up
        for slug in list(_running):
            r = _running[slug]
            if r["proc"].poll() is not None:
                finish_build(slug, st)
            elif time.time() - r["started"] > BUILD_TIMEOUT_MIN * 60:
                log(f"{slug}: build exceeded {BUILD_TIMEOUT_MIN} min -- killing")
                r["proc"].kill()
                if r["worker"] == "c4130":
                    sh(["ssh", C4130, f"pkill -f 'onboard_politician.py {slug}'"])
                finish_build(slug, st)

        # politicians built but not yet pushed live -- a politicus-search
        # restart (~30s) doesn't disturb in-flight builds, so don't wait for a
        # quiet moment (which may never come with 2 workers always busy)
        fresh = [s for s in done_now if not st["politicians"][s].get("served")]
        if fresh and (st["completed_since_restart"] >= RESTART_EVERY or not pending):
            restart_and_commit(st, fresh)

        # status heartbeat
        counts = {}
        for p in st["politicians"].values():
            counts[p["status"]] = counts.get(p["status"], 0) + 1
        if not _running and not pending:
            log(f"idle. {counts}")
            time.sleep(POLL_SECONDS * 8)   # queue drained: check back less often
        else:
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
