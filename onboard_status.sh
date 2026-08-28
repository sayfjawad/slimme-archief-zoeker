#!/usr/bin/env bash
# Snapshot of the autonomous onboarding run. Safe to run any time, anywhere
# that can see the NAS share (c4130, hp-z8) or over ssh.
set -u
STATE=${STATE_FILE:-/data/SHARED/onboarding_state.json}
QUEUE=${QUEUE_CSV:-$(dirname "$0")/data/onboarding_queue.csv}

[ -f "$STATE" ] || { echo "no state file at $STATE -- orchestrator hasn't started"; exit 1; }

python3 - "$STATE" "$QUEUE" <<'PY'
import csv, json, sys, time
st = json.load(open(sys.argv[1]))
try:
    total = sum(1 for _ in csv.DictReader(open(sys.argv[2])))
except OSError:
    total = len(st["politicians"])
p = st["politicians"]
by = {}
for v in p.values():
    by[v["status"]] = by.get(v["status"], 0) + 1
done = by.get("done", 0)
print(f"queue {total}   done {done}   building {by.get('building',0)}   "
      f"pending {by.get('pending',0)}   failed {by.get('failed',0)}")
print(f"restarts {st.get('restarts',0)}   commits {st.get('commits',0)}   "
      f"since-restart {st.get('completed_since_restart',0)}")

now = time.time()
building = [(k, v) for k, v in p.items() if v["status"] == "building"]
if building:
    print("\nbuilding now:")
    for k, v in building:
        age = ""
        try:
            from datetime import datetime
            age = f"  {(now - datetime.fromisoformat(v['started']).timestamp())/60:.0f} min"
        except Exception:
            pass
        print(f"  {k:<20} on {v.get('worker','?'):<6}{age}")

fail = [(k, v) for k, v in p.items() if v["status"] == "failed"]
if fail:
    print("\nfailed (need a look):")
    for k, v in fail:
        print(f"  {k:<20} {v.get('error','')[:90]}")

recent = sorted((v for v in p.values() if v["status"] == "done" and v.get("finished")),
                key=lambda v: v["finished"])[-8:]
if recent:
    print("\nlast done:")
    for v in recent:
        print(f"  {v.get('person','?'):<28} {v.get('videos','?')} debates  "
              f"{v.get('person_chunks','?')} by them  ({v.get('dur_min','?')} min)")
if done >= total and not building:
    print("\n*** QUEUE COMPLETE ***  (orchestrator idling; add rows to re-arm)")
PY

echo
echo "orchestrator log:  ssh sayf@100.64.0.2 'journalctl --user -u politici-onboard -n 30 --no-pager'"
