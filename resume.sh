#!/usr/bin/env bash
# Idempotent recovery/resume for every tracked person's pipeline (see
# PERSONS below). Safe to run any time (boot, after power loss, manually):
# every sync script resumes from its own state (OData GewijzigdOp cursor,
# ob/dg state.json + file-exists checks, yt-dlp --download-archive), so
# double work is avoided. Starts only what is not already running.
# Log: /data/WILDERS/resume.log (shared control-plane log for all persons).
cd "$(dirname "$0")"
LOG=/data/WILDERS/resume.log
exec >> "$LOG" 2>&1
# make `systemctl --user` work from cron @reboot (no login session env)
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
echo "=== resume $(date '+%F %T')"
./backup_daily.sh

# slug:systemd-unit -- add a line here when a new person gets their own app
PERSONS="
wilders:wilders-search
yesilgoz:yesilgoz-search
"

start_if_absent() {  # <pgrep-pattern> <command...>
  local pattern=$1; shift
  if pgrep -f "$pattern" > /dev/null; then
    echo "  already running: $*"
  else
    nohup "$@" >> /data/WILDERS/pipeline.log 2>&1 &
    echo "  started (pid $!): $*"
  fi
}

# leftover partial video downloads from a crash (shared pool, only clear
# when nothing is actively downloading for ANY person)
DG=$(python3 -c 'from pipeline_config import load_config; print(load_config()["_paths"]["debatgemist"])')
pgrep -f 'python3 dg_sync\.py' > /dev/null || rm -f "$DG"/*.part.mp4

for spec in $PERSONS; do
  IFS=: read -r slug unit <<< "$spec"
  echo "--- $slug"

  # 1. text sources (each chains its parser; both incremental; shared
  # tk_xml/ob_xml pool means a second person's tk_sync/ob_sync mostly just
  # re-lists already-downloaded XML, but tk_parse/ob_parse still need a run
  # per person to pull their own vergaderingen into the shared transcript pool)
  start_if_absent "PERSON=$slug python3 (tk_sync|tk_parse)\.py" \
    env "PERSON=$slug" bash -c 'python3 tk_sync.py && python3 tk_parse.py'
  start_if_absent "PERSON=$slug python3 (ob_sync|ob_parse)\.py" \
    env "PERSON=$slug" bash -c 'python3 ob_sync.py && python3 ob_parse.py'

  # 2. youtube audio (rate-limit friendly; archive.txt makes it incremental)
  start_if_absent "PERSON=$slug python3 yt_sync\.py" env "PERSON=$slug" python3 yt_sync.py

  # 3. debate videos: distributed shards (local + remote hosts) + puller,
  # into the SHARED pool; dg_distributed.sh is itself idempotent and remote
  # workers survive our reboots. One person's run at a time (dg_sync's
  # local shard pgrep-guard would otherwise let two persons' orchestrators
  # collide), so only start this person's run if no dg_sync is active at all.
  if pgrep -f 'python3 dg_sync\.py' > /dev/null; then
    echo "  dg_sync already running for another person; skip video run for $slug this pass"
  else
    PERSON=$slug ./dg_distributed.sh
  fi

  # 4. search app -- runs as a systemd --user service (auto-starts at boot
  # via linger); this just makes sure it is up after a manual resume.
  index_db="$(PERSON=$slug python3 -c 'from pipeline_config import load_config; print(load_config()["_paths"]["index"] / "index.sqlite")')"
  if [ -f "$index_db" ]; then
    systemctl --user start "$unit" 2>/dev/null && echo "  app service ensured up ($unit)" \
      || echo "  systemctl start $unit faalde"
  else
    echo "  no index yet for $slug; app not started"
  fi
done

# 5. milestone watchers (Wilders-only mail milestones for now; see
# milestone_watch.sh -- not yet generalized per person)
for m in text youtube video; do
  if ! pgrep -f "milestone_watch\.sh $m" > /dev/null; then
    setsid nohup ./milestone_watch.sh "$m" < /dev/null > /dev/null 2>&1 &
    echo "  watcher '$m' started"
  else
    echo "  watcher '$m' already running"
  fi
done

for spec in $PERSONS; do
  IFS=: read -r slug unit <<< "$spec"
  PERSON=$slug python3 status.py > /dev/null && echo "  progress snapshot written ($slug)"
done
echo "=== resume done $(date '+%F %T')"
