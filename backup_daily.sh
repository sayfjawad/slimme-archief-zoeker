#!/usr/bin/env bash
# Daily backup of the shared transcript pool + every tracked person's own
# transcripts/index, before any sync/rebuild touches them. Added 2026-07-24
# after a build_index.py run on abo-ali-search wiped its live index with no
# way back except a lucky, undocumented copy found on another host --
# transcription/diarization is real compute cost, so its output must never
# be only one bad run away from unrecoverable. Idempotent (one backup per
# calendar day); retention: keep last 14 days.
set -u
cd "$(dirname "$0")"
BACKUP_ROOT=/data/backups/wilders-search
RETENTION_DAYS=14
LOG=/data/WILDERS/backup.log
exec >> "$LOG" 2>&1

TODAY=$(date +%F)
DEST="$BACKUP_ROOT/$TODAY"
if [ -d "$DEST" ]; then
  echo "=== backup_daily $(date '+%F %T') -- $TODAY bestaat al, sla over"
else
  echo "=== backup_daily $(date '+%F %T') -> $DEST"
  mkdir -p "$DEST/shared"
  SHARED=$(python3 -c 'from pipeline_config import load_config; print(load_config()["_paths"]["shared_transcripts"])')
  rsync -a "$SHARED/" "$DEST/shared/"
  echo "  shared pool: $(ls "$DEST/shared" | wc -l) bestanden"

  for slug in wilders yesilgoz; do
    DATA=$(PERSON=$slug python3 -c 'from pipeline_config import load_config; print(load_config()["_paths"]["data"])')
    [ -d "$DATA" ] || continue
    mkdir -p "$DEST/$slug"
    rsync -a "$DATA/transcripts/" "$DEST/$slug/transcripts/" 2>/dev/null
    rsync -a "$DATA/index/index.sqlite" "$DATA/index/embeddings.npy" "$DEST/$slug/" 2>/dev/null
    echo "  $slug: $(ls "$DEST/$slug/transcripts" 2>/dev/null | wc -l) eigen transcripten + index"
  done
fi

find "$BACKUP_ROOT" -maxdepth 1 -type d -name '20*' -mtime +$RETENTION_DAYS -exec rm -rf {} \;
