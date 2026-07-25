#!/usr/bin/env bash
# Mirrors the shared transcript pool + every tracked person's own
# transcripts into archive-data/ inside this repo and commits+pushes any
# change. This is the transcripts themselves (the asset that actually
# matters -- real GPU-hours of transcription/diarization work), NOT the
# compiled index.sqlite/embeddings.npy: those are large binary blobs that
# regenerate from these transcripts in minutes (`build_index.py`) and would
# blow past GitHub's 100MB single-file limit, so they stay out of git and
# rely on backup_daily.sh's separate dated-backup mechanism instead.
#
# Added 2026-07-25 as a *third* durable copy of the transcripts (alongside
# the live host and backup_daily.sh's /data/backups/), after a rebuild
# briefly wiped a sibling project's live index with nothing to restore
# from except a lucky, undocumented copy found on another host -- git
# gives every version a permanent, off-host, diffable copy for free.
set -u
cd "$(dirname "$0")"
LOG=/data/WILDERS/commit_transcripts.log
exec >> "$LOG" 2>&1
echo "=== commit_transcripts $(date '+%F %T')"

mkdir -p archive-data/shared

SHARED=$(python3 -c 'from pipeline_config import load_config; print(load_config()["_paths"]["shared_transcripts"])')
rsync -a --delete "$SHARED/" archive-data/shared/

# every tracked politician is a config/<slug>.json -- no hardcoded name list,
# so a newly onboarded politician is picked up automatically the next run
for cfg in config/*.json; do
  [ -e "$cfg" ] || continue
  slug=$(basename "$cfg" .json)
  DATA=$(PERSON=$slug python3 -c 'from pipeline_config import load_config; print(load_config()["_paths"]["data"])')
  [ -d "$DATA/transcripts" ] || continue
  mkdir -p "archive-data/$slug"
  rsync -a --delete "$DATA/transcripts/" "archive-data/$slug/"
done

git add archive-data/
if git diff --cached --quiet; then
  echo "  geen wijzigingen"
else
  N=$(git diff --cached --stat | tail -1)
  if ! git commit -q -m "Daily transcript sync: $(date '+%F')

$N

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"; then
    echo "  commit FAALDE (zie output hierboven, bv. ontbrekende git user.email/name)"
    exit 1
  fi
  git push && echo "  gecommit en gepusht" || echo "  push FAALDE"
fi
