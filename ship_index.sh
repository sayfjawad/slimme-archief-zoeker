#!/usr/bin/env bash
# Push a person's freshly (re)built index to the c4130 serving host and
# restart that host's system service there. The Z8 keeps the heavy pipeline
# (transcription/diarization = the scrib-r whisperx venv + GPU); c4130 only
# serves the prebuilt index over CPU, so every index rebuild or rollback on
# the Z8 must ship its result across and bounce the c4130 unit -- the Z8's
# own `systemctl --user` unit no longer exists (moved to .moved-20260823).
#
# Usage: ship_index.sh <slug>   (e.g. wilders, yesilgoz)
set -u
cd "$(dirname "$0")"
SERVE_HOST=100.64.0.13
slug=${1:?usage: ship_index.sh <slug>}

INDEX_DIR=$(PERSON="$slug" python3 -c 'from pipeline_config import load_config; print(load_config()["_paths"]["index"])') \
  || { echo "ship_index: load_config failed for $slug" >&2; exit 1; }

rsync -a --timeout=1800 "$INDEX_DIR/index.sqlite" "$INDEX_DIR/embeddings.npy" \
  "sayf@$SERVE_HOST:$INDEX_DIR/" \
  && ssh "sayf@$SERVE_HOST" sudo systemctl restart "$slug-search"
