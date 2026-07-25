#!/usr/bin/env bash
# Build a working search index straight from the transcripts checked into
# this repo (archive-data/) -- no TK/Handelingen download, no YouTube sync,
# no whisperx/diarization needed. Useful to try the app, or to rebuild an
# index from scratch if a live deployment's compiled index (index.sqlite +
# embeddings.npy, deliberately NOT checked in -- see archive-data/README.md)
# is ever lost: those regenerate from these transcripts in minutes.
#
# Usage: ./quickstart.sh [slug]   (default: wilders; try yesilgoz too)
#
# Everything lands under ./local-data/, next to this script -- doesn't
# touch /data or any other path a production deployment might use.
set -eu
cd "$(dirname "$0")"
SLUG=${1:-wilders}

if [ ! -f "config/${SLUG}.json" ]; then
  echo "no config/${SLUG}.json -- available: $(ls config/*.json | xargs -n1 basename | sed 's/\.json$//' | tr '\n' ' ')" >&2
  exit 1
fi
if [ ! -d "archive-data/${SLUG}" ] || [ ! -d "archive-data/shared" ]; then
  echo "archive-data/${SLUG} or archive-data/shared missing -- did you clone with the archive-data/ directory intact?" >&2
  exit 1
fi

export SHARED_DIR="$PWD/local-data/SHARED"
export DATA_DIR="$PWD/local-data/${SLUG}"
export PERSON="$SLUG"

mkdir -p "$SHARED_DIR/transcripts" "$DATA_DIR/transcripts"
echo "copying archive-data/shared -> $SHARED_DIR/transcripts"
cp -r archive-data/shared/. "$SHARED_DIR/transcripts/"
echo "copying archive-data/${SLUG} -> $DATA_DIR/transcripts"
cp -r "archive-data/${SLUG}/." "$DATA_DIR/transcripts/"

echo "building the index (this embeds every chunk with BGE-M3 -- uses a GPU if"
echo "one is available via torch.cuda, otherwise falls back to CPU and is slow)"
python3 build_index.py "$SLUG"

cat <<EOF

Done. Index built at $DATA_DIR/index/.

Start the app with:
  SHARED_DIR="$SHARED_DIR" DATA_DIR="$DATA_DIR" PERSON="$SLUG" python3 -m uvicorn app:app --port 8000

Then open http://localhost:8000 -- search and the AI-summary work; local
audio/video playback won't, since the raw media files aren't checked into
the repo (only the transcripts are). Playback links to the original source
(YouTube, Tweede Kamer) still work.
EOF
