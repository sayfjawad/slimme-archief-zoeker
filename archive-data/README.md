# archive-data/

The actual transcripts — real transcription/diarization work, not
regenerable in minutes the way the compiled search index is. Committed here
as a third durable copy (alongside the live host and the daily
`/data/backups/` snapshot) after a rebuild once wiped a sibling project's
live index with no way back except a lucky copy found on an undocumented
third host. See `../docs/postmortem.md` (section 5.11) for the full story.

- `shared/` — the shared multi-speaker debate pool (TK verslagen +
  Handelingen), reused across every tracked politician.
- `<slug>/` — that politician's own transcripts (their YouTube channel,
  diarized), matching `config/<slug>.json`.

**What's deliberately *not* here:** `index.sqlite` / `embeddings.npy`. Those
are large compiled binaries (embeddings.npy alone routinely exceeds
GitHub's 100MB file limit) that `build_index.py` regenerates from exactly
these transcripts in a few minutes to low tens of minutes — see
`../quickstart.sh` to do that yourself, or `../README.md`'s "Quickstart
from the checked-in archive" section.

Kept in sync daily by `../commit_transcripts.sh` (cron), which mirrors from
the live pipeline's output directories and commits only what changed.
