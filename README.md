# slimme-archief-zoeker

A smart, context-aware search engine over **Dutch politicians' public
record** — parliamentary transcripts, historical Handelingen, and video
appearances, searchable by meaning (not just keywords), with source, date
and a playback link that jumps to the exact moment. Configured per person
(`config/<slug>.json`); Kamerverslagen, Handelingen and debate video are
shared across every tracked politician's config, so onboarding a new one
reuses whatever the archive already has instead of redownloading it (see
"Multiple politicians, one shared archive" below).

Live at **https://politicus.zoek-r.nl** — one app, the politician picked from
a dropdown (`/api/persons` drives it; `?person=<slug>` deep-links). Currently
Geert Wilders and Dilan Yeşilgöz.

The old per-politician subdomains **301-redirect** here (since 2026-08-28):
`wilders.scrib-r.com` → `?person=wilders`, `yesilgoz.scrib-r.com` →
`?person=yesilgoz`. The `<slug>-search` units still exist on disk but are
stopped and disabled.

Same architecture and transcript/index format as
[abo-ali-search](https://github.com/sayfjawad/abo-ali-search).

![Politician under a magnifying glass](static/hero.webp)

Ask a question in plain language and get the matching fragments across three
decades of parliamentary records and videos — each with date, timestamp,
speaker, source link and a play button that jumps to the exact moment. An
optional local LLM writes a summary with verifiable citations (RAG). No cloud
APIs anywhere: embeddings, ASR and the LLM all run on own GPUs.

## Quickstart from the checked-in archive

The actual transcripts — the expensive part, real transcription/diarization
work — are committed to this repo under `archive-data/` (see
`archive-data/README.md`). That means you can get a working, searchable app
running **without** downloading anything from the Tweede Kamer, YouTube, or
running whisperx/pyannote at all:

```bash
git clone https://github.com/sayfjawad/slimme-archief-zoeker
cd slimme-archief-zoeker
pip install -r requirements.txt   # or: fastapi uvicorn torch transformers numpy

./quickstart.sh wilders            # or: yesilgoz
# builds local-data/wilders/index/ from archive-data/, using a GPU if one's
# visible to torch, otherwise CPU (slower, still works)

SHARED_DIR="$PWD/local-data/SHARED" DATA_DIR="$PWD/local-data/wilders" \
  PERSON=wilders python3 -m uvicorn app:app --port 8000
```

Open `http://localhost:8000` — search and the AI summary work fully; local
audio/video playback doesn't, since the raw media files (large, and not
everyone's to redistribute) aren't checked in, only the transcripts derived
from them. Every result still links out to its original source (YouTube,
the Tweede Kamer's own site) instead.

`quickstart.sh` is also the fastest way to recover a *production* index if
one is ever lost or corrupted (`build_index.py <slug>` alone does the same
thing against whatever's already in the real data directories) — this is
exactly how abo-ali-search's live index was restored after the incident
documented in `docs/postmortem.md`.

## Sources

1. **Tweede Kamer verslagen** (official, corrected transcripts, 2013–now) via
   the open OData API (`gegevensmagazijn.tweedekamer.nl`, no key). vlos 2.0
   XML is parsed into per-vergadering transcripts; only vergaderingen where
   the person speaks are kept. Segments carry a `wallclock` timestamp
   (markeertijdbegin) used to link into debate video.
2. **Handelingen 1995–2013** from `officielebekendmakingen.nl` (SRU API), so
   coverage starts at the person's first year in parliament.
3. **YouTube channels** (e.g. the party channel) — audio-only opus via
   yt-dlp, transcribed with whisperx (ASR, optionally sharded over several
   GPU hosts with `remote_worker.sh`).
4. **Debat Direct video** (plenary sessions, ~2010–now). The agenda API gives
   per-day debates with start/end wallclocks; the debate's HLS `vodUrl` only
   yields real footage when windowed with `?start=&end=` (otherwise you get a
   "nomeeting" stub). Lowest variant (320×180 + separate audio rendition) is
   stream-copied with ffmpeg — ~130 MB per debate hour, no re-encode.

## Pipeline

```bash
python3 tk_sync.py            # incremental OData sync -> <data>/tk/xml/
python3 tk_parse.py           # vlos XML -> <data>/transcripts/tk_*.json
python3 ob_sync.py            # 1995-2013 Handelingen XML (SRU)
python3 ob_parse.py           #   -> <data>/transcripts/ob_*.json
python3 yt_sync.py            # yt-dlp audio+info-json -> <data>/youtube/
python3 transcribe_batch.py   # whisperx (--shard i/n, --diarize) -> yt_*.json
python3 dg_sync.py            # Debat Direct video -> <data>/debatgemist/
python3 build_index.py        # BGE-M3 embeddings -> <data>/index/
./run.sh                      # FastAPI app (default port 8902)
```

All scripts take an optional person slug argument (default `wilders`). Every
step is incremental/idempotent: OData `GewijzigdOp` cursor, yt-dlp download
archive, file-exists + state.json checks — rerunning is always safe, and
rerunning `tk_parse.py` upgrades transcripts as corrected verslagen appear.

## Distributed video download

The Debat Direct CDN paces individual HLS streams, so the archive is fetched
with N parallel shards spread over multiple hosts (see `hosts.env.example`):

- `dg_distributed.sh` — idempotent orchestrator: starts missing local shards,
  pushes `dg_sync.py` + a dates manifest to remote hosts (only ffmpeg +
  python3 + ssh needed there) and starts them detached, plus the puller.
- `dg_pull.sh` — drains finished files from the remotes (rsync
  `--remove-source-files`, partial downloads excluded) and merges the
  per-shard state files into `state.json`, which the app uses to map a
  transcript `wallclock` to a local video file + offset.
- `dg_sync.py --shard i/n --dates-json … --have …` — a shard worker; `--have`
  prevents re-downloading files that already live on the main host.

## Self-healing operation & reporting

- `resume.sh` (cron `@reboot`) restarts every sync, the distributed download,
  the app service and the milestone watchers after a crash or power loss;
  everything resumes from its own state.
- `milestone_watch.sh {text|youtube|video|transcribe}` — detached watchers
  that rebuild the index, restart the app and send milestone e-mails
  (`notify.py`) when a phase completes.
- `status.py` (cron, 15 min) appends a machine-readable snapshot to
  `progress.log`; `progress_report.py` (cron: hourly file, daily mail) turns
  that history into a human progress report with download rate and a
  completion prognosis.

## Multiple politicians, one shared archive

A parliamentary debate is inherently multi-speaker; a politician's own
YouTube channel is not (and may feature other people entirely). So only
what's objectively shared is shared:

- **Shared** (`/data/SHARED`, `SHARED_DIR` env var): TK verslag + Handelingen
  XML, the multi-speaker transcripts parsed from them (`tk_parse.py`/
  `ob_parse.py` already keep every speaker in a vergadering, not just the
  configured person, and skip re-parsing one already in the pool), and
  Debat Direct video (keyed by date+slug, reused as-is by anyone who spoke
  in that session). Each shared transcript's metadata carries a `speakers`
  list, so `build_index.py` can filter the pool down to "debates this
  person was actually in" per config.
- **Per person** (`<data_dir>` in their config): their own YouTube channel(s)
  audio + ASR transcripts, and their own search index/app instance — never
  shared, since there's no cross-person attribution inside it.

Onboarding a new politician (~a few hours of hands-on work, not days):
persoon-id via TK OData (`contains(Achternaam,'...')`), verify their party's
YouTube channel (don't guess — confirm it), write `config/<slug>.json`,
`tk_parse.py <slug>` + `ob_parse.py <slug>` (usually mostly free reuse from
the shared pool), `PERSON=<slug> ./dg_distributed.sh` for their few missing
debate videos, `PERSON=<slug> python3 yt_sync.py` + diarized
`transcribe_batch.py`, `PERSON=<slug> python3 build_index.py`, then a
`<slug>-search.service` unit + nginx vhost + certbot on the edge host. In
practice Yeşilgöz reused 326 of 529 relevant vergaderingen and 85 of 279
debate days for free from Wilders' already-downloaded data.

Every rebuild (`build_index.py`, both this repo's and abo-ali-search's) now
backs up the current index to `index/backups/<date>/` first, and
`backup_daily.sh` mirrors the transcript pool + each person's own data to
`/data/backups/` daily -- added after a rebuild once wiped a sibling
project's live index with nothing to restore from.

## Search app

`app.py` + `static/index.html`: FastAPI, semantic search (GPU), date and
"only statements by <person>" filters, `/api/ask` RAG endpoint (auto-detects
a local llama.cpp server), NL/EN interface, audio/video playback at the
matched timestamp. Sources are tagged official record vs ASR so quotes can
always be verified against the original.

## Transcript format (abo-ali compatible)

- `<base>.json`: `{"title", "duration_seconds", "segments": [{speaker_id,
  speaker, start, end, text}]}` — TK segments additionally have `wallclock`.
- `<base>.metadata.json`: `{"id", "title", "url", "upload_date",
  "duration_seconds", "source": "tk_verslag" | "ob_handeling" |
  "youtube:<channel>", "transcript_source": "official" | "asr"}`.

## Deployment

- Checkout lives at `/data/politicus-search` on the GPU host (c4130).
- **Combined app**: one systemd service `politicus-search` (port 8905, no
  `PERSON=` env) serves every `config/*.json`; the visitor picks the politician
  from a dropdown. Unit + edge nginx vhost are in `deploy/`. CPU-only search,
  RAG via the shared llama.cpp server.
- **`PERSON=<slug>` env** pins a process to one politician (only their index
  loaded, no dropdown, cross-person requests 404). The legacy `<slug>-search`
  units used this; they are now stopped + disabled (kept on disk) and the
  subdomains 301-redirect to `politicus.zoek-r.nl`.
- A small edge VPS (`vmi2702091`) runs nginx with TLS (certbot) and proxies
  `politicus.zoek-r.nl` to the GPU host over a private Tailscale network, with
  streaming-friendly proxy settings for the media endpoint.
- After any person's index is rebuilt, `ship_index.sh <slug>` restarts every
  *enabled* search unit that serves it — normally just `politicus-search`,
  which reloads every politician's index. `resume.sh` also ensures it is up.
- Onboarding a new politician: add `config/<slug>.json` + build their index,
  then restart `politicus-search` — they appear in the dropdown. A dedicated
  subdomain (optional) additionally needs a `<slug>-search` unit
  (`Environment=PERSON=<slug>`) + nginx vhost.

## Gotchas worth knowing

- OData: `$top` caps the *total* result count and suppresses
  `@odata.nextLink` — omit it and follow server paging.
- vlos XML: speaker-label alineaitems ("De **heer X**:") must be stripped;
  nested interruption text must not be double-counted; `spreker@objectid` is
  *not* the OData Persoon Id — match by name.
- Debat Direct: without the `?start=&end=` window the vodUrl returns a stub
  playlist; yt-dlp's `-f worst` picks a keyframes-only track — use ffmpeg
  with the explicit variant instead.
- ffmpeg ≥ 7/8 refuses the CDN's `.m4v` HLS segments unless
  `-allowed_extensions ALL -extension_picky 0` is set (auto-detected in
  `dg_sync.py`).
- YouTube bulk downloads without `-t sleep` get rate-limited within minutes.

## TODO

- Diarization for YouTube ASR (`transcribe_batch.py --diarize`, needs
  `HF_TOKEN`).
- Optional extra sources: interviews on broadcaster channels, X/Twitter video.
