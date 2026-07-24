# slimme-archief-zoeker

A smart, context-aware search engine over **Dutch politicians' public
record** — parliamentary transcripts, historical Handelingen, and video
appearances, searchable by meaning (not just keywords), with source, date
and a playback link that jumps to the exact moment. Configured per person
(`config/<slug>.json`); Kamerverslagen, Handelingen and debate video are
shared across every tracked politician's config, so onboarding a new one
reuses whatever the archive already has instead of redownloading it (see
"Multiple politicians, one shared archive" below).

Two live instances so far:
- **https://wilders.scrib-r.com** — Geert Wilders
- **https://yesilgoz.scrib-r.com** — Dilan Yeşilgöz

Same architecture and transcript/index format as
[abo-ali-search](https://github.com/sayfjawad/abo-ali-search).

![Politician under a magnifying glass](static/hero.webp)

Ask a question in plain language and get the matching fragments across three
decades of parliamentary records and videos — each with date, timestamp,
speaker, source link and a play button that jumps to the exact moment. An
optional local LLM writes a summary with verifiable citations (RAG). No cloud
APIs anywhere: embeddings, ASR and the LLM all run on own GPUs.

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

- Each politician gets their own systemd **user** service (linger enabled,
  its own port) so it survives reboots: `systemctl --user {status,restart}
  <slug>-search`, e.g. `wilders-search` (8902), `yesilgoz-search` (8903).
- A small edge VPS runs nginx with TLS (certbot) and proxies each politician's
  subdomain to the GPU host over a private Tailscale network, with
  streaming-friendly proxy settings for the media endpoints. `*.scrib-r.com`
  is wildcard-DNS'd, so a new politician's subdomain needs no DNS work.
- `resume.sh` loops over every tracked person for sync/video/app recovery.

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
