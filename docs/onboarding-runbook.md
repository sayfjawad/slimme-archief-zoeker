# Onboarding a politician

*TK parliamentary text only. YouTube / Handelingen 1995-2013 are a separate,
later job — see "Deferred" at the bottom.*

Everything runs on **c4130** (`/data/politicus-search`, `100.64.0.13`). The
combined app `politicus-search` (:8905) serves every `config/*.json`.

## Safeguards — the shared transcript pool has 4 copies

`tk_parse.py` writes debate transcripts to the **shared** pool
(`/data/SHARED/transcripts/`, on rs815p-1). It is *append-only* — a debate
already parsed is skipped, never rewritten or deleted. Copies:

1. live — `rs815p-1:/volume1/archief-media/shared/transcripts` + DSM snapshots
2. `c4130:/data/backups/wilders-search/<date>/shared/` — daily, 14-day retention
3. GitHub `archive-data/shared/` — `commit_transcripts.sh`, daily + after each batch
4. `rs815p-2:/volume1/archief-media/shared/transcripts` — rsync mirror, cron `5,35 * * * *`

`build_index.py` backs the old index up to `<data>/index/backups/<date>/` before
every rebuild and **refuses to leave a >10 % smaller index in place** (restores
the backup, exits non-zero) unless `--force`. `rollback_index.py <slug>` restores
any dated backup.

Non-rotating pre-campaign snapshot (2026-08-28):
`c4130:/data/backups/pre-onboarding-2026-08-28/` and
`rs815p-1:/volume1/NetBackup/pre-onboarding/`.

## One politician

```bash
cd /data/politicus-search
git pull
# in the queue (data/onboarding_queue.csv):
.venv/bin/python3 onboard_politician.py <slug>
# not in the queue yet (e.g. a hand-pick):
.venv/bin/python3 onboard_politician.py <slug> --person "Mona Keijzer" \
    --voornaam Mona --achternaam Keijzer

sudo systemctl restart politicus-search
curl -s "localhost:8905/api/stats?person=<slug>"      # person_chunks must be > 0
./commit_transcripts.sh                                # push the grown pool
```

`onboard_politician.py` holds `flock(/tmp/daily_sync.lock)` (so it can't race
the 03:00 `daily_sync` cron), pre-checks that today's `backup_daily.sh` ran,
writes `config/<slug>.json` (tk-only), runs `tk_parse` then `build_index`, and
asserts the index has `videos>0`, `chunks>0`, and `>0` chunks spoken by the
match name. It does **not** touch the live service — you restart once, after.

## A batch

```bash
cd /data/politicus-search && git pull
./onboard_batch.sh 10        # next 10 un-onboarded queue rows, serially
```

Then it restarts `politicus-search` **once**, verifies every new slug, and runs
`commit_transcripts.sh`. Stops on the first failure with the config left for
inspection. Re-run to resume (already-onboarded slugs are skipped).

Do batches of ~10, eyeball the dropdown + a couple of searches, repeat.

## Rebuilding the queue

```bash
.venv/bin/python3 rank_speakers.py            # ~3 min, scans /data/SHARED/tk (7.5k debates)
.venv/bin/python3 build_onboarding_queue.py   # --per-year N (default 20)
$EDITOR data/onboarding_queue.csv             # review / reorder / trim
```

`rank_speakers.py` is read-only. Both output to `data/` (git-tracked).

## If something goes wrong

| symptom | do |
|---|---|
| `build_index` aborted "index shrank" | a parse regression. `rollback_index.py <slug>`, investigate `tk_parse`, don't `--force` blindly |
| new politician missing from dropdown | `journalctl -u politicus-search` — a bad `config/<slug>.json` fails startup for *everyone*; fix or `git checkout config/<slug>.json` and restart |
| `person_chunks == 0` after onboarding | wrong `match_achternaam`/`voornaam` — check `data/speaker_ranking.csv` for the exact spelling, fix the config, re-run `build_index.py <slug> --force` |
| shared pool looks wrong | restore copy #2/#3/#4 above; `git log archive-data/` shows every daily state |

## Deferred

- **Handelingen 1995-2013**: bulk `ob_sync` (currently person-filtered), then
  `rank_speakers` over `ob/*.xml`, extend the queue backwards.
- **YouTube per politician**: confirm the party channel, `yt_sync` +
  `transcribe_batch --diarize`. Add `"youtube": {"channels": [...]}` to the
  config; `daily_sync` picks it up automatically.
