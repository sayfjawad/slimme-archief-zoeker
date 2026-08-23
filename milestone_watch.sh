#!/usr/bin/env bash
# Watches the wilders-search pipeline and mails milestones via notify.py.
# Usage: ./milestone_watch.sh {text|youtube|video}
# Each mode is one detached watcher; see repo README. Logs: /data/WILDERS/milestones.log
cd "$(dirname "$0")"
LOG=/data/WILDERS/milestones.log
exec >> "$LOG" 2>&1
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}

running() { pgrep -f "$1" | grep -v $$ > /dev/null; }
log() { echo "$(date '+%F %T') [$MODE] $*"; }
DG=$(python3 -c 'from pipeline_config import load_config; print(load_config()["_paths"]["debatgemist"])')

index_stats() {
python3 - <<'EOF'
import sqlite3
db = sqlite3.connect('/data/WILDERS/index/index.sqlite')
v = db.execute('SELECT COUNT(*) FROM videos').fetchone()[0]
c = db.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
w = db.execute("SELECT COUNT(*) FROM chunks WHERE speaker LIKE '%Wilders%'").fetchone()[0]
mn, mx = db.execute('SELECT MIN(upload_date), MAX(upload_date) FROM videos').fetchone()
print(f"{v} bronnen, {c} fragmenten ({w} uitgesproken door Wilders), periode {mn}..{mx}")
EOF
}

restart_app() {
  # app now serves from c4130 (CPU-only); ship the index across and restart
  # it there (the Z8 --user unit is gone -- see ship_index.sh)
  ./ship_index.sh wilders \
    && log "app herstart op c4130 (wilders-search.service)" \
    || log "ship+restart faalde"
}

MODE=$1
case "$MODE" in

text)
  log "wacht op tekst-backfills (tk_sync/tk_parse/ob_sync/ob_parse)"
  while running 'python3 (tk_sync|tk_parse|ob_sync|ob_parse)\.py'; do sleep 120; done
  log "tekst-backfills klaar; index herbouwen"
  python3 build_index.py
  restart_app
  STATS=$(index_stats)
  python3 notify.py "[wilders-search] Mijlpaal 1: tekstarchief compleet & doorzoekbaar" \
"Alle officiële teksten zijn binnen en geïndexeerd.

$STATS

Bronnen: Tweede Kamer verslagen (OData, 2013-nu) + Handelingen officielebekendmakingen.nl (1995-2013).
De zoek-app op http://localhost:8902 draait nu op de volledige index.

Volgende stappen die nog lopen: YouTube-audio, debatvideo's."
  ;;

youtube)
  log "wacht op YouTube-audiodownload"
  sleep 60
  while running 'python3 yt_sync\.py'; do sleep 300; done
  N=$(ls /data/WILDERS/youtube/*.opus 2>/dev/null | wc -l)
  GB=$(du -sh /data/WILDERS/youtube 2>/dev/null | cut -f1)
  python3 notify.py "[wilders-search] Mijlpaal 2: YouTube-audio compleet" \
"De audiodownload van het PVV-kanaal is klaar: $N audiobestanden ($GB).

Volgende stap (nog niet gestart): whisperx-transcriptie op de V100's, daarna herindexeren.
Die stap claimt beide GPU's een tijd - start hem wanneer het uitkomt met:
  cd ~/git/wilders-search && python3 transcribe_batch.py && python3 build_index.py"
  ;;

video)
  log "wacht op gedistribueerde dg-run (lokale shards + remote workers + puller)"
  while running 'python3 dg_sync\.py' || running 'dg_pull\.sh' \
        || ls /data/WILDERS/.dg_remote_*.active > /dev/null 2>&1; do sleep 300; done
  N=$(ls "$DG"/*.mp4 2>/dev/null | grep -vc part)
  GB=$(du -sh "$DG" 2>/dev/null | cut -f1)
  restart_app
  python3 notify.py "[wilders-search] Mijlpaal 3: eerste batch debatvideo's binnen" \
"Eerste videorun klaar: $N debatsessies ($GB), laagste kwaliteit (320x180, ~123 MB/uur).
De app is herstart en koppelt zoekresultaten nu aan lokale video met het juiste tijdstip.

Tweede run start zodra de tekst-backfill klaar is (voor de nieuw ontdekte debatdagen)."
  log "wacht tot tekst-backfill klaar is voor run 2"
  while running 'python3 (tk_sync|tk_parse)\.py'; do sleep 300; done
  log "start tweede dg_sync-run"
  python3 dg_sync.py
  N=$(ls "$DG"/*.mp4 2>/dev/null | grep -vc part)
  GB=$(du -sh "$DG" 2>/dev/null | cut -f1)
  DISK=$(df -h /data | tail -1 | awk '{print $4}')
  restart_app
  python3 notify.py "[wilders-search] Mijlpaal 4: videodekking compleet" \
"Alle beschikbare debatvideo's zijn binnen: $N sessies ($GB). Vrije schijfruimte: $DISK.
De app is herstart; alle Kamerfragmenten vanaf ~2010 zijn nu afspeelbaar op het juiste moment.

Daarmee is de pipeline volledig: tekst 1995-nu, video ~2010-nu, YouTube-audio gereed voor transcriptie."
  ;;

transcribe)
  # local shards 0/4 + 1/4 on both V100s; remote workers (gx10, c4130) get
  # shards 2/4 and 3/4 via remote_worker.sh, which sets a .remote_*.active
  # marker. The final sweep (no --shard) picks up anything a worker left.
  log "start whisperx-transcriptie: lokale shards 0/4 (GPU0) + 1/4 (GPU1)"
  HF_HOME=/data/huggingface CUDA_VISIBLE_DEVICES=0 python3 transcribe_batch.py --shard 0/4 &
  P0=$!
  HF_HOME=/data/huggingface CUDA_VISIBLE_DEVICES=1 python3 transcribe_batch.py --shard 1/4 &
  P1=$!
  wait $P0 $P1
  log "lokale shards klaar; wachten op remote workers"
  while ls /data/WILDERS/.remote_*.active > /dev/null 2>&1; do sleep 300; done
  log "veegronde voor achtergebleven bestanden"
  HF_HOME=/data/huggingface CUDA_VISIBLE_DEVICES=0 python3 transcribe_batch.py
  N=$(ls /data/WILDERS/transcripts/yt_*.metadata.json 2>/dev/null | wc -l)
  log "transcriptie klaar ($N transcripten); index herbouwen"
  python3 build_index.py
  restart_app
  STATS=$(index_stats)
  python3 notify.py "[wilders-search] Mijlpaal 5: YouTube-transcripties doorzoekbaar" \
"whisperx is klaar: $N video-transcripties toegevoegd en geherindexeerd.

$STATS

De app is herstart; YouTube-fragmenten zijn nu doorzoekbaar met directe
YouTube-links (op tijdstip) en lokale audio-playback."
  ;;

*)
  echo "usage: $0 {text|youtube|video|transcribe}"; exit 1 ;;
esac
log "klaar"
