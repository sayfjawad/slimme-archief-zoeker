#!/usr/bin/env bash
# Onboard the next N politicians from data/onboarding_queue.csv, in order,
# skipping any that already have a config/. Serial (shared-pool writes).
#
#   ./onboard_batch.sh [N]         default N = 10
#
# After the batch: restart politicus-search ONCE, verify every new slug is
# served with person_chunks>0, then commit_transcripts.sh (pushes the grown
# shared pool to GitHub as backup #3).
set -u
cd "$(dirname "$0")"
N=${1:-10}
QUEUE=data/onboarding_queue.csv
PY=/data/abo-ali-search/.venv/bin/python3
[ -f "$QUEUE" ] || { echo "no $QUEUE -- run rank_speakers.py + build_onboarding_queue.py first"; exit 1; }

done_count=0
new_slugs=()
# columns: rank,slug,verslagnaam,voornaam,match_achternaam,...
while IFS=, read -r rank slug rest; do
  [ "$rank" = "rank" ] && continue
  [ "$done_count" -ge "$N" ] && break
  if [ -f "config/$slug.json" ]; then
    continue
  fi
  echo; echo "================ onboarding #$rank  $slug ================"
  if /data/abo-ali-search/.venv/bin/python3 onboard_politician.py "$slug"; then
    new_slugs+=("$slug")
    done_count=$((done_count + 1))
  else
    echo ">>> $slug FAILED -- stopping the batch. Fix, then re-run."
    break
  fi
done < "$QUEUE"

if [ "${#new_slugs[@]}" -eq 0 ]; then
  echo "nothing new onboarded."
  exit 0
fi

echo; echo "restarting politicus-search for: ${new_slugs[*]}"
sudo systemctl restart politicus-search
for _ in $(seq 1 30); do curl -sf localhost:8905/api/persons >/dev/null && break; sleep 2; done

fail=0
for slug in "${new_slugs[@]}"; do
  pc=$(curl -s "localhost:8905/api/stats?person=$slug" | "$PY" -c 'import sys,json;d=json.load(sys.stdin);print(d.get("person_chunks",0))' 2>/dev/null || echo 0)
  if [ "${pc:-0}" -gt 0 ]; then
    echo "  OK   $slug  person_chunks=$pc"
  else
    echo "  FAIL $slug  person_chunks=$pc"; fail=1
  fi
done
[ "$fail" = 0 ] || { echo "verification failed for one or more slugs -- NOT committing"; exit 1; }

echo; echo "committing grown shared transcript pool"
./commit_transcripts.sh
echo; echo "batch done: ${#new_slugs[@]} politicians onboarded and served."
