#!/usr/bin/env bash
# c4130 now runs the full pipeline AND serves the index locally (2026-08-23),
# so "shipping" a freshly rebuilt index is just restarting the local system
# service. (Formerly this rsync'd index.sqlite + embeddings.npy Z8 -> c4130
# and restarted the service there over SSH.)
#
# Usage: ship_index.sh <slug>   (e.g. wilders, yesilgoz)
#
# Restarts:
#   - <slug>-search        the legacy single-person unit, if it still exists
#   - politicus-search     the combined multi-politician app, if it exists
#     (it reloads EVERY politician's index on start, so one restart after any
#     person's rebuild is enough -- a few redundant restarts a night are fine)
set -u
slug=${1:?usage: ship_index.sh <slug>}

unit_exists() { systemctl list-unit-files "$1.service" --no-legend | grep -q .; }

for unit in "$slug-search" politicus-search; do
  if unit_exists "$unit"; then
    sudo systemctl restart "$unit" && echo "restarted $unit" || echo "FAILED restart $unit"
  fi
done
