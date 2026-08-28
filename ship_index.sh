#!/usr/bin/env bash
# c4130 now runs the full pipeline AND serves the index locally (2026-08-23),
# so "shipping" a freshly rebuilt index is just restarting the local system
# service. (Formerly this rsync'd index.sqlite + embeddings.npy Z8 -> c4130
# and restarted the service there over SSH.)
#
# Usage: ship_index.sh <slug>   (e.g. wilders, yesilgoz)
#
# Restarts every ENABLED search unit that would serve this rebuild:
#   - politicus-search   the combined multi-politician app (reloads EVERY
#     politician's index on start, so one restart after any person's rebuild
#     is enough -- a few redundant restarts a night are harmless)
#   - <slug>-search      the legacy single-person unit, only if still enabled
#     (both are disabled since the 2026-08-28 merge -> the subdomains now
#     301-redirect to politicus.zoek-r.nl -- so this normally no-ops)
set -u
slug=${1:?usage: ship_index.sh <slug>}

for unit in politicus-search "$slug-search"; do
  if systemctl is-enabled --quiet "$unit" 2>/dev/null; then
    sudo systemctl restart "$unit" && echo "restarted $unit" || echo "FAILED restart $unit"
  fi
done
