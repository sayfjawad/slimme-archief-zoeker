#!/usr/bin/env bash
# c4130 now runs the full pipeline AND serves the index locally (2026-08-23),
# so "shipping" a freshly rebuilt index is just restarting the local system
# service. (Formerly this rsync'd index.sqlite + embeddings.npy Z8 -> c4130
# and restarted the service there over SSH.)
#
# Usage: ship_index.sh <slug>   (e.g. wilders, yesilgoz)
set -u
slug=${1:?usage: ship_index.sh <slug>}
sudo systemctl restart "$slug-search"
