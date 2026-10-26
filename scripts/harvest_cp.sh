#!/bin/bash
# Refresh the CORRECTED grid (results/master_grid_cp{,_ood}.json) without disturbing the "everything" grid the
# frozen draft reads (2026-09-23, STATUS v1.107/v1.118). master_grid.py always writes master_grid{_ood}.json, so
# the _cp files are made by a SC_GCF_TAG=_cp4 harvest followed by a copy, and the untagged files are restored
# from a backup taken first. Backups accumulate under results/_bk_<epoch>/ (never deleted by this script).
# Usage: bash scripts/harvest_cp.sh          (takes ~1-2 min; run on the login node)
set -e
cd "${SC_PROJECT:-$(pwd)}"
export SC=$PWD HF_HOME=~/.cache/huggingface
command -v module >/dev/null 2>&1 && module load cuda/toolkit/12.8.1-1 python/3.12.3 >/dev/null 2>&1 || true || true   # cluster-specific
source .venv/bin/activate
BK=results/_bk_$(date +%s); mkdir -p "$BK"
for f in master_grid.json master_grid_runs.json master_grid.md master_grid_ood.json master_grid_ood_runs.json; do
  [ -f results/$f ] && cp results/$f "$BK/"; done
SC_GCF_TAG=_cp4 python scripts/master_grid.py | tail -1
cp results/master_grid.json results/master_grid_cp.json
cp results/master_grid_runs.json results/master_grid_cp_runs.json
SPLIT=test_ood SC_GCF_TAG=_cp4 python scripts/master_grid.py | tail -1 || true
[ -f results/master_grid_ood.json ] && cp results/master_grid_ood.json results/master_grid_cp_ood.json
for f in master_grid.json master_grid_runs.json master_grid.md master_grid_ood.json master_grid_ood_runs.json; do
  [ -f "$BK/$f" ] && cp "$BK/$f" results/$f; done
echo "[harvest_cp] master_grid_cp.json refreshed $(date +%H:%M); untagged grid restored from $BK"
