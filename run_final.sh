#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_final.sh — FULL reference benchmark for the final comparison:
#   baseline + final1 (TP8) + final2 (DP8EP), each a FULL sweep, then a clean
#   3-way combined CSV.
#
# All three are mtp-mode configs, so each gets BOTH lanes:
#   random 1k/1k + 8k/1k  (InferenceX-comparable) and
#   ShareGPT OSL 1024 + 256 (representative spec-decode acceptance),
# at conc 1..128. Add ref_nonmtp to the set if you also want the no-spec floor
# on the same full grid.
#
# Reuses run_all.sh (per-config timeout, GPU reap, W&B sync per config) with
# --only + full sweeps, so each config is launched, benchmarked, and torn down
# with the same resilience as the screening campaign.
#
# Usage:
#   bash run_final.sh
#   SKIP_EXISTING=1 bash run_final.sh             # skip any of the three already done
#   CONFIGS="baseline final1" bash run_final.sh    # override the set
#   DATASETS=random bash run_final.sh              # random lane only
#   CONFIGS="baseline final1 final2 ref_nonmtp" bash run_final.sh
# ---------------------------------------------------------------------------
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

read -r -a CONFIGS <<< "${CONFIGS:-baseline final1 final2}"
export DATASETS="${DATASETS:-}"     # empty = per-config default (see bench/bench.sh)

echo "### FINAL FULL BENCH: ${CONFIGS[*]} (full sweep each, lanes=${DATASETS:-auto}) ###"
# Full sweep for every config here (baseline AND the finals).
OPT_SWEEP=full BASELINE_SWEEP=full bash "$REPO_ROOT/run_all.sh" --only "${CONFIGS[@]}"

echo "### combined final CSV (vs baseline) ###"
dirs=()
for c in "${CONFIGS[@]}"; do dirs+=("$REPO_ROOT/results/$c"); done
python3 "$REPO_ROOT/aggregate.py" "${dirs[@]}" \
    --baseline baseline --out "$REPO_ROOT/results/final_full.csv"
echo "Combined: results/final_full.csv"
