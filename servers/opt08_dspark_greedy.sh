#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 8 — DSpark(7) with greedy draft sampling instead of probabilistic.
#
# The recipe pins draft_sample_method=probabilistic. Greedy drafting is cheaper
# per draft token and often raises acceptance for low-temperature / benchmark
# traffic (which is what --ignore-eos sweeps and MMLU-Pro both are), at the cost
# of draft diversity. Cheap to test, occasionally a free win.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt08_dspark_greedy"
BENCH_MODE="spec"
DRAFT_SAMPLE_METHOD="greedy"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
