#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 5e — (DP8EP) All2All: deepep_high_throughput (prefill-oriented DeepEP mode).
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt05e_a2a_deepep_high_throughput"
BENCH_MODE="spec"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --data-parallel-size 8
    --enable-expert-parallel
    --all2all-backend deepep_high_throughput
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
