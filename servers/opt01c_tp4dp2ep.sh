#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 1c — Parallelism: hybrid TP4 x DP2 + Expert Parallel.
# Middle ground between TP8 (lowest latency, most collectives) and DP8 (highest
# throughput, 8-way all2all): halves the TP collective width while keeping the
# all2all group small.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt01c_tp4dp2ep"
BENCH_MODE="spec"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 4
    --data-parallel-size 2
    --enable-expert-parallel
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
