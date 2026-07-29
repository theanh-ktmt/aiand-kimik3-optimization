#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 1a — Parallelism: TP8 + Expert Parallel (TP8EP).
# With 896 routed experts, sharding experts across ranks instead of replicating
# the whole MoE per rank is the single biggest memory/bandwidth lever available
# inside one node.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt01a_tp8ep"
BENCH_MODE="spec"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --enable-expert-parallel
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
