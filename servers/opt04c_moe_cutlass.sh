#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 4c — MoE backend: cutlass.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt04c_moe_cutlass"
BENCH_MODE="spec"
MOE_BACKEND="cutlass"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
