#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 4d — MoE backend: triton (portable reference path; usually the slowest,
# useful as the floor and as a fallback if a fused kernel miscompiles on MXFP4).
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt04d_moe_triton"
BENCH_MODE="spec"
MOE_BACKEND="triton"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
