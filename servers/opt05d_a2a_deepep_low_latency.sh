#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 5d — (DP8EP) All2All: deepep_low_latency (decode-oriented DeepEP mode).
# Requires DeepEP to be built into the image.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt05d_a2a_deepep_low_latency"
BENCH_MODE="spec"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --data-parallel-size 8
    --enable-expert-parallel
    --all2all-backend deepep_low_latency
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
