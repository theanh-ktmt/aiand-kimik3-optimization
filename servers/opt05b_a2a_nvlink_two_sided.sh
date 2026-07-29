#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 5b — (DP8EP) All2All: flashinfer_nvlink_two_sided.
# The two-sided sibling of 5a; costs an extra handshake but can pipeline better
# at large batch. Worth a data point since 5a is the recipe's blanket advice.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt05b_a2a_nvlink_two_sided"
BENCH_MODE="spec"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --data-parallel-size 8
    --enable-expert-parallel
    --all2all-backend flashinfer_nvlink_two_sided
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
