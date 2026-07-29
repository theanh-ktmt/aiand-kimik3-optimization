#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 13 — AllReduce: VLLM_ALLREDUCE_USE_FLASHINFER=0 (the recipe sets 1).
# At TP8 across 93 layers there are a lot of allreduces per token; this isolates
# whether the FlashInfer path actually beats vLLM's custom all-reduce on B300.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt13_no_flashinfer_allreduce"
BENCH_MODE="spec"
export VLLM_ALLREDUCE_USE_FLASHINFER=0
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
