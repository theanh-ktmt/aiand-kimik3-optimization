#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 4b — MoE backend: flashinfer_trtllm (TRT-LLM fused MoE via FlashInfer).
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt04b_moe_flashinfer_trtllm"
BENCH_MODE="spec"
MOE_BACKEND="flashinfer_trtllm"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
