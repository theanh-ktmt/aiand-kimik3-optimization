#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 3b — Attention backend: FLASHMLA (DeepSeek FlashMLA kernels) for the
# Gated MLA layers instead of FlashInfer.
# Run `preflight.sh` first: it prints the AttentionBackendEnum names this build
# actually exposes, since the sparse/dense MLA variants get renamed often.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt03b_attn_flashmla"
BENCH_MODE="spec"
export VLLM_ATTENTION_BACKEND=FLASHMLA
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
