#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 9 — Sampler: VLLM_USE_FLASHINFER_SAMPLER=1.
# K3's vocabulary is 160K, so the sampler's softmax/top-k work per step is ~25%
# larger than a 128K-vocab model's — the FlashInfer sampler has correspondingly
# more to save, and speculative decoding invokes it once per draft token.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt09_flashinfer_sampler"
BENCH_MODE="spec"
export VLLM_USE_FLASHINFER_SAMPLER=1
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
