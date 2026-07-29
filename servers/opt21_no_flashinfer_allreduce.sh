#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 21 — FlashInfer allreduce OFF (VLLM_ALLREDUCE_USE_FLASHINFER=0).
#
# Base-preset item, tested in the "skip it" direction: falls back to vLLM's own
# custom all-reduce. At TP8 across 93 layers there are a lot of allreduces per
# token, so if FlashInfer is NOT the faster path on B300 this is a free win — and
# if it is, the gap tells you how much of the preset's benefit is collective
# kernels rather than compute kernels.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="${CONFIG:-opt21_no_flashinfer_allreduce}"
BENCH_MODE="mtp"
export VLLM_ALLREDUCE_USE_FLASHINFER=0
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
