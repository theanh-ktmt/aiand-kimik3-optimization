#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 3a — Attention backend: force FLASHINFER_MLA for the target model's decode.
#
# K3's attention is hybrid: 69 KDA (Kimi Delta Attention, linear) layers + 24
# Gated MLA layers. VLLM_ATTENTION_BACKEND selects the kernel for the MLA half;
# the KDA half is served by its own kernels regardless. FLASHINFER_MLA is also
# what the recipe pins for the DSpark verify step, so pinning it for the target
# too keeps a single kernel family in play.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt03a_attn_flashinfer_mla"
BENCH_MODE="spec"
export VLLM_ATTENTION_BACKEND=FLASHINFER_MLA
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
