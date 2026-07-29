#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 6 — MLA attention kernel: --attention-backend FLASHMLA.
#
# Only affects the 24 Gated MLA layers (the other 69 are KDA — see opt05).
# Uses the CLI flag rather than VLLM_ATTENTION_BACKEND so preflight can verify it
# and it lands in the launch command recorded at the head of server.log.
#
# Other non-sparse MLA backends this build exposes, if FLASHMLA disappoints:
#   FLASHINFER_MLA (what the recipe pins for the DSpark verify step),
#   CUTLASS_MLA, TRITON_MLA, FLASH_ATTN_MLA.
# The *_SPARSE variants are for sparse-MLA models (DeepSeek DSA / GLM-5.2), not
# K3 gated MLA — do not use them here.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt06_attn_flashmla"
BENCH_MODE="mtp"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --attention-backend "${ATTN_BACKEND:-FLASHMLA}"
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
