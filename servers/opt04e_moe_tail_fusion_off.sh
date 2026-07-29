#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 4e — LatentMoE tail fusion OFF (VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=0).
#
# K3's "Stable LatentMoE" projects experts through a 3584-dim latent space; the
# recipe turns on a K3-specific fusion for that tail on every NVIDIA target.
# This config quantifies what that fusion is actually worth (and gives you an
# escape hatch if it turns out to be numerically unstable on MXFP4).
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt04e_moe_tail_fusion_off"
BENCH_MODE="spec"
export VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=0
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
