#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 20 — LatentMoE tail fusion OFF (VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=0).
#
# Another base-preset item tested in the "skip it" direction. K3's Stable
# LatentMoE projects experts through a 3584-dim latent space, and the preset turns
# on a K3-specific fusion for that tail. This config prices the fusion.
#
# It also doubles as the first thing to try if MoE output looks numerically wrong
# on MXFP4 — a fused tail is exactly where a precision bug would hide, so having
# a ready one-flag bypass is worth a config on its own.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="${CONFIG:-opt20_no_tail_fusion}"
BENCH_MODE="mtp"
export VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=0
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
