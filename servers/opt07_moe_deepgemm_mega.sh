#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 7 — MoE backend: auto -> deep_gemm_mega_moe.
#
# The recipe recommends it explicitly ("use deep_gemm_mega_moe for any DEP
# environment") and hardcodes it in its own decode profile, so it is the leading
# MoE hypothesis rather than a guess. Tested on TP8 so the kernel is the only
# variable; re-test on top of the winning parallelism if it lands.
#
# Other values available in this build if it disappoints: flashinfer_trtllm,
# flashinfer_cutlass, flashinfer_cutedsl, cutlass, triton, triton_unfused.
# marlin is a Hopper-only override in the recipe — not for B300.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt07_moe_deepgemm_mega"
BENCH_MODE="mtp"
export VLLM_USE_DEEP_GEMM=1
MOE_BACKEND="${MOE_BACKEND:-deep_gemm_mega_moe}"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
