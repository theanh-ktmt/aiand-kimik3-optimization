#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 4a — MoE backend: deep_gemm_mega_moe (instead of the recipe's `auto`).
# The recipe recommends it explicitly: "use deep_gemm_mega_moe for any DEP
# environment", and its own decode profile hardcodes it. Pairs naturally with
# opt01b/opt05 (expert parallel), but tested here on TP8 first so the MoE kernel
# is the only variable.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt04a_moe_deepgemm_mega"
BENCH_MODE="spec"
export VLLM_USE_DEEP_GEMM=1
MOE_BACKEND="deep_gemm_mega_moe"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
