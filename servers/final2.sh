#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# final2 — Proposed config #2 (DP8EP base, aimed at the high-concurrency end).
#
# PLACEHOLDER, same rule as final1: replace every [SCREEN] line with the winner
# from results/all.csv before quoting these numbers anywhere.
#
# It starts from flashinfer_nvlink_one_sided + deep_gemm_mega_moe because those
# are the recipe own explicit recommendations for an NVLink expert-parallel
# deployment, so they are a documented starting hypothesis rather than a guess.
#
# NOTE: the recipe treats DEP as a >=16-GPU strategy (strategy_min_gpus
# multi_node_dep: 16); on one node this is exploratory. VALIDATE with
# `bash run.sh final2 subset` before committing to the full run.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="${CONFIG:-final2}"
BENCH_MODE="mtp"
export VLLM_USE_DEEP_GEMM=1
MOE_BACKEND="${MOE_BACKEND:-deep_gemm_mega_moe}"                     # [SCREEN] opt07
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --data-parallel-size 8
    --enable-expert-parallel
    --enable-ep-weight-filter
    --all2all-backend flashinfer_nvlink_one_sided                     # [SCREEN] opt09
    --mamba-backend "${MAMBA_BACKEND:-FLASHINFER}"                    # [SCREEN] opt05
    --no-disable-hybrid-kv-cache-manager                              # [SCREEN] opt08
    --max-num-batched-tokens 16384                                    # [SCREEN] opt03
    --max-num-seqs 128                                                # [SCREEN] opt03
    --speculative-config "$(spec_config "${FINAL2_SPEC_METHOD:-dspark}" "${FINAL2_SPEC:-3}")"  # [SCREEN] run_spec_sweep
)
serve_main
