#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# final1 — Proposed config #1 (TP8 base).
#
# PLACEHOLDER until the screening campaign reports. It currently carries the
# hypotheses most likely to survive, so it is runnable today, but every line
# marked [SCREEN] must be replaced with the actual winner from results/all.csv
# before this is presented as a recommendation.
#
#   [SCREEN] --max-num-seqs / batched tokens   from opt02a / opt02b / opt02c
#   [SCREEN] hybrid KV cache manager           from opt03e
#   [SCREEN] MoE backend                       from opt04*
#   [SCREEN] DSpark token count                from opt08*
#   [SCREEN] Model Runner v2 / Rust frontend   from opt11*
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="final1"
BENCH_MODE="spec"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"    # [SCREEN] opt11a
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --no-disable-hybrid-kv-cache-manager                            # [SCREEN] opt03e
    --max-num-batched-tokens 16384                                  # [SCREEN] opt02b
    --max-num-seqs 128                                              # [SCREEN] opt02a
    --speculative-config "$(dspark_config "${FINAL1_SPEC:-3}")"     # [SCREEN] opt08*
)
serve_main
