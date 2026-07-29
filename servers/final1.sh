#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# final1 — Proposed config #1 (TP8 base).
#
# PLACEHOLDER until the screening campaign reports. It carries the hypotheses
# most likely to survive, so it is runnable today, but every line marked [SCREEN]
# must be replaced with the actual winner from results/all.csv before this is
# presented as a recommendation.
#
#   [SCREEN] KDA/linear kernel        opt05   <- highest expected value (69/93 layers)
#   [SCREEN] hybrid KV manager        opt08
#   [SCREEN] batching hyperparams     opt03 / opt04
#   [SCREEN] spec method + tokens     opt12 / opt13 / opt14 / ref_nonmtp
#   [SCREEN] async scheduling         opt15
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="final1"
BENCH_MODE="mtp"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"     # [SCREEN] opt17
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --mamba-backend "${MAMBA_BACKEND:-FLASHINFER}"                   # [SCREEN] opt05
    --no-disable-hybrid-kv-cache-manager                             # [SCREEN] opt08
    --async-scheduling                                               # [SCREEN] opt15
    --max-num-batched-tokens 16384                                   # [SCREEN] opt03
    --max-num-seqs 128                                               # [SCREEN] opt03
    --speculative-config "$(dspark_config "${FINAL1_SPEC:-3}")"      # [SCREEN] opt12/13
)
serve_main
