#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# final1 — Proposed config #1 (TP8 base).
#
# PLACEHOLDER until the screening campaign reports. It carries the hypotheses
# most likely to survive, so it is runnable today, but every line marked [SCREEN]
# must be replaced with the actual winner from results/all.csv before this is
# presented as a recommendation.
#
#   [SCREEN] KDA/linear kernel        opt06   <- highest expected value (69/93 layers)
#   [SCREEN] hybrid KV manager        opt10
#   [SCREEN] batching hyperparams     opt03 / opt05
#   [SCREEN] spec method + tokens     run_spec_sweep.sh / opt16 / ref_nonmtp
#   [SCREEN] async scheduling         opt14
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="${CONFIG:-final1}"
BENCH_MODE="mtp"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --mamba-backend "${MAMBA_BACKEND:-FLASHINFER}"                   # [SCREEN] opt06
    --no-disable-hybrid-kv-cache-manager                             # [SCREEN] opt10
    --async-scheduling                                               # [SCREEN] opt14
    --max-num-batched-tokens 16384                                   # [SCREEN] opt03
    --max-num-seqs 128                                               # [SCREEN] opt03
    --speculative-config "$(spec_config "${FINAL1_SPEC_METHOD:-dspark}" "${FINAL1_SPEC:-3}")"  # [SCREEN] run_spec_sweep
)
serve_main
