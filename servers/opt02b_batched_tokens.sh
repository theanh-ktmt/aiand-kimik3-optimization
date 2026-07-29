#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 2b — Hyperparameters: bigger prefill chunk (--max-num-batched-tokens)
# on top of the lifted sequence cap. 16384 matches the recipe's own TEP prefill
# profile (strategy_overrides.pd_cluster.prefill).
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt02b_batched_tokens"
BENCH_MODE="spec"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-16384}"
    --max-num-seqs 128
    --speculative-config "$(dspark_config 7)"
)
serve_main
