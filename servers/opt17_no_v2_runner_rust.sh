#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 17 — Model Runner v2 + Rust frontend OFF.
#
# THE DIRECTION MATTERS: both are already in the BASE PRESET
# (VLLM_USE_V2_MODEL_RUNNER=1, VLLM_USE_RUST_FRONTEND=1 in the team's docker
# launch), so "turn them on" would be a no-op. The informative experiment is to
# turn them OFF and see what the preset is buying.
#
# Two outcomes, both useful:
#   * baseline wins  -> the preset is earning its keep; quantified, not assumed.
#   * this wins      -> one of them is a regression on K3 and the preset should
#                       drop it. Split into two runs
#                       (VLLM_USE_V2_MODEL_RUNNER=0 alone, then
#                        VLLM_USE_RUST_FRONTEND=0 alone) to find which.
#
# Note the Rust frontend is also why the base env sets
# VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900 — with it off, that no longer matters.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="${CONFIG:-opt17_no_v2_runner_rust}"
BENCH_MODE="mtp"
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_RUST_FRONTEND=0
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
