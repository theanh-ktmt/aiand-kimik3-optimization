#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 11c — Model Runner v2 + Rust frontend together.
# They attack different layers (engine step vs HTTP front end), so the gains
# should compose; this config checks that they actually do rather than assuming.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt11c_v2_and_rust"
BENCH_MODE="spec"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_RUST_FRONTEND=1
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
