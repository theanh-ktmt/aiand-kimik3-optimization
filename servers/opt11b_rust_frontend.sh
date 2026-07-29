#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 11b — Rust frontend (VLLM_USE_RUST_FRONTEND=1).
# Replaces the Python API server front end. Pure request-path overhead: it should
# show up as lower TTFT and a higher ceiling at conc=128, and do nothing at conc=1.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt11b_rust_frontend"
BENCH_MODE="spec"
export VLLM_USE_RUST_FRONTEND=1
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
