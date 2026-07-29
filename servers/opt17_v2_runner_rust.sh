#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 17 — Model Runner v2 + Rust frontend.
#
# The recipe states plainly that "Model Runner v2 and Rust Frontend fully support
# this model and can be enabled if needed" — validated but off by default, i.e.
# free candidates. They attack different layers (engine step vs HTTP front end),
# so they are bundled here to spend one config instead of three; split them only
# if the bundle regresses and you need to know which half did it.
#
# Expect the gain to show up as lower TTFT and a higher ceiling at conc=128, and
# to be near-zero at conc=1.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt17_v2_runner_rust"
BENCH_MODE="mtp"
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
