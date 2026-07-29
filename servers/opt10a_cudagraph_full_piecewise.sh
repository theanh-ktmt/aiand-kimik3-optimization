#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 10a — CUDA graph: FULL_AND_PIECEWISE with capture sizes tuned to the
# lifted --max-num-seqs. 93 layers means a lot of per-layer launch overhead to
# amortise, and DSpark makes decode steps small and frequent — exactly the
# regime where graph capture pays.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt10a_cudagraph_full_piecewise"
BENCH_MODE="spec"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs 128
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16,24,32,48,64,96,128]}'
    --speculative-config "$(dspark_config 7)"
)
serve_main
