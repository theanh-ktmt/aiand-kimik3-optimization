#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 10b — CUDA graph: FULL_DECODE_ONLY.
# This is what the recipe's own decode workers use
# (strategy_overrides.pd_cluster.decode.compilation-config). Graphs the decode
# path only, leaving prefill eager — cheaper capture, no graph-vs-chunked-prefill
# interaction to debug.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt10b_cudagraph_decode_only"
BENCH_MODE="spec"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs 128
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --speculative-config "$(dspark_config 7)"
)
serve_main
