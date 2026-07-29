#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 16 — CUDA graph: FULL_DECODE_ONLY (+ the lifted sequence cap).
#
# This is what the recipe own decode workers use
# (strategy_overrides.pd_cluster.decode.compilation-config). It graphs the decode
# path only and leaves prefill eager: cheaper capture, and no
# graph-vs-chunked-prefill interaction to debug. With 93 layers there is a lot of
# kernel-launch overhead to amortise, and DSpark makes decode steps small and
# frequent — the regime where capture pays.
#
# --max-cudagraph-capture-size is pinned to --max-num-seqs because, per
# InferenceX's K3 script, "a 93-layer 2.8T model makes capturing vLLM's full
# 2048-wide ladder prohibitively slow" — without the cap, capture time alone can
# dominate startup.
#
# Valid cudagraph_mode values in this build: NONE, PIECEWISE, FULL,
# FULL_DECODE_ONLY, FULL_AND_PIECEWISE. Try FULL_AND_PIECEWISE via CUDAGRAPH_MODE
# if prefill turns out to be launch-bound too.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt16_cudagraph_decode_only"
BENCH_MODE="mtp"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs "${MAX_NUM_SEQS:-128}"
    --max-cudagraph-capture-size "${MAX_NUM_SEQS:-128}"
    --compilation-config "{\"cudagraph_mode\":\"${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}\"}"
    --speculative-config "$(dspark_config 7)"
)
serve_main
