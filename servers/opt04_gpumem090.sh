#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 22 — --gpu-memory-utilization 0.95 -> 0.90 (a DIFFERENT VALUE for a
# base-preset item, not a new flag).
#
# The preset uses 0.95, but BOTH of InferenceX's own Kimi-K3 B300 scripts run
# 0.90. On a ~1.4 TB checkpoint that is ~14 GB/GPU of KV cache given up, so it
# should lose on throughput — unless 0.95 is close enough to the edge that
# allocator pressure or fragmentation costs more than the extra cache buys.
#
# Run this together with opt03 (--max-num-seqs 512): if the big sequence cap OOMs
# at 0.95, this is the value to pair it with rather than abandoning the cap.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="${CONFIG:-opt04_gpumem090}"
BENCH_MODE="mtp"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs "${MAX_NUM_SEQS:-32}"
    --speculative-config "$(dspark_config 7)"
)
serve_main
