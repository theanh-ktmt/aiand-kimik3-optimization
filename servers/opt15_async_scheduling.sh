#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 15 — --async-scheduling.
#
# Per the help: "Async scheduling helps to avoid gaps in GPU utilization, leading
# to better latency and throughput." Default in this build is None (engine
# decides), so pinning it on is a one-line, zero-risk candidate. It matters most
# exactly where K3 lives: 93 layers with speculative decoding makes each engine
# step short, so scheduler gaps are a larger fraction of the step.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt15_async_scheduling"
BENCH_MODE="mtp"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --async-scheduling
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
