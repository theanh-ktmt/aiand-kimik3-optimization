#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 4 — --performance-mode throughput.
#
# A single high-level switch this build exposes: per the help, "throughput favors
# aggregate tokens/sec at high concurrency (larger CUDA graphs, more aggressive
# batching, throughput-oriented kernels)". It therefore overlaps opt03 (batching)
# and opt16 (CUDA graphs) on purpose — if this one config matches or beats them,
# the tuned configs are not worth the extra maintenance.
#
# Run --performance-mode interactivity too (PERF_MODE=interactivity) if the
# priority ever shifts to TTFT/TPOT at low concurrency.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt04_perf_mode_throughput"
BENCH_MODE="mtp"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --performance-mode "${PERF_MODE:-throughput}"
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
