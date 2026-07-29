#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 2c — Hyperparameters: --gpu-memory-utilization 0.95 -> 0.97.
# 2 extra points of 288 GB is ~5.8 GB/GPU of additional KV cache, which is what
# lets a bigger --max-num-seqs actually hold its batch. Watch for OOM at the
# tail of the sweep; back off to 0.96 if the server dies under conc=128.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt02c_gpumem097"
BENCH_MODE="spec"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.97}"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs 128
    --speculative-config "$(dspark_config 7)"
)
serve_main
