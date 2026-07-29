#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 1 — Parallelism: TP8 + Expert Parallel.
#
# With 896 routed experts, sharding experts across ranks instead of replicating
# the MoE on every rank is the biggest memory/bandwidth lever inside one node.
#
# --enable-ep-weight-filter is included because it only works with EP and pays
# for itself immediately: per the help, each rank then "only reads its own expert
# shard from disk, which can drastically reduce storage I/O for MoE models with
# per-expert weight tensors". On a ~1.4 TB checkpoint that is the difference
# between a tolerable and an intolerable campaign wall-clock.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt01_tp8ep"
BENCH_MODE="mtp"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --enable-expert-parallel
    --enable-ep-weight-filter
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
