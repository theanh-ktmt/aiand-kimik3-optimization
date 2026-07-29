#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 1b — Parallelism: DP8 attention + Expert Parallel (DP8EP). Base for
# opt05 (all2all) / opt06 (EPLB) / opt07 (DBO).
#
# NOTE: the recipe lists multi_node_dep with strategy_min_gpus 16, i.e. DEP is
# officially a >=2-node strategy. Running it on a single 8x B300 node is
# EXPLORATORY — validate on a subset sweep before trusting the numbers.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt01b_dp8ep"
BENCH_MODE="spec"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --data-parallel-size 8
    --enable-expert-parallel
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
