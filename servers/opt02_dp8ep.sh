#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 2 — Parallelism: DP8 attention + Expert Parallel. Base for opt11 (all2all),
# opt12 (EPLB) and opt13 (DBO) — only run those if this one beats TP8.
#
# NOTE: the recipe lists multi_node_dep with strategy_min_gpus 16, i.e. DEP is
# officially a >=2-node strategy. Running it on a single 8x B300 node is
# EXPLORATORY — validate on a subset sweep before trusting the numbers.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="${CONFIG:-opt02_dp8ep}"
BENCH_MODE="mtp"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --data-parallel-size 8
    --enable-expert-parallel
    --enable-ep-weight-filter
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
