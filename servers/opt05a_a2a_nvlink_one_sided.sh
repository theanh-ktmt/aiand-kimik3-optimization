#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 5a — (DP8EP) All2All: flashinfer_nvlink_one_sided.
#
# THE recommended intra-node backend: the recipe says "use --all2all-backend
# flashinfer_nvlink_one_sided for NVLink", and its own decode profile hardcodes
# it. On a single 8x B300 node every rank is NVLink-connected, so this is the
# expected winner of the opt05 group — the others exist to prove it.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt05a_a2a_nvlink_one_sided"
BENCH_MODE="spec"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --data-parallel-size 8
    --enable-expert-parallel
    --all2all-backend flashinfer_nvlink_one_sided
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
