#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 9 — (DP8EP) All2All: flashinfer_nvlink_one_sided.
#
# THE recommended intra-node backend: the recipe says "use --all2all-backend
# flashinfer_nvlink_one_sided for NVLink", and hardcodes it in its own decode
# profile. On a single 8x B300 node every rank is NVLink-connected, so this is
# the one all2all backend worth screening — the alternatives
# (flashinfer_nvlink_two_sided, deepep_v2 for RDMA, deepep_low_latency,
# deepep_high_throughput, allgather_reducescatter) are all available in this
# build and can be tried by editing the value here if this underperforms.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="${CONFIG:-opt09_a2a_nvlink_one_sided}"
BENCH_MODE="mtp"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --data-parallel-size 8
    --enable-expert-parallel
    --enable-ep-weight-filter
    --all2all-backend "${A2A_BACKEND:-flashinfer_nvlink_one_sided}"
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
