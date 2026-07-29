#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 5c — (DP8EP) All2All: deepep_v2.
#
# The recipe's RDMA / cross-node choice ("--all2all-backend deepep_v2 for
# RDMA"), paired with UCX_TLS="rc,cuda_copy" so transfers really go over RDMA.
# On a single node it should lose to NVLink — that gap is the number worth
# recording, because it tells you what a future 2-node scale-out costs.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt05c_a2a_deepep_v2"
BENCH_MODE="spec"
export UCX_TLS="${UCX_TLS:-rc,cuda_copy}"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --data-parallel-size 8
    --enable-expert-parallel
    --all2all-backend deepep_v2
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
