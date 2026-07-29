#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 6 — (DP8EP) EPLB expert load balancing.
#
# K3 routes 16 of 896 experts per token — 112 experts per GPU at EP8. That is
# ~7x more experts than GLM-5.2-class models, so routing skew has far more room
# to create a straggler rank, and a straggler rank stalls the whole all2all.
# EPLB replicates the hottest experts onto extra slots to flatten that.
# Tune with EPLB_REDUNDANT (default 32 = ~3.6% of 896).
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt06_eplb"
BENCH_MODE="spec"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --data-parallel-size 8
    --enable-expert-parallel
    --all2all-backend flashinfer_nvlink_one_sided
    --enable-eplb
    --eplb-config "{\"window_size\":1000,\"step_interval\":3000,\"num_redundant_experts\":${EPLB_REDUNDANT:-32}}"
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
