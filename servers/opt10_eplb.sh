#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 10 — (DP8EP) Expert load balancing.
#
# K3 routes 16 of 896 experts per token: 112 experts per GPU at EP8, roughly 7x
# more than a GLM-5.2-class model. Routing skew therefore has far more room to
# create a straggler rank, and one straggler stalls the entire all2all.
#
# Two mechanisms, both cheap:
#   --enable-eplb            replicates the hottest experts onto redundant slots
#                            (tune with EPLB_REDUNDANT, default 32 = ~3.6% of 896)
#   --expert-placement-strategy round_robin
#                            per the help, "can help improve load balancing for
#                            grouped expert models with no redundant experts"
# They target the same problem from opposite ends, so try EPLB first (as here) and
# then EPLB_REDUNDANT=0 with round_robin alone to see which mechanism carries it.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="${CONFIG:-opt10_eplb}"
BENCH_MODE="mtp"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --data-parallel-size 8
    --enable-expert-parallel
    --enable-ep-weight-filter
    --all2all-backend flashinfer_nvlink_one_sided
    --enable-eplb
    --eplb-config "{\"window_size\":1000,\"step_interval\":3000,\"num_redundant_experts\":${EPLB_REDUNDANT:-32}}"
    --expert-placement-strategy "${EXPERT_PLACEMENT:-linear}"
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
