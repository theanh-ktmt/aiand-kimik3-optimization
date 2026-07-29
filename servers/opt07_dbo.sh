#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 7 — (DP8EP) DBO: dual-batch overlap (compute/communication overlap).
# Splits each step into two micro-batches so the MoE all2all of one overlaps the
# GEMMs of the other. The bigger the all2all, the more there is to hide — and
# with 896 experts there is a lot.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt07_dbo"
BENCH_MODE="spec"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --data-parallel-size 8
    --enable-expert-parallel
    --all2all-backend flashinfer_nvlink_one_sided
    --enable-dbo
    --dbo-decode-token-threshold "${DBO_DECODE_THR:-32}"
    --dbo-prefill-token-threshold "${DBO_PREFILL_THR:-512}"
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
