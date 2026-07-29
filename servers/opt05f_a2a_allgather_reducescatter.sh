#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 5f — (DP8EP) All2All: allgather_reducescatter.
# The zero-extra-setup fallback (no DeepEP, no FlashInfer comms). Keep it in the
# sweep as the safety net: if every specialised backend fails to initialise in
# this image, this is the one that still runs, and you need its number to know
# what you are giving up.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt05f_a2a_allgather_reducescatter"
BENCH_MODE="spec"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --data-parallel-size 8
    --enable-expert-parallel
    --all2all-backend allgather_reducescatter
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
