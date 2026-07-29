#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 2a — Hyperparameters: lift --max-num-seqs from 32 to 128.
#
# The recipe caps concurrent sequences at 32 *specifically because DSpark needs
# extra VRAM* ("Speculative decoding needs additional VRAM, so cap concurrent
# sequences"). That cap is the hard ceiling on batch size, so on B300 (288 GB/GPU
# vs the H200 the cap was likely tuned for) it is the first thing to challenge —
# and it is exactly what limits the conc=128 end of the sweep. The recipe's own
# AMD override already uses 128, which suggests headroom exists.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt02a_maxseqs128"
BENCH_MODE="spec"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs 128
    --speculative-config "$(dspark_config 7)"
)
serve_main
