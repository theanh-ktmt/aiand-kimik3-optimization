#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reference — DSpark disabled (no speculative decoding).
#
# Quantifies what speculative decoding actually contributes, and doubles as the
# control for the --max-num-seqs question: without DSpark the recipe's VRAM
# reason for capping at 32 disappears, so this runs at 128.
# BENCH_MODE=nospec -> the client hits /v1/completions (no chat template).
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="ref_nospec"
BENCH_MODE="nospec"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs 128
)
serve_main
