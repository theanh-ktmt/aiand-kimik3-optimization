#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reference — no speculative decoding at all.
#
# The denominator for the whole MTP question: what does the server do with no
# drafting? Also the control for the --max-num-seqs argument, because without
# spec decoding the recipe VRAM reason for capping at 32 disappears — so this
# runs at 128.
#
# BENCH_MODE=nonmtp, which per the InferenceX method means: random dataset, RAW
# prompts (no --use-chat-template), and no ShareGPT lane. Those numbers are the
# non-MTP reference and must not be compared against an mtp-mode row of a
# different dataset lane.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="ref_nonmtp"
BENCH_MODE="nonmtp"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs "${MAX_NUM_SEQS:-128}"
)
serve_main
