#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reference — THE BASE PRESET ITSELF: no speculative decoding at all.
#
# This is the team's docker launch command, verbatim, minus the two harness
# deviations (prefix caching off, max-model-len 16384). Notably the preset itself
# specifies NO --speculative-config and NO --max-num-seqs, so neither is set here
# — the sequence cap of 32 exists only because the recipe's spec_decoding feature
# needs the VRAM, and with no drafting that reason is gone. Leaving it unset lets
# vLLM pick its own default, exactly as the preset does.
#
# Role: the denominator for the whole MTP question. Everything in
# run_spec_sweep.sh is measured against this.
#
# BENCH_MODE=nonmtp -> per the InferenceX method: random dataset, RAW prompts (no
# --use-chat-template) and no ShareGPT lane. Those numbers are the non-MTP
# reference and must not be compared against an mtp-mode row.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="${CONFIG:-ref_nonmtp}"
BENCH_MODE="nonmtp"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
)
serve_main
