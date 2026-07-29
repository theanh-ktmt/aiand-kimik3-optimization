#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Baseline (Day 0) — the published recipes.vllm.ai Kimi-K3 config for a single
# 8x B300 node: strategy single_node_tp (TP8) + the Blackwell/NVIDIA hardware
# block + DSpark speculative decoding (7 tokens) with --max-num-seqs 32.
#
# Everything here comes from models/moonshotai/Kimi-K3.yaml — nothing invented.
# The ONE deliberate deviation is --no-enable-prefix-caching (from common.sh):
# the recipe enables prefix caching, but the harness must measure real prefill.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="baseline"
BENCH_MODE="spec"   # DSpark is on, so the client must send chat-formatted input
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
