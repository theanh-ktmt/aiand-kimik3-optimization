#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Baseline (Day 0) — the published recipes.vllm.ai Kimi-K3 config for a single
# 8x B300 node: strategy single_node_tp (TP8) + the Blackwell/NVIDIA hardware
# block + DSpark speculative decoding (7 tokens) with --max-num-seqs 32.
#
# Everything here comes from models/moonshotai/Kimi-K3.yaml, with two documented
# deviations, both applied uniformly to every config in the repo:
#   1. --no-enable-prefix-caching (from common.sh) — the recipe enables prefix
#      caching, but the harness must measure real prefill.
#   2. --load-format is `auto`, not the recipe `fastsafetensors`, which does not
#      exist in this image. See the note in common.sh.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="baseline"
BENCH_MODE="mtp"   # DSpark is on -> client wraps prompts in the chat template,
                   # and the ShareGPT lane runs in addition to random.
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
