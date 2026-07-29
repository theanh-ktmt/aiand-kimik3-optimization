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
#   2. --max-model-len is 16384, not the preset's 1048576, so KV capacity is not
#      the variable under test.
#
# Everything in the BASE PRESET (Model Runner v2, Rust frontend, tail fusion,
# FlashInfer allreduce, fastsafetensors, moe-backend auto, gpu-mem 0.95,
# kv-cache fp8, TRTLLM_RAGGED prefill) is inherited from common.sh, so it is NOT
# repeated here. What baseline ADDS on top of the preset is the recipe's opt-in
# spec_decoding feature: DSpark(7) plus the --max-num-seqs 32 cap it requires.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="${CONFIG:-baseline}"
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
