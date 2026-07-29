#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 8 — DSpark speculative decoding: num_speculative_tokens=5
#         (recipe default is 7).
#
# This is THE knob for Kimi-K3. Note it is DSpark, not MTP/EAGLE: an
# Inferact-trained draft model ($DRAFT_MODEL) with probabilistic draft sampling
# and block rejection sampling. More draft tokens = more accepted tokens per
# verify step at low concurrency, but each step costs proportionally more compute
# and VRAM, so the optimum moves down as batch size grows.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt08_dspark5"
BENCH_MODE="spec"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs 32
    --speculative-config "$(dspark_config 5)"
)
serve_main
