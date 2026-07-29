#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# spec.sh — PARAMETERIZED speculative-decoding config. One script, whole sweep.
#
# The number of speculative tokens moves performance more than anything else in
# this campaign, so it gets a real sweep rather than one or two hand-picked
# points. Driven entirely by env:
#
#   SPEC_METHOD   dspark (default) | kimi_k3_mtp | none
#   SPEC_TOKENS   number of speculative tokens (ignored when SPEC_METHOD=none)
#   MAX_NUM_SEQS  concurrent-sequence cap (default 32, the recipe value for
#                 spec decoding; it exists because drafting costs extra VRAM)
#
# Use run_spec_sweep.sh rather than calling this directly — it labels each run
# (CONFIG_LABEL) so every (method, tokens) pair lands in its own results dir and
# its own W&B run, then builds the acceptance-length curve.
#
# Manual single point:
#   SPEC_METHOD=dspark SPEC_TOKENS=5 CONFIG_LABEL=spec_dspark_5 bash run.sh spec subset
#
# WHY BOTH METHODS: `vllm serve --help=all` shows --spec-method accepting both
# `dspark` (the recipe's external Inferact draft model) and `kimi_k3_mtp` (an
# in-model MTP head the recipe never mentions). They have different cost
# profiles — kimi_k3_mtp needs no draft weights and no extra draft VRAM — so the
# sweep covers both, plus `none` as the floor.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"

SPEC_METHOD="${SPEC_METHOD:-dspark}"
SPEC_TOKENS="${SPEC_TOKENS:-3}"

# Default label mirrors run_spec_sweep.sh so a direct call is still self-describing.
if [[ "$SPEC_METHOD" == "none" ]]; then
    CONFIG="${CONFIG:-spec_none}"
    BENCH_MODE="nonmtp"        # no drafting -> InferenceX method says raw prompts
else
    CONFIG="${CONFIG:-spec_${SPEC_METHOD}_${SPEC_TOKENS}}"
    BENCH_MODE="mtp"           # drafting -> chat-wrapped prompts + the ShareGPT lane
fi

k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs "${MAX_NUM_SEQS:-32}"
)
# `none` omits --speculative-config entirely rather than passing an empty value.
if [[ "$SPEC_METHOD" != "none" ]]; then
    _spec_json="$(spec_config "$SPEC_METHOD" "$SPEC_TOKENS")" || exit 1
    SERVE_ARGS+=(--speculative-config "$_spec_json")
fi

serve_main
