#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 13 — In-model MTP instead of DSpark:  --spec-method kimi_k3_mtp
#
# THE CONFIG THAT ANSWERS "what is the real effect of MTP".
#
# The recipe only ever mentions DSpark (a separate Inferact-trained draft model),
# but this build advertises BOTH in --spec-method:
#     {..., dspark, ..., kimi_k3_mtp, ...}
# kimi_k3_mtp is an in-model MTP head — the same shape of mechanism GLM-5.2 used,
# with no draft model to load and no draft/rejection sampling knobs.
#
# Against baseline (DSpark 7) and ref_nonmtp (nothing), this gives the full
# three-way comparison:
#     no spec decoding   vs   in-model MTP   vs   external DSpark draft
# and the *.accept.json files make it a comparison of acceptance length, not just
# throughput.
#
# Cost note: this is also the cheapest spec-decode config to operate — no
# Inferact/Kimi-K3-DSpark download, and no extra draft-model VRAM, so if it comes
# close to DSpark it may win on total cost of ownership.
#
# Token count via SPEC_TOKENS (default 3; MTP heads usually peak lower than an
# external draft model).
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt13_mtp_kimik3"
BENCH_MODE="mtp"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs 32
    --speculative-config "$(kimi_k3_mtp_config "${SPEC_TOKENS:-3}")"
)
serve_main
