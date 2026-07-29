#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 12 — DSpark with 3 speculative tokens instead of the recipe default 7.
#
# More draft tokens buy more accepted tokens per verify step at LOW concurrency,
# but each step costs proportionally more compute and VRAM — so the optimum moves
# down as batch size grows. 3 is the single most informative alternative point:
# together with baseline (7) and opt14 (batch-gated) it brackets the trade-off
# without spending a config on every value.
#
# Read this against the acceptance-length numbers in
# results/<config>/*.accept.json, not just throughput — a config can win on
# throughput while accepting fewer tokens, and that tells you something different.
#
# DRAFT_SAMPLE_METHOD=greedy is the other cheap variant worth a manual run.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt12_dspark3"
BENCH_MODE="mtp"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs 32
    --speculative-config "$(dspark_config "${SPEC_TOKENS:-3}")"
)
serve_main
