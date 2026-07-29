#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 8 — DSpark(7) for batch 1..64, disabled above 64.
#
# Speculative decoding is a latency optimization that becomes a throughput TAX
# once the GPU is already saturated: at large batch every rejected draft token is
# wasted compute that a plain decode step would have spent on a real request.
# The schedule below keeps DSpark where it pays and turns it off where it does
# not, expressed as a num_speculative_tokens_per_batch_size list of
# (start, end, num_spec) tuples — there is no standalone disable_by_batch_size
# flag. This is what should win the conc=128 column.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt08_dspark_disable_bs64"
BENCH_MODE="spec"
SPEC_EXTRA_JSON='"num_speculative_tokens_per_batch_size":[[1,64,7],[65,100000,0]]'
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
