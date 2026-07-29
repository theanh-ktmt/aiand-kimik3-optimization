#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 14 — DSpark(7) for batch 1..64, disabled above 64.
#
# Speculative decoding is a latency optimization that becomes a throughput TAX
# once the GPU is already saturated: at large batch every rejected draft token is
# compute a real request could have used. This schedule keeps it where it pays and
# turns it off where it does not, and should win the conc=128 column.
#
# Expressed as a num_speculative_tokens_per_batch_size list of
# (start, end, num_spec) tuples — there is no standalone disable_by_batch_size flag.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt14_spec_disable_bs64"
BENCH_MODE="mtp"
SPEC_EXTRA_JSON='"num_speculative_tokens_per_batch_size":[[1,64,7],[65,100000,0]]'
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs "${MAX_NUM_SEQS:-128}"
    --speculative-config "$(dspark_config 7)"
)
serve_main
