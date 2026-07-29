#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 3e — Hybrid KV cache manager ON (--no-disable-hybrid-kv-cache-manager).
#
# THIS IS THE MOST K3-SPECIFIC KNOB IN THE WHOLE SWEEP. K3 has 69 KDA layers
# (constant-size recurrent state) and 24 Gated MLA layers (grows with context).
# The hybrid manager allocates those two layer types differently instead of
# sizing every layer for the worst case; on a hybrid-attention model that is a
# large KV-capacity win, which turns into batch size, which turns into
# throughput. The recipe's own P/D profile enables it on BOTH prefill and decode
# workers, so it is well-trodden for this architecture.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt03e_hybrid_kv"
BENCH_MODE="spec"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --no-disable-hybrid-kv-cache-manager
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
