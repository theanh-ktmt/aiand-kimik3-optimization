#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 8 — Hybrid KV cache manager ON (--no-disable-hybrid-kv-cache-manager).
#
# K3 mixes two KV shapes: 69 KDA layers hold a CONSTANT-size recurrent state,
# 24 Gated MLA layers hold a cache that GROWS with context. The hybrid manager
# budgets those separately instead of sizing every layer for the worst case — on
# a hybrid-attention model that is a large KV-capacity win, which becomes batch
# size, which becomes throughput.
#
# The recipe own P/D profile enables it on BOTH prefill and decode workers, so it
# is well-trodden for this architecture.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="${CONFIG:-opt08_hybrid_kv}"
BENCH_MODE="mtp"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --no-disable-hybrid-kv-cache-manager
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
