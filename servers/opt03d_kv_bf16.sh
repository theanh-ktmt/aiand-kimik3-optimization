#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 3d — KV cache dtype: fp8 (recipe Blackwell default) -> auto (bf16).
#
# Doubles KV footprint, so fewer sequences fit — but removes the per-token
# quantize/dequantize work and the prefill query quantization it requires. The
# recipe's note is explicit that use_prefill_query_quantization exists *because*
# of FP8 KV, so both are dropped together here.
# Expect this to LOSE on throughput and be a quality reference point.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt03d_kv_bf16"
BENCH_MODE="spec"
KV_CACHE_DTYPE="auto"
ATTENTION_CONFIG='{"mla_prefill_backend":"TRTLLM_RAGGED"}'
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
