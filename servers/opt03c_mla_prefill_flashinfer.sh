#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 3c — MLA prefill kernel: TRTLLM_RAGGED (recipe default) -> FLASHINFER.
# Isolates the prefill half of --attention-config. TTFT is the metric to watch;
# on a decode-heavy ShareGPT sweep this should be close to neutral, which is
# itself useful information (it means prefill is not the bottleneck).
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt03c_mla_prefill_flashinfer"
BENCH_MODE="spec"
ATTENTION_CONFIG='{"mla_prefill_backend":"FLASHINFER","use_prefill_query_quantization":true}'
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
