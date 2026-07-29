#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 19 — MLA prefill kernel: TRTLLM_RAGGED -> FLASHINFER.
#
# This one exists because the two sources DISAGREE, and both are credible:
#   * the recipe yaml (hardware_overrides.blackwell) says
#       mla_prefill_backend: TRTLLM_RAGGED     <- what baseline uses
#   * BOTH of InferenceX's own Kimi-K3 B300 scripts say
#       mla_prefill_backend: FLASHINFER        <- what this config uses
#     (third_party/InferenceX/benchmarks/single_node/{speedbench,agentic}/kimik3_fp4_b300_vllm.sh,
#      commented "MLA prefill runs on FlashInfer per the production recipe")
#
# So one of the two is stale, and a single subset sweep settles it. Watch TTFT
# rather than output throughput — this only affects prefill.
#
# use_prefill_query_quantization stays true: it is required whenever the KV cache
# is FP8, in both sources.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="${CONFIG:-opt19_mla_prefill_flashinfer}"
BENCH_MODE="mtp"
ATTENTION_CONFIG='{"mla_prefill_backend":"FLASHINFER","use_prefill_query_quantization":true}'
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
