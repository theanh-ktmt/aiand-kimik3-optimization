#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 12 — Text-only serving: --language-model-only.
#
# UNIQUE TO K3 vs GLM-5.2: K3 is natively multimodal and loads a MoonViT-V2
# vision encoder (401M params) plus its multimodal preprocessing pipeline. Our
# benchmark traffic (ShareGPT) is 100% text, so the encoder is pure overhead —
# VRAM, a CPU-side image pipeline, and multimodal branches in the scheduler.
# The recipe exposes this as a first-class opt-in feature for exactly this case.
#
# CAVEAT: this changes the served capability set, so it is only a legitimate
# final config if the deployment really is text-only. It also makes MMMU-Pro
# impossible — never run eval/quality_check.sh with RUN_MMMU=1 against this, and
# do not set SMOKE_TEST_MM=1 here.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt12_language_model_only"
BENCH_MODE="spec"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --language-model-only
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
