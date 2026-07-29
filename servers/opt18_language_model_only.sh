#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 18 — Text-only serving: --language-model-only.
#
# No GLM-5.2 analogue: K3 is natively multimodal (MoonViT-V2 vision encoder).
# Our benchmark traffic is 100% text, so the multimodal path is pure overhead.
#
# WHAT IT ACTUALLY DOES, per the help: "disables all multimodal inputs by setting
# all modality limits to 0. Equivalent to setting --limit-mm-per-prompt to 0 for
# every modality." So it removes multimodal *profiling* and the scheduler
# multimodal branches; it does NOT promise that the vision encoder weights are
# skipped. Treat any VRAM saving as something to measure, not assume — compare
# the KV-cache block count that the two server logs report.
#
# CAVEATS: it changes the served capability set, so it is only a legitimate final
# config if the deployment really is text-only; and it makes MMMU-Pro impossible
# (eval/quality_check.sh refuses RUN_MMMU=1 against it). Do not set SMOKE_TEST_MM=1.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt18_language_model_only"
BENCH_MODE="mtp"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --language-model-only
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
