#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 11a — Model Runner v2 (VLLM_USE_V2_MODEL_RUNNER=1).
# The recipe states plainly that "Model Runner v2 ... fully supports this model
# and can be enabled if needed" — i.e. it is validated but not on by default, so
# it is a free candidate. v2 cuts per-step Python/scheduler overhead, which is
# what dominates when each decode step is tiny (93 layers, DSpark, batch <=32).
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt11a_model_runner_v2"
BENCH_MODE="spec"
export VLLM_USE_V2_MODEL_RUNNER=1
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
