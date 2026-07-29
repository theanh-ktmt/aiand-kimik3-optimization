#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# quality_check.sh — accuracy gate: baseline vs one or more candidate configs.
#
# Confirms the optimized config(s) did not regress quality versus Day-0. Two
# halves, because Kimi-K3 is NATIVELY MULTIMODAL (this is the difference from the
# GLM-5.2 harness, where MMMU-Pro was N/A):
#   * MMLU-Pro  (text)   — always run; this is the primary gate.
#   * MMMU-Pro  (vision) — opt in with RUN_MMMU=1. lm-eval's API-backed image
#     path is less battle-tested than its text path, so a failure here is
#     reported but does not invalidate the MMLU-Pro verdict.
#
# Runs the eval on each config (launch -> lm-eval -> teardown), then prints one
# comparison table per task (delta + Pass? vs the FIRST config) and writes
# results/quality_check.csv (+ quality_check_mmmu.csv).
#
# Usage:
#   bash eval/quality_check.sh                          # baseline final1 final2
#   bash eval/quality_check.sh baseline final1          # just one candidate
#   RUN_MMMU=1 bash eval/quality_check.sh baseline final1
#   EVAL_CONC=128 bash eval/quality_check.sh            # override eval concurrency
#   SKIP_RUN=1 bash eval/quality_check.sh               # reuse existing eval results
#
# The FIRST config is the baseline reference for the delta/Pass? verdict.
#
# WARNING: do not include opt12_language_model_only when RUN_MMMU=1 — that config
# drops the vision encoder, so every image request fails by construction.
# ---------------------------------------------------------------------------
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Default set: baseline + both proposed configs.
if [[ $# -gt 0 ]]; then CONFIGS=("$@"); else CONFIGS=(baseline final1 final2); fi

PY="$(command -v python3 || command -v python)"
RUN_MMMU="${RUN_MMMU:-0}"

if [[ "$RUN_MMMU" == "1" ]]; then
    for cfg in "${CONFIGS[@]}"; do
        if grep -q -- '--language-model-only' "$REPO_ROOT/servers/$cfg.sh" 2>/dev/null; then
            echo "ERROR: $cfg serves with --language-model-only (no vision encoder);" >&2
            echo "       it cannot be evaluated on MMMU-Pro. Drop RUN_MMMU=1 or the config." >&2
            exit 1
        fi
    done
fi

for cfg in "${CONFIGS[@]}"; do
    script="$REPO_ROOT/servers/$cfg.sh"
    [[ -f "$script" ]] || { echo "ERROR: no such config: $script"; exit 1; }
    if [[ "${SKIP_RUN:-0}" == "1" && -d "$REPO_ROOT/results/$cfg/mmlu_pro" ]]; then
        echo "### SKIP eval (existing results): $cfg ###"
        continue
    fi
    echo "### Accuracy eval: $cfg (mmmu=$RUN_MMMU) ###"
    RUN_EVAL=1 RUN_MMMU="$RUN_MMMU" bash "$script" || echo "WARN: eval for $cfg returned non-zero"
done

echo "### COMPARE MMLU-Pro (baseline = ${CONFIGS[0]}) ###"
dirs=(); for cfg in "${CONFIGS[@]}"; do dirs+=("$REPO_ROOT/results/$cfg/mmlu_pro"); done
"$PY" "$REPO_ROOT/eval/parse_mmlu.py" \
    "${dirs[@]}" \
    --names "${CONFIGS[@]}" \
    --task mmlu_pro \
    --out "$REPO_ROOT/results/quality_check.csv"
mmlu_rc=$?

if [[ "$RUN_MMMU" == "1" ]]; then
    echo "### COMPARE MMMU-Pro (vision; baseline = ${CONFIGS[0]}) ###"
    mdirs=(); for cfg in "${CONFIGS[@]}"; do mdirs+=("$REPO_ROOT/results/$cfg/mmmu_pro"); done
    "$PY" "$REPO_ROOT/eval/parse_mmlu.py" \
        "${mdirs[@]}" \
        --names "${CONFIGS[@]}" \
        --task mmmu_pro \
        --out "$REPO_ROOT/results/quality_check_mmmu.csv" \
        || echo "WARN: MMMU-Pro comparison reported a regression or missing results (see above)"
fi

# The text gate is the one that decides the exit code.
exit "$mmlu_rc"
