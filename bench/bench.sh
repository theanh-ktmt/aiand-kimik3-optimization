#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# bench.sh — InferenceX-method serving benchmark sweep for Kimi-K3
#
# Drives a benchmark client against an ALREADY-RUNNING vLLM server, sweeping
# concurrency for each scenario with a saturated server (--request-rate inf).
# One result JSON is written per (scenario, concurrency) cell.
#
# ---------------------------------------------------------------------------
# TWO DATASETS, TWO CLIENTS (this is the one real difference vs the GLM harness)
# ---------------------------------------------------------------------------
# DATASET=sharegpt   (DEFAULT)  ->  client: `vllm bench serve`
#   Real multi-turn chat prompts. This is the right dataset for a model whose
#   headline optimization is speculative decoding: DSpark acceptance length is
#   only representative on natural, chat-formatted text. Synthetic random tokens
#   are unpredictable by construction, so they understate every spec-decode
#   config and make the MTP/DSpark sweep meaningless.
#
#   Why a different client: InferenceX's benchmark_serving.py is random-only in
#   this vendored revision — its --dataset-name is literally choices=["random"]
#   and anything else raises "Unknown dataset". So ShareGPT runs use vLLM's own
#   `vllm bench serve` (the upstream that InferenceX's copy was forked from). It
#   emits the SAME result-JSON schema, so aggregate.py / wandb_sync.py are
#   unchanged, and the InferenceX *method* is preserved exactly: saturated
#   server, --ignore-eos, prefix caching off, warmup, concurrency sweep.
#
# DATASET=random               ->  client: third_party/InferenceX
#   Kept for parity with the GLM-5.2 harness and because ShareGPT prompts are
#   capped at ~1k tokens by vLLM's sampler, so it is the only way to run a
#   prefill-heavy 8k-input scenario. Use it as a cross-check, not the primary.
#
# ---------------------------------------------------------------------------
# SPEC vs NON-SPEC (the ONLY client-side difference)
# ---------------------------------------------------------------------------
#   * spec   -> requests go to /v1/chat/completions so the server applies the
#               Kimi-K3 chat template (DSpark was trained on chat-formatted
#               input; raw prompts silently tank acceptance length).
#   * nospec -> raw /v1/completions.
# Select with: BENCH_MODE=spec (default) | nospec, or --mode spec|nospec.
#
# Usage:
#   bash bench/bench.sh --config baseline --mode spec
#   SWEEP=subset bash bench/bench.sh --config opt04a_moe_deepgemm_mega
#   DATASET=random bash bench/bench.sh --config baseline --sweep full
#
# Env knobs:
#   CONFIG        label used in result filenames (required, or --config)
#   BENCH_MODE    spec | nospec                      (default spec)
#   DATASET       sharegpt | random                  (default sharegpt)
#   SWEEP         full | subset                      (default full)
#                   sharegpt full   -> conc 1..128, OSL 1024 + 256
#                   sharegpt subset -> conc 1/16/128, OSL 1024
#                   random   full   -> conc 1..128, 1k/1k + 8k/1k
#                   random   subset -> conc 1/16/128, 1k/1k
#   CONCS         override concurrency list          (e.g. "1 8 32")
#   SCENARIOS     override scenarios                 (sharegpt: "sharegpt:1024 sharegpt:256"
#                                                     random:   "1024:1024 8192:1024")
#   SHAREGPT_PATH path to ShareGPT_V3_...json        (auto-fetched if missing)
#   WARMUP        1 (default) | 0 — 2x-concurrency throwaway requests per cell
#   RANDOM_RANGE_RATIO                               (default 0.8, random only)
#   RESULT_DIR    where JSONs land                   (default results/<CONFIG>)
# ---------------------------------------------------------------------------
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFERENCEX_DIR="${INFERENCEX_DIR:-$REPO_ROOT/third_party/InferenceX}"
BENCH_PY="$INFERENCEX_DIR/utils/bench_serving/benchmark_serving.py"

MODEL="${MODEL:-moonshotai/Kimi-K3}"
PORT="${PORT:-8888}"
BENCH_MODE="${BENCH_MODE:-spec}"
DATASET="${DATASET:-sharegpt}"
SWEEP="${SWEEP:-full}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"
CONFIG="${CONFIG:-}"
SHAREGPT_PATH="${SHAREGPT_PATH:-$REPO_ROOT/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
# Tokenizer for the client. When serving from a local weights dir ($MODEL_PATH),
# use it directly so tokenization works offline (the HF id may not be cached).
TOKENIZER="${TOKENIZER:-${MODEL_PATH:-$MODEL}}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)  CONFIG="$2"; shift 2 ;;
        --mode)    BENCH_MODE="$2"; shift 2 ;;
        --sweep)   SWEEP="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --port)    PORT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

[[ -z "$CONFIG" ]] && { echo "ERROR: --config (or CONFIG env) is required"; exit 1; }
case "$BENCH_MODE" in spec|nospec) ;; *) echo "ERROR: BENCH_MODE must be spec|nospec"; exit 1 ;; esac
case "$DATASET"    in sharegpt|random) ;; *) echo "ERROR: DATASET must be sharegpt|random"; exit 1 ;; esac
case "$SWEEP"      in full|subset) ;; *) echo "ERROR: SWEEP must be full|subset"; exit 1 ;; esac

# --- Scenario / concurrency matrix -----------------------------------------
if [[ "$DATASET" == "sharegpt" ]]; then
    # Scenario token = "sharegpt:<output-len>". Input length comes from the
    # dataset (vLLM's ShareGPT sampler keeps prompts <=1024 tok and
    # prompt+output <=2048, so OSL above 1024 would filter out nearly everything).
    case "$SWEEP" in
        full)   CONCS="${CONCS:-1 2 4 8 16 32 64 128}"; SCENARIOS="${SCENARIOS:-sharegpt:1024 sharegpt:256}" ;;
        subset) CONCS="${CONCS:-1 16 128}";             SCENARIOS="${SCENARIOS:-sharegpt:1024}" ;;
    esac
else
    case "$SWEEP" in
        full)   CONCS="${CONCS:-1 2 4 8 16 32 64 128}"; SCENARIOS="${SCENARIOS:-1024:1024 8192:1024}" ;;
        subset) CONCS="${CONCS:-1 16 128}";             SCENARIOS="${SCENARIOS:-1024:1024}" ;;
    esac
fi

RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/results/$CONFIG}"
mkdir -p "$RESULT_DIR"

# Tee all benchmark console output to a saved log alongside the result JSONs.
BENCH_LOG="$RESULT_DIR/bench.log"
exec > >(tee -a "$BENCH_LOG") 2>&1
echo "# bench run @ $(date -u +%Y-%m-%dT%H:%M:%SZ)  config=$CONFIG mode=$BENCH_MODE dataset=$DATASET sweep=$SWEEP"

echo "=============================================================="
echo "  Benchmark: config=$CONFIG mode=$BENCH_MODE dataset=$DATASET sweep=$SWEEP"
echo "  scenarios=[$SCENARIOS] concs=[$CONCS]"
echo "  results -> $RESULT_DIR"
echo "=============================================================="

pip install -q datasets pandas 2>/dev/null || true

# ---------------------------------------------------------------------------
# ShareGPT via `vllm bench serve`
# ---------------------------------------------------------------------------
run_sharegpt() {
    command -v vllm >/dev/null 2>&1 || { echo "ERROR: 'vllm' CLI not found (needed for DATASET=sharegpt)"; exit 1; }
    if [[ ! -s "$SHAREGPT_PATH" ]]; then
        echo ">>> ShareGPT dataset missing; fetching..."
        SHAREGPT_PATH="$SHAREGPT_PATH" bash "$REPO_ROOT/bench/get_sharegpt.sh" || exit 1
    fi

    # spec -> chat endpoint (server applies the K3 chat template); nospec -> raw.
    local backend endpoint
    if [[ "$BENCH_MODE" == "spec" ]]; then backend="openai-chat"; endpoint="/v1/chat/completions"
    else                                   backend="openai";      endpoint="/v1/completions"; fi

    local scenario OSL CONC fname
    for scenario in $SCENARIOS; do
        OSL="${scenario##*:}"
        for CONC in $CONCS; do
            fname="${CONFIG}__${BENCH_MODE}_sharegpt_osl${OSL}_conc${CONC}"
            echo ">>> [$CONFIG/$BENCH_MODE] sharegpt OSL=$OSL CONC=$CONC"

            # Warmup: InferenceX runs 2x-concurrency throwaway requests before
            # each cell so the first measured request isn't paying for CUDA-graph
            # / kernel-autotune warmup. `vllm bench serve` has no --num-warmups,
            # so we do it as a separate discarded run.
            if [[ "${WARMUP:-1}" == "1" ]]; then
                echo "    (warmup: $((2 * CONC)) requests)"
                vllm bench serve \
                    --model "$MODEL" --tokenizer "$TOKENIZER" \
                    --backend "$backend" --base-url "http://0.0.0.0:$PORT" --endpoint "$endpoint" \
                    --dataset-name sharegpt --dataset-path "$SHAREGPT_PATH" \
                    --sharegpt-output-len "$OSL" \
                    --num-prompts "$((2 * CONC))" --max-concurrency "$CONC" \
                    --request-rate inf --ignore-eos --trust-remote-code \
                    --seed 12345 --disable-tqdm >/dev/null 2>&1 || true
            fi

            vllm bench serve \
                --model "$MODEL" \
                --tokenizer "$TOKENIZER" \
                --backend "$backend" \
                --base-url "http://0.0.0.0:$PORT" \
                --endpoint "$endpoint" \
                --dataset-name sharegpt \
                --dataset-path "$SHAREGPT_PATH" \
                --sharegpt-output-len "$OSL" \
                --num-prompts "$((CONC * 10))" \
                --max-concurrency "$CONC" \
                --request-rate inf \
                --ignore-eos \
                --trust-remote-code \
                --seed 0 \
                --save-result \
                --percentile-metrics 'ttft,tpot,itl,e2el' \
                --metric-percentiles '90,99' \
                --result-dir "$RESULT_DIR" \
                --result-filename "${fname}.json" \
                || echo "WARN: cell failed (sharegpt OSL=$OSL CONC=$CONC), continuing"
        done
    done
}

# ---------------------------------------------------------------------------
# Random via InferenceX (parity with the GLM-5.2 harness)
# ---------------------------------------------------------------------------
run_random() {
    [[ -f "$BENCH_PY" ]] || { echo "ERROR: $BENCH_PY missing. Run: git submodule update --init"; exit 1; }

    # InferenceX expresses "chat-formatted client input" as --use-chat-template.
    local -a CHAT_TEMPLATE_ARG=()
    [[ "$BENCH_MODE" == "spec" ]] && CHAT_TEMPLATE_ARG=(--use-chat-template)

    local scenario ISL OSL CONC fname
    for scenario in $SCENARIOS; do
        ISL="${scenario%%:*}"
        OSL="${scenario##*:}"
        for CONC in $CONCS; do
            fname="${CONFIG}__${BENCH_MODE}_isl${ISL}_osl${OSL}_conc${CONC}"
            echo ">>> [$CONFIG/$BENCH_MODE] random ISL=$ISL OSL=$OSL CONC=$CONC"
            python3 "$BENCH_PY" \
                --model "$MODEL" \
                --tokenizer "$TOKENIZER" \
                --backend vllm \
                --base-url "http://0.0.0.0:$PORT" \
                --dataset-name random \
                --random-input-len "$ISL" \
                --random-output-len "$OSL" \
                --random-range-ratio "$RANDOM_RANGE_RATIO" \
                --num-prompts "$((CONC * 10))" \
                --max-concurrency "$CONC" \
                --request-rate inf \
                --ignore-eos \
                --num-warmups "$((2 * CONC))" \
                --trust-remote-code \
                --save-result \
                --percentile-metrics 'ttft,tpot,itl,e2el' \
                --metric-percentiles '90,99' \
                --result-dir "$RESULT_DIR" \
                --result-filename "${fname}.json" \
                "${CHAT_TEMPLATE_ARG[@]}" \
                || echo "WARN: cell failed (ISL=$ISL OSL=$OSL CONC=$CONC), continuing"
        done
    done
}

if [[ "$DATASET" == "sharegpt" ]]; then run_sharegpt; else run_random; fi

echo "Done. Aggregate with:  python3 aggregate.py results/$CONFIG --config $CONFIG"
