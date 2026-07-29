#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# bench.sh — InferenceX serving benchmark sweep for Kimi-K3
#
# Drives InferenceX's benchmark_serving.py against an ALREADY-RUNNING vLLM
# server, sweeping concurrency for each scenario with a saturated server
# (--request-rate inf). One result JSON per (dataset, scenario, concurrency).
#
# ---------------------------------------------------------------------------
# WHAT GETS RUN, AND WHY
# ---------------------------------------------------------------------------
# The InferenceX method is kept as-is so numbers stay comparable:
#
#   BENCH_MODE=nonmtp -> random dataset, RAW prompts
#   BENCH_MODE=mtp    -> random dataset, prompts WRAPPED in the chat template
#                        (speculative decoding was trained on chat-formatted
#                        input; raw prompts silently tank acceptance length)
#
# MTP configs additionally get a second lane:
#
#   BENCH_MODE=mtp    -> ShareGPT dataset, prompts WRAPPED in the chat template
#
# Rationale for the extra lane: acceptance length on synthetic random tokens is
# an artifact — random input drives degenerate, often highly repetitive output,
# which a draft model predicts unusually well, so random-dataset acceptance can
# be biased in EITHER direction. ShareGPT gives workload-representative
# acceptance, which is what makes the spec-decoding sweep mean something. The
# random lane stays the comparability anchor; the ShareGPT lane is the decision
# input. They are separate rows, never averaged.
#
# Both lanes go through bench/sharegpt_client.py, which is a thin shim that
# reuses InferenceX's OWN parser and main(); with --dataset-name random it
# forwards to upstream unchanged, so the random lane is bit-for-bit an
# InferenceX run. See that file's docstring for the details and caveats.
#
# Usage:
#   bash bench/bench.sh --config baseline --mode mtp
#   SWEEP=subset bash bench/bench.sh --config opt05_linear_flashinfer
#   DATASETS=sharegpt bash bench/bench.sh --config opt13_mtp_kimik3 --mode mtp
#
# Env knobs:
#   CONFIG        label used in result filenames (required, or --config)
#   BENCH_MODE    mtp | nonmtp                      (default mtp)
#   DATASETS      override the lanes                (default: mtp -> "random sharegpt"
#                                                             nonmtp -> "random")
#   SWEEP         full | subset                     (default full)
#                   full   -> conc 1 2 4 8 16 32 64 128
#                             random 1k/1k + 8k/1k ; sharegpt OSL 1024 + 256
#                   subset -> conc 1 16 128
#                             random 1k/1k         ; sharegpt OSL 1024
#   CONCS         override concurrency list         (e.g. "1 8 32")
#   SCENARIOS     override random scenarios "ISL:OSL ..."
#   SHAREGPT_OSLS override sharegpt output lengths  (e.g. "1024 256")
#   SHAREGPT_PATH path to ShareGPT_V3_...json       (auto-fetched if missing)
#   SHAREGPT_MAX_INPUT_LEN  drop longer prompts     (default 4096)
#   TOKENIZER_MODE  passed to the client if set     (e.g. kimi_k3 for XTML rendering)
#   RANDOM_RANGE_RATIO                              (default 0.8)
#   ACCEPT_METRICS  1 (default) | 0 — scrape /metrics spec-decode counters per cell
#   RESULT_DIR    where JSONs land                  (default results/<CONFIG>)
# ---------------------------------------------------------------------------
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFERENCEX_DIR="${INFERENCEX_DIR:-$REPO_ROOT/third_party/InferenceX}"
export INFERENCEX_DIR
CLIENT="$REPO_ROOT/bench/sharegpt_client.py"
BENCH_PY="$INFERENCEX_DIR/utils/bench_serving/benchmark_serving.py"

MODEL="${MODEL:-moonshotai/Kimi-K3}"
PORT="${PORT:-8888}"
BENCH_MODE="${BENCH_MODE:-mtp}"
SWEEP="${SWEEP:-full}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"
CONFIG="${CONFIG:-}"
SHAREGPT_PATH="${SHAREGPT_PATH:-$REPO_ROOT/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
SHAREGPT_MAX_INPUT_LEN="${SHAREGPT_MAX_INPUT_LEN:-4096}"
# Tokenizer for the client. When serving from a local weights dir ($MODEL_PATH),
# use it directly so tokenization works offline (the HF id may not be cached).
TOKENIZER="${TOKENIZER:-${MODEL_PATH:-$MODEL}}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)   CONFIG="$2"; shift 2 ;;
        --mode)     BENCH_MODE="$2"; shift 2 ;;
        --sweep)    SWEEP="$2"; shift 2 ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        --port)     PORT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

[[ -z "$CONFIG" ]] && { echo "ERROR: --config (or CONFIG env) is required"; exit 1; }
case "$BENCH_MODE" in mtp|nonmtp) ;; *) echo "ERROR: BENCH_MODE must be mtp|nonmtp"; exit 1 ;; esac
case "$SWEEP"      in full|subset) ;; *) echo "ERROR: SWEEP must be full|subset"; exit 1 ;; esac
[[ -f "$CLIENT"   ]] || { echo "ERROR: $CLIENT missing"; exit 1; }
[[ -f "$BENCH_PY" ]] || { echo "ERROR: $BENCH_PY missing. Run: git submodule update --init"; exit 1; }

# --- Lanes ------------------------------------------------------------------
# ShareGPT is the extra MTP-only lane; non-MTP stays pure InferenceX.
if [[ -z "${DATASETS:-}" ]]; then
    if [[ "$BENCH_MODE" == "mtp" ]]; then DATASETS="random sharegpt"; else DATASETS="random"; fi
fi

# --- Scenario / concurrency matrix -----------------------------------------
case "$SWEEP" in
    full)   CONCS="${CONCS:-1 2 4 8 16 32 64 128}"
            SCENARIOS="${SCENARIOS:-1024:1024 8192:1024}"
            SHAREGPT_OSLS="${SHAREGPT_OSLS:-1024 256}" ;;
    subset) CONCS="${CONCS:-1 16 128}"
            SCENARIOS="${SCENARIOS:-1024:1024}"
            SHAREGPT_OSLS="${SHAREGPT_OSLS:-1024}" ;;
esac

RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/results/$CONFIG}"
mkdir -p "$RESULT_DIR"

# Tee all benchmark console output to a saved log alongside the result JSONs.
BENCH_LOG="$RESULT_DIR/bench.log"
exec > >(tee -a "$BENCH_LOG") 2>&1
echo "# bench run @ $(date -u +%Y-%m-%dT%H:%M:%SZ)  config=$CONFIG mode=$BENCH_MODE datasets=[$DATASETS] sweep=$SWEEP"

echo "=============================================================="
echo "  Benchmark: config=$CONFIG mode=$BENCH_MODE sweep=$SWEEP"
echo "  lanes=[$DATASETS]  concs=[$CONCS]"
echo "  random scenarios=[$SCENARIOS]  sharegpt OSLs=[$SHAREGPT_OSLS]"
echo "  results -> $RESULT_DIR"
echo "=============================================================="

pip install -q datasets pandas 2>/dev/null || true

# --- MTP toggle: the single client-side difference (InferenceX semantics) ---
CHAT_TEMPLATE_ARG=()
[[ "$BENCH_MODE" == "mtp" ]] && CHAT_TEMPLATE_ARG=(--use-chat-template)

# Optional tokenizer mode (e.g. kimi_k3 -> XTML chat rendering instead of Jinja).
TOKENIZER_MODE_ARG=()
[[ -n "${TOKENIZER_MODE:-}" ]] && TOKENIZER_MODE_ARG=(--tokenizer-mode "$TOKENIZER_MODE")

# --- Speculative-decoding acceptance instrumentation ------------------------
# Throughput alone cannot tell you WHY an MTP config won or lost. vLLM exports
# spec-decode counters on /metrics; diffing them across a cell gives the mean
# acceptance length for exactly that cell. This is read-only and changes nothing
# about the measurement — it just records what the engine already counted.
_scrape_spec() {   # $1 = output file
    [[ "${ACCEPT_METRICS:-1}" == "1" ]] || return 0
    curl -sS -m 20 "http://0.0.0.0:$PORT/metrics" 2>/dev/null \
        | grep -E '^vllm:spec_decode[^ ]* ' > "$1" 2>/dev/null || true
}

_accept_report() {  # $1 = before file  $2 = after file  $3 = json out
    [[ "${ACCEPT_METRICS:-1}" == "1" ]] || return 0
    [[ -s "$2" ]] || { rm -f "$1" "$2"; return 0; }
    BEFORE="$1" AFTER="$2" OUT="$3" python3 - <<'PY' 2>/dev/null || true
import json, os, re

def load(p):
    d = {}
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                m = re.match(r"(\S+?)(\{.*\})?\s+([0-9.eE+-]+)$", line.strip())
                if m:
                    # Sum across label sets (one engine per DP rank).
                    d[m.group(1)] = d.get(m.group(1), 0.0) + float(m.group(3))
    except FileNotFoundError:
        pass
    return d

before, after = load(os.environ["BEFORE"]), load(os.environ["AFTER"])
delta = {k: after[k] - before.get(k, 0.0) for k in after}
drafts = delta.get("vllm:spec_decode_num_drafts_total")
draft_tok = delta.get("vllm:spec_decode_num_draft_tokens_total")
accepted = delta.get("vllm:spec_decode_num_accepted_tokens_total")
out = {"counters_delta": delta}
if drafts:
    # Mean accepted tokens per draft step, +1 for the always-correct bonus token.
    out["mean_acceptance_length"] = round(1.0 + (accepted or 0.0) / drafts, 4)
if draft_tok:
    out["draft_acceptance_rate"] = round((accepted or 0.0) / draft_tok, 4)
with open(os.environ["OUT"], "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
if "mean_acceptance_length" in out:
    print(f"    acceptance length={out['mean_acceptance_length']} "
          f"draft_accept_rate={out.get('draft_acceptance_rate')}")
PY
    rm -f "$1" "$2"
}

# run_cell <result-basename> <extra client args...>
run_cell() {
    local fname="$1"; shift
    local before="$RESULT_DIR/.spec_before" after="$RESULT_DIR/.spec_after"
    _scrape_spec "$before"
    python3 "$CLIENT" \
        --model "$MODEL" \
        --tokenizer "$TOKENIZER" \
        --backend vllm \
        --base-url "http://0.0.0.0:$PORT" \
        --request-rate inf \
        --ignore-eos \
        --trust-remote-code \
        --save-result \
        --percentile-metrics 'ttft,tpot,itl,e2el' \
        --metric-percentiles '90,99' \
        --result-dir "$RESULT_DIR" \
        --result-filename "${fname}.json" \
        "${TOKENIZER_MODE_ARG[@]}" \
        "${CHAT_TEMPLATE_ARG[@]}" \
        "$@" \
        || echo "WARN: cell failed ($fname), continuing"
    _scrape_spec "$after"
    _accept_report "$before" "$after" "$RESULT_DIR/${fname}.accept.json"
}

# --- Lane 1: random (InferenceX, unchanged) --------------------------------
run_random() {
    local scenario ISL OSL CONC
    for scenario in $SCENARIOS; do
        ISL="${scenario%%:*}"; OSL="${scenario##*:}"
        for CONC in $CONCS; do
            echo ">>> [$CONFIG/$BENCH_MODE] random ISL=$ISL OSL=$OSL CONC=$CONC"
            run_cell "${CONFIG}__${BENCH_MODE}_isl${ISL}_osl${OSL}_conc${CONC}" \
                --dataset-name random \
                --random-input-len "$ISL" \
                --random-output-len "$OSL" \
                --random-range-ratio "$RANDOM_RANGE_RATIO" \
                --num-prompts "$((CONC * 10))" \
                --max-concurrency "$CONC" \
                --num-warmups "$((2 * CONC))"
        done
    done
}

# --- Lane 2: ShareGPT (MTP only) -------------------------------------------
run_sharegpt() {
    if [[ ! -s "$SHAREGPT_PATH" ]]; then
        echo ">>> ShareGPT dataset missing; fetching..."
        SHAREGPT_PATH="$SHAREGPT_PATH" bash "$REPO_ROOT/bench/get_sharegpt.sh" || return 1
    fi
    local OSL CONC
    for OSL in $SHAREGPT_OSLS; do
        for CONC in $CONCS; do
            echo ">>> [$CONFIG/$BENCH_MODE] sharegpt OSL=$OSL CONC=$CONC (ISL from dataset)"
            run_cell "${CONFIG}__${BENCH_MODE}_sharegpt_osl${OSL}_conc${CONC}" \
                --dataset-name sharegpt \
                --dataset-path "$SHAREGPT_PATH" \
                --sharegpt-output-len "$OSL" \
                --sharegpt-max-input-len "$SHAREGPT_MAX_INPUT_LEN" \
                --random-range-ratio "$RANDOM_RANGE_RATIO" \
                --num-prompts "$((CONC * 10))" \
                --max-concurrency "$CONC" \
                --num-warmups "$((2 * CONC))"
        done
    done
}

for ds in $DATASETS; do
    case "$ds" in
        random)   run_random ;;
        sharegpt) run_sharegpt ;;
        *) echo "ERROR: unknown dataset lane '$ds' (want random|sharegpt)"; exit 1 ;;
    esac
done

echo "Done. Aggregate with:  python3 aggregate.py results/$CONFIG --config $CONFIG"
