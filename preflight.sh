#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# preflight.sh — run this ON THE CLOUD BOX before a long campaign.
#
# Validates the environment WITHOUT launching the 1.4 TB model:
#   * required tools (python3, curl, nvidia-smi, git, vllm)
#   * vLLM version (recipe requires >= 0.27.0) + GPU count + GPU memory
#   * InferenceX submodule present (only needed for DATASET=random)
#   * `vllm bench serve` exists and accepts the ShareGPT flags we pass
#   * ShareGPT dataset present (or fetchable)
#   * EVERY `--flag` used by common.sh + servers/*.sh exists in `vllm serve --help`
#     (catches flags that don't exist in this exact vLLM build — the #1 risk)
#   * enum values we pass to --moe-backend / --all2all-backend / --kv-cache-dtype
#   * attention-backend names referenced by opt03* / the DSpark config
#   * Kimi-K3 parser names (--tool-call-parser / --reasoning-parser kimi_k3)
#   * bash syntax of every script; wandb availability (warn only)
#
# Exit 0 = good to go. Exit 1 = at least one hard failure (see FAIL lines).
# ---------------------------------------------------------------------------
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
# Load .env the same way run.sh does (CR-stripped, so a Windows CRLF .env
# doesn't corrupt values like WANDB_API_KEY).
[[ -f "$REPO_ROOT/.env" ]] && { set -a; source <(tr -d '\r' < "$REPO_ROOT/.env"); set +a; }

FAIL=0
ok()   { echo "  OK   $*"; }
warn() { echo "  WARN $*"; }
bad()  { echo "  FAIL $*"; FAIL=1; }

echo "== Tools =="
for t in python3 curl nvidia-smi git vllm; do
    if command -v "$t" >/dev/null 2>&1; then ok "$t -> $(command -v "$t")"; else
        if [[ "$t" == "git" ]]; then warn "$t not found (only needed for submodule update)"; else bad "$t not found"; fi
    fi
done

echo "== vLLM =="
if command -v vllm >/dev/null 2>&1; then
    ver="$(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo '?')"
    echo "  vLLM version: $ver  (recipe requires min_vllm_version 0.27.0)"
    # Compare against 0.27.0 numerically so 0.28/1.x don't read as "too old".
    if python3 - "$ver" <<'PY' 2>/dev/null
import re, sys
m = re.match(r"(\d+)\.(\d+)\.(\d+)", sys.argv[1])
sys.exit(0 if m and tuple(map(int, m.groups())) >= (0, 27, 0) else 1)
PY
    then ok "version >= 0.27.0"
    else warn "version is below 0.27.0 (or unparseable) — Kimi-K3 support / flag names may be missing"; fi
    echo "  expected image: vllm/vllm-openai:kimi-k3"
fi

echo "== Model =="
MODEL="${MODEL:-moonshotai/Kimi-K3}"
DRAFT_MODEL="${DRAFT_MODEL:-Inferact/Kimi-K3-DSpark}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/workspace/models}"
echo "  MODEL=$MODEL  (client --model / --served-model-name)"
echo "  DRAFT_MODEL=$DRAFT_MODEL  (DSpark; every spec-decoding config needs it)"
if [[ -n "${MODEL_PATH:-}" ]]; then
    echo "  MODEL_PATH=$MODEL_PATH  (the weights the server will actually load)"
    if [[ -d "$MODEL_PATH" && -n "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        ok "MODEL_PATH exists and is non-empty"
        sz="$(du -sb "$MODEL_PATH" 2>/dev/null | cut -f1)"
        if [[ -n "$sz" ]]; then
            gb=$((sz / 1024 / 1024 / 1024))
            echo "  MODEL_PATH size: ${gb} GiB"
            # 2.8T params at MXFP4 (~0.5 byte/param) is ~1.3-1.4 TB.
            [[ "$gb" -lt 1000 ]] && warn "only ${gb} GiB on disk — a full MXFP4 K3 checkpoint is ~1.4 TB; download may be incomplete"
        fi
    else
        bad "MODEL_PATH is set but missing/empty"
    fi
    case "$MODEL_PATH" in
        *[Kk]imi*[Kk]3*|*[Kk]3*) : ;;
        *) warn "MODEL_PATH doesn't look like Kimi-K3 — the harness serves THIS path, so double-check it (unset MODEL_PATH to pull $MODEL into $DOWNLOAD_DIR)" ;;
    esac
else
    echo "  MODEL_PATH unset -> server pulls '$MODEL' into --download-dir $DOWNLOAD_DIR"
    if [[ -d "$DOWNLOAD_DIR" ]]; then
        avail="$(df -B1 --output=avail "$DOWNLOAD_DIR" 2>/dev/null | tail -1)"
        if [[ -n "$avail" ]]; then
            agb=$((avail / 1024 / 1024 / 1024))
            echo "  free space at $DOWNLOAD_DIR: ${agb} GiB"
            [[ "$agb" -lt 1600 ]] && warn "less than 1600 GiB free — a K3 MXFP4 download (~1.4 TB) may not fit"
        fi
    else
        warn "$DOWNLOAD_DIR does not exist yet"
    fi
fi

echo "== GPUs =="
if command -v nvidia-smi >/dev/null 2>&1; then
    n="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
    echo "  visible GPUs: $n"
    if [[ "$n" -eq "${TP:-8}" ]]; then ok "GPU count matches TP=${TP:-8}"; else warn "GPU count ($n) != TP=${TP:-8}"; fi
    # 2.8T MXFP4 needs ~1680 GiB aggregate per the recipe's vram_minimum_gb.
    total_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | paste -sd+ | bc 2>/dev/null)"
    if [[ -n "$total_mib" ]]; then
        total_gb=$((total_mib / 1024))
        echo "  aggregate GPU memory: ${total_gb} GiB (recipe vram_minimum_gb: 1680)"
        if [[ "$total_gb" -ge 1680 ]]; then ok "enough aggregate VRAM for the MXFP4 checkpoint"; else
            bad "only ${total_gb} GiB aggregate VRAM — below the recipe's 1680 GiB minimum for Kimi-K3"; fi
    fi
fi

echo "== Dataset (ShareGPT) =="
SHAREGPT_PATH="${SHAREGPT_PATH:-$REPO_ROOT/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
if [[ -s "$SHAREGPT_PATH" ]]; then
    sz=$(stat -c %s "$SHAREGPT_PATH" 2>/dev/null || echo 0)
    if [[ "$sz" -gt 10000000 ]]; then ok "ShareGPT present ($((sz / 1024 / 1024)) MB) at $SHAREGPT_PATH"; else
        bad "ShareGPT at $SHAREGPT_PATH is only $sz bytes (truncated?) — rerun bench/get_sharegpt.sh"; fi
else
    warn "ShareGPT not downloaded yet at $SHAREGPT_PATH — run: bash bench/get_sharegpt.sh"
fi

echo "== Benchmark client: 'vllm bench serve' (DATASET=sharegpt) =="
if command -v vllm >/dev/null 2>&1; then
    BHELP="$(vllm bench serve --help 2>/dev/null)"
    if [[ -z "$BHELP" ]]; then
        bad "'vllm bench serve --help' produced nothing — the ShareGPT client is unavailable"
    else
        ok "'vllm bench serve --help' works"
        for fl in --dataset-name --dataset-path --sharegpt-output-len --num-prompts \
                  --max-concurrency --request-rate --ignore-eos --backend --endpoint \
                  --base-url --save-result --result-dir --result-filename \
                  --percentile-metrics --metric-percentiles --tokenizer --trust-remote-code --seed; do
            if grep -qF -- "$fl" <<< "$BHELP"; then ok "$fl"; else bad "$fl missing from 'vllm bench serve --help'"; fi
        done
        # sharegpt must be one of the --dataset-name choices.
        if grep -q "sharegpt" <<< "$BHELP"; then ok "--dataset-name supports sharegpt"; else
            bad "'sharegpt' not offered by --dataset-name in this build"; fi
    fi
fi

echo "== InferenceX submodule (only needed for DATASET=random) =="
IX="${INFERENCEX_DIR:-$REPO_ROOT/third_party/InferenceX}"
if [[ -f "$IX/benchmarks/benchmark_lib.sh" ]]; then ok "benchmark_lib.sh present"; else bad "missing $IX (run: git submodule update --init --recursive)"; fi
BENCH_PY="$IX/utils/bench_serving/benchmark_serving.py"
if [[ -f "$BENCH_PY" ]]; then
    ok "benchmark_serving.py present"
    # This vendored revision is random-only (choices=["random"]). Confirm, so the
    # reason bench.sh uses `vllm bench serve` for ShareGPT stays visible.
    if grep -q 'choices=\["random"\]' "$BENCH_PY"; then
        warn "InferenceX client is random-only in this revision — ShareGPT runs use 'vllm bench serve' (by design)"
    fi
else bad "missing $BENCH_PY"; fi

echo "== bash syntax =="
for f in "$REPO_ROOT"/common.sh "$REPO_ROOT"/run.sh "$REPO_ROOT"/run_all.sh \
         "$REPO_ROOT"/run_final.sh "$REPO_ROOT"/bench/bench.sh \
         "$REPO_ROOT"/bench/get_sharegpt.sh "$REPO_ROOT"/eval/quality_check.sh \
         "$REPO_ROOT"/servers/*.sh; do
    bash -n "$f" 2>/dev/null && ok "$(basename "$f")" || bad "syntax: $f"
done

echo "== Serve flags vs 'vllm serve --help=all' =="
# NOTE: plain 'vllm serve --help' only prints config-group pointers; the actual
# flags appear under --help=all. Use that so the check is accurate.
HELP=""
if command -v vllm >/dev/null 2>&1; then
    HELP="$(vllm serve --help=all 2>/dev/null)"
    [[ -z "$HELP" ]] && HELP="$(vllm serve --help 2>/dev/null)"   # fallback
    # Collect serve flags ONLY from inside SERVE_ARGS=(...) / COMMON_SERVE_ARGS=(...)
    # / K3_BASE_ARGS=(...) array blocks. This excludes flags from other commands in
    # the file (e.g. the lm_eval invocation in run_mmlu_pro).
    mapfile -t FLAGS < <(awk '
        /(COMMON_SERVE_ARGS|SERVE_ARGS|K3_BASE_ARGS)=\(/ { inarr=1; next }
        inarr && /^[[:space:]]*\)/                       { inarr=0; next }
        inarr                                            { print }
    ' "$REPO_ROOT/common.sh" "$REPO_ROOT"/servers/*.sh \
        | grep -oE -- '--[a-zA-Z0-9][a-zA-Z0-9-]*' | sort -u)
    for fl in "${FLAGS[@]}"; do
        # --no-* are the negated halves of BooleanOptionalAction flags.
        probe="$fl"; [[ "$fl" == --no-* ]] && probe="--${fl#--no-}"
        if grep -qF -- "$fl" <<< "$HELP" || grep -qF -- "$probe" <<< "$HELP"; then
            ok "$fl"
        else
            bad "$fl not in 'vllm serve --help=all' (this build may not support it)"
        fi
    done
else
    warn "vllm not found; skipped flag check"
fi

echo "== Enum flag values (moe/all2all/kv-cache-dtype) =="
# Verify the specific choice we pass is accepted by THIS build, by matching it
# inside the help's '--flag {a,b,c}' choice list.
check_choice() {  # $1=flag  $2=value
    local line set
    line="$(grep -F -- "$1 {" <<< "$HELP" | head -1)"
    if [[ -z "$line" ]]; then warn "$1: no choice list in help; skipped '$2'"; return; fi
    set="${line#*\{}"; set="${set%%\}*}"
    if grep -qE "(^|,)$2(,|\$)" <<< "$set"; then ok "$1 $2"; else bad "$1 $2 not in {$set}"; fi
}
if [[ -n "$HELP" ]]; then
    # --all2all-backend values appear literally in the servers/*.sh arrays.
    while IFS= read -r val; do
        [[ -n "$val" ]] && check_choice --all2all-backend "$val"
    done < <(grep -rhE '^[[:space:]]*--all2all-backend ' "$REPO_ROOT"/servers/*.sh \
             | awk '{print $2}' | sort -u)
    # --moe-backend / --kv-cache-dtype are set through MOE_BACKEND / KV_CACHE_DTYPE
    # env assignments consumed by k3_base_args, so scrape those instead.
    while IFS= read -r val; do
        [[ -n "$val" ]] && check_choice --moe-backend "$val"
    done < <(grep -rhoE '^MOE_BACKEND="[^"]+"' "$REPO_ROOT"/servers/*.sh \
             | sed -E 's/.*"([^"]+)"/\1/' | sort -u; echo auto)
    while IFS= read -r val; do
        [[ -n "$val" ]] && check_choice --kv-cache-dtype "$val"
    done < <(grep -rhoE '^KV_CACHE_DTYPE="[^"]+"' "$REPO_ROOT"/servers/*.sh \
             | sed -E 's/.*"([^"]+)"/\1/' | sort -u; echo fp8)
else
    warn "no help text; skipped enum value check"
fi

echo "== Kimi-K3 parsers =="
if [[ -n "$HELP" ]]; then
    for p in tool-call-parser reasoning-parser; do
        if grep -qF "kimi_k3" <<< "$HELP"; then ok "--$p kimi_k3 appears in help"; else
            warn "--$p: 'kimi_k3' not visible in help — confirm the parser name in this build"; fi
    done
fi

echo "== Attention backends (opt03* + DSpark verify) =="
if command -v python3 >/dev/null 2>&1; then
    python3 - <<'PY' 2>/dev/null && bk_ok=1 || bk_ok=0
import sys
try:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum as E
    names = {b.name for b in E}
except Exception as e:
    print("registry import failed:", e); sys.exit(3)
want = ["FLASHINFER_MLA", "FLASHMLA"]
missing = [w for w in want if w not in names]
print("available:", ", ".join(sorted(names)))
if missing:
    print("MISSING:", ", ".join(missing)); sys.exit(4)
PY
    if [[ "${bk_ok:-0}" == "1" ]]; then ok "FLASHINFER_MLA / FLASHMLA present"; else
        warn "could not confirm opt03 attention backends — pick names from the 'available:' list above and edit servers/opt03*.sh"; fi
fi

echo "== Optional: W&B =="
if [[ "${WANDB:-1}" == "0" ]]; then
    warn "WANDB=0 — W&B sync disabled; results will NOT be pushed off this box"
else
    if [[ -n "${WANDB_API_KEY:-}" ]]; then ok "WANDB_API_KEY loaded from .env (len ${#WANDB_API_KEY})"; \
    elif [[ -f "$HOME/.netrc" ]]; then ok "no WANDB_API_KEY but ~/.netrc login present"; \
    else warn "WANDB_API_KEY unset and no ~/.netrc — results will NOT sync (put key in .env, or set WANDB=0)"; fi
    echo "  WANDB_PROJECT=${WANDB_PROJECT:-aiand-kimik3-mxfp4}${WANDB_MODE:+  WANDB_MODE=$WANDB_MODE}"
    [[ "${WANDB_MODE:-}" == "offline" ]] && warn "WANDB_MODE=offline — runs stay local in ./wandb, not synced to the cloud"
    python3 -c "import wandb" 2>/dev/null && ok "wandb importable" || warn "wandb not installed (run.sh pip-installs it on demand; needs internet)"
fi

echo
if [[ "$FAIL" -eq 0 ]]; then echo "PREFLIGHT: PASS — good to go."; else echo "PREFLIGHT: FAIL — fix the FAIL lines above before a long run."; fi
exit "$FAIL"
