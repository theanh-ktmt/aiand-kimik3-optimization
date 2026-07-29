#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# preflight.sh — run this ON THE CLOUD BOX before a long campaign.
#
# Validates the environment WITHOUT launching the 1.4 TB model:
#   * required tools (python3, curl, nvidia-smi, git, vllm)
#   * vLLM version (recipe requires >= 0.27.0) + GPU count + aggregate VRAM
#     against the recipe vram_minimum_gb, + disk space for ~1.4 TB
#   * EVERY --flag used by common.sh + servers/*.sh is checked with
#     `vllm serve --help=<flag>` — a definitive per-flag answer, unlike grepping
#     a substring out of `--help=all` (which false-positives on prose mentions)
#   * the exact enum VALUES we pass (--moe-backend / --all2all-backend /
#     --kv-cache-dtype / --load-format / --mamba-backend / --attention-backend /
#     --performance-mode / --expert-placement-strategy / cudagraph_mode /
#     --spec-method)
#   * InferenceX submodule + the ShareGPT shim import and parser rebuild
#   * ShareGPT dataset present (or fetchable)
#   * Kimi-K3 parser + tokenizer-mode names
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

HAVE_VLLM=0
command -v vllm >/dev/null 2>&1 && HAVE_VLLM=1

echo "== vLLM =="
if [[ "$HAVE_VLLM" == "1" ]]; then
    ver="$(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo '?')"
    echo "  vLLM version: $ver  (recipe requires min_vllm_version 0.27.0)"
    if python3 - "$ver" <<'PY' 2>/dev/null
import re, sys
m = re.match(r"(\d+)\.(\d+)\.(\d+)", sys.argv[1])
sys.exit(0 if m and tuple(map(int, m.groups())) >= (0, 27, 0) else 1)
PY
    then ok "version >= 0.27.0"
    else warn "version below 0.27.0 (or unparseable) — Kimi-K3 support / flag names may be missing"; fi
    echo "  expected image: vllm/vllm-openai:kimi-k3"
fi

echo "== Model =="
MODEL="${MODEL:-moonshotai/Kimi-K3}"
DRAFT_MODEL="${DRAFT_MODEL:-Inferact/Kimi-K3-DSpark}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/workspace/models}"
echo "  MODEL=$MODEL"
echo "  DRAFT_MODEL=$DRAFT_MODEL  (DSpark; needed by every config except ref_nonmtp"
echo "                             and the spec_kimi_k3_mtp_* / spec_none sweep points)"
if [[ -n "${MODEL_PATH:-}" ]]; then
    echo "  MODEL_PATH=$MODEL_PATH  (the weights the server will actually load)"
    if [[ -d "$MODEL_PATH" && -n "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        ok "MODEL_PATH exists and is non-empty"
        sz="$(du -sb "$MODEL_PATH" 2>/dev/null | cut -f1)"
        if [[ -n "$sz" ]]; then
            gb=$((sz / 1024 / 1024 / 1024))
            echo "  MODEL_PATH size: ${gb} GiB"
            [[ "$gb" -lt 1000 ]] && warn "only ${gb} GiB on disk — a full MXFP4 K3 checkpoint is ~1.4 TB; download may be incomplete"
        fi
    else
        bad "MODEL_PATH is set but missing/empty"
    fi
    case "$MODEL_PATH" in
        *[Kk]imi*|*[Kk]3*) : ;;
        *) warn "MODEL_PATH doesn't look like Kimi-K3 — the harness serves THIS path, so double-check it" ;;
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
    total_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | paste -sd+ | bc 2>/dev/null)"
    if [[ -n "$total_mib" ]]; then
        total_gb=$((total_mib / 1024))
        echo "  aggregate GPU memory: ${total_gb} GiB (recipe vram_minimum_gb: 1680)"
        if [[ "$total_gb" -ge 1680 ]]; then ok "enough aggregate VRAM for the MXFP4 checkpoint"; else
            bad "only ${total_gb} GiB aggregate VRAM — below the recipe 1680 GiB minimum for Kimi-K3"; fi
    fi
fi

echo "== Dataset (ShareGPT — the extra MTP lane) =="
SHAREGPT_PATH="${SHAREGPT_PATH:-$REPO_ROOT/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
if [[ -s "$SHAREGPT_PATH" ]]; then
    sz=$(stat -c %s "$SHAREGPT_PATH" 2>/dev/null || echo 0)
    if [[ "$sz" -gt 10000000 ]]; then ok "ShareGPT present ($((sz / 1024 / 1024)) MB) at $SHAREGPT_PATH"; else
        bad "ShareGPT at $SHAREGPT_PATH is only $sz bytes (truncated?) — rerun bench/get_sharegpt.sh"; fi
else
    warn "ShareGPT not downloaded yet at $SHAREGPT_PATH — run: bash bench/get_sharegpt.sh"
fi

echo "== InferenceX submodule + ShareGPT shim =="
IX="${INFERENCEX_DIR:-$REPO_ROOT/third_party/InferenceX}"
export INFERENCEX_DIR="$IX"
if [[ -f "$IX/benchmarks/benchmark_lib.sh" ]]; then ok "benchmark_lib.sh present"; else bad "missing $IX (run: git submodule update --init --recursive)"; fi
BENCH_PY="$IX/utils/bench_serving/benchmark_serving.py"
if [[ -f "$BENCH_PY" ]]; then
    ok "benchmark_serving.py present"
else
    bad "missing $BENCH_PY"
fi
# The shim rebuilds InferenceX's own parser by exec'ing its __main__ block. That
# is the one brittle coupling in this repo, so verify it here rather than 40
# minutes into a run.
if [[ -f "$REPO_ROOT/bench/sharegpt_client.py" && -f "$BENCH_PY" ]]; then
    if REPO_ROOT="$REPO_ROOT" INFERENCEX_DIR="$IX" python3 - >/dev/null 2>&1 <<'PY'
import argparse, importlib.util, os, sys, types
from pathlib import Path
root = Path(os.environ["REPO_ROOT"])
spec = importlib.util.spec_from_file_location("sgc", root / "bench" / "sharegpt_client.py")
sgc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sgc)
ix = Path(os.environ["INFERENCEX_DIR"]) / "utils" / "bench_serving" / "benchmark_serving.py"
stub = types.ModuleType("bs_stub"); stub.__file__ = str(ix)
stub.FlexibleArgumentParser = argparse.ArgumentParser
stub.ASYNC_REQUEST_FUNCS = {"vllm": 1, "openai": 2}
p = sgc._widen_parser(sgc._build_upstream_parser(stub))
ns = p.parse_args(["--model", "m", "--dataset-name", "sharegpt", "--dataset-path", "x",
                   "--sharegpt-output-len", "1024"])
assert ns.dataset_name == "sharegpt" and ns.sharegpt_max_input_len
PY
    then ok "sharegpt_client.py can rebuild InferenceX's parser"
    else bad "sharegpt_client.py could NOT rebuild InferenceX's parser — upstream layout changed; fix the shim before running the ShareGPT lane"; fi
fi

echo "== bash syntax =="
for f in "$REPO_ROOT"/common.sh "$REPO_ROOT"/run.sh "$REPO_ROOT"/run_all.sh \
         "$REPO_ROOT"/run_final.sh "$REPO_ROOT"/run_spec_sweep.sh \
         "$REPO_ROOT"/bench/bench.sh \
         "$REPO_ROOT"/bench/get_sharegpt.sh "$REPO_ROOT"/eval/quality_check.sh \
         "$REPO_ROOT"/servers/*.sh; do
    bash -n "$f" 2>/dev/null && ok "$(basename "$f")" || bad "syntax: $f"
done
echo "== python syntax =="
for f in "$REPO_ROOT"/aggregate.py "$REPO_ROOT"/wandb_sync.py \
         "$REPO_ROOT"/eval/parse_mmlu.py "$REPO_ROOT"/bench/sharegpt_client.py; do
    python3 -m py_compile "$f" 2>/dev/null && ok "$(basename "$f")" || bad "syntax: $f"
done

# ---------------------------------------------------------------------------
# Flag verification. `vllm serve --help=<flag>` answers for ONE flag, so it does
# not false-positive the way grepping --help=all does. We fall back to the big
# blob only if the per-flag form is unavailable.
# ---------------------------------------------------------------------------
HELP_ALL=""
if [[ "$HAVE_VLLM" == "1" ]]; then
    HELP_ALL="$(vllm serve --help=all 2>/dev/null)"
fi

flag_exists() {   # $1 = --flag
    local fl="$1" probe="$1" out
    [[ "$fl" == --no-* ]] && probe="--${fl#--no-}"
    out="$(vllm serve "--help=${probe#--}" 2>&1)"
    # A known flag prints its own help; an unknown one says so / prints nothing useful.
    if grep -qF -- "$probe" <<< "$out"; then return 0; fi
    # Fall back to the full help blob.
    [[ -n "$HELP_ALL" ]] && grep -qE "^[[:space:]]*(-[a-zA-Z]+, )?${probe}([ ,=]|$)" <<< "$HELP_ALL"
}

echo "== Serve flags (per-flag 'vllm serve --help=<flag>') =="
if [[ "$HAVE_VLLM" == "1" ]]; then
    # Collect flags ONLY from inside COMMON_SERVE_ARGS / SERVE_ARGS / K3_BASE_ARGS
    # array blocks, so unrelated flags in the file (e.g. lm_eval's) are excluded.
    mapfile -t FLAGS < <(awk '
        /(COMMON_SERVE_ARGS|SERVE_ARGS|K3_BASE_ARGS)=\(/ { inarr=1; next }
        inarr && /^[[:space:]]*\)/                       { inarr=0; next }
        inarr                                            { print }
    ' "$REPO_ROOT/common.sh" "$REPO_ROOT"/servers/*.sh \
        | grep -oE -- '--[a-zA-Z0-9][a-zA-Z0-9-]*' | sort -u)
    # Flags appended conditionally by common_serve_args (not inside the array).
    FLAGS+=(--safetensors-load-strategy --tokenizer-mode)
    for fl in "${FLAGS[@]}"; do
        if flag_exists "$fl"; then ok "$fl"; else bad "$fl not accepted by this build"; fi
    done
else
    warn "vllm not found; skipped flag check"
fi

echo "== Enum VALUES we actually pass =="
# Match a value against the '--flag {a,b,c}' choice list in the full help.
check_choice() {  # $1=flag  $2=value
    local line set
    line="$(grep -F -- "$1 {" <<< "$HELP_ALL" | head -1)"
    if [[ -z "$line" ]]; then warn "$1: no choice list in help; skipped '$2'"; return; fi
    set="${line#*\{}"; set="${set%%\}*}"
    if grep -qE "(^|,)$2(,|\$)" <<< "$set"; then ok "$1 $2"; else bad "$1 $2 not in {$set}"; fi
}
if [[ -n "$HELP_ALL" ]]; then
    # Literal values in servers/*.sh arrays.
    for enum in --all2all-backend --expert-placement-strategy --performance-mode; do
        while IFS= read -r val; do
            [[ -n "$val" && "$val" != \"* && "$val" != \$* ]] && check_choice "$enum" "$val"
        done < <(grep -rhE "^[[:space:]]*$enum " "$REPO_ROOT"/servers/*.sh \
                 | awk -v f="$enum" '{for(i=1;i<NF;i++) if($i==f) print $(i+1)}' | sort -u)
    done
    # Values passed via env-var indirection consumed by k3_base_args / defaults.
    for v in auto deep_gemm_mega_moe; do check_choice --moe-backend "$v"; done
    for v in auto fp8;               do check_choice --kv-cache-dtype "$v"; done
    # --performance-mode default in opt05 comes from ${PERF_MODE:-throughput}.
    check_choice --performance-mode throughput
    # --load-format is a free-form string in this build, so a bad value fails only
    # at runtime. Verify ours appears in the help text.
    LF="${LOAD_FORMAT:-auto}"
    if grep -qF "\"$LF\"" <<< "$HELP_ALL"; then ok "--load-format $LF"; else
        bad "--load-format '$LF' is not one of the documented values (the recipe's 'fastsafetensors' does not exist in this image; 'instanttensor' is the fast-loader equivalent)"; fi
else
    warn "no --help=all output; skipped enum value check"
fi

echo "== Kimi-K3 parsers / tokenizer mode =="
if [[ -n "$HELP_ALL" ]]; then
    grep -q 'kimi_k3' <<< "$HELP_ALL" \
        && ok "kimi_k3 appears in help (tool-call parser / tokenizer mode)" \
        || bad "'kimi_k3' not found anywhere in help — this build may not know Kimi-K3"
    if [[ -n "${TOKENIZER_MODE:-}" ]]; then
        grep -q "'$TOKENIZER_MODE'" <<< "$HELP_ALL" \
            && ok "--tokenizer-mode $TOKENIZER_MODE available" \
            || bad "--tokenizer-mode '$TOKENIZER_MODE' not offered by this build"
    else
        echo "  TOKENIZER_MODE unset (vLLM default 'auto', Jinja chat template)."
        echo "  If the smoke test shows malformed chat output, set TOKENIZER_MODE=kimi_k3"
        echo "  (renders prompts with K3's XTML encoding instead of Jinja)."
    fi
fi

echo "== Backend enums (python registry) =="
if command -v python3 >/dev/null 2>&1; then
    python3 - <<'PY'
import importlib
def show(mod, name, want):
    try:
        m = importlib.import_module(mod)
        names = [x.name for x in getattr(m, name)]
    except Exception as exc:                      # noqa: BLE001
        print(f"  WARN {name}: could not import ({exc})")
        return
    missing = [w for w in want if w not in names]
    print(f"  {'FAIL' if missing else 'OK  '} {name}: "
          + (f"MISSING {missing}" if missing else "all required present"))
    print(f"       available: {', '.join(names)}")

# opt07 uses FLASHMLA; the DSpark verify step uses FLASHINFER_MLA.
show("vllm.v1.attention.backends.registry", "AttentionBackendEnum",
     ["FLASHMLA", "FLASHINFER_MLA"])
# opt06 uses FLASHINFER for the 69 KDA layers.
for mod in ("vllm.config.mamba", "vllm.config.cache", "vllm.config"):
    try:
        importlib.import_module(mod).MambaBackendEnum
        show(mod, "MambaBackendEnum", ["TRITON", "FLASHINFER"])
        break
    except Exception:                             # noqa: BLE001, S112
        continue
else:
    print("  WARN MambaBackendEnum: not found — verify --mamba-backend values by hand")
# opt15 uses FULL_DECODE_ONLY.
show("vllm.config", "CUDAGraphMode", ["FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"])
PY
    # --spec-method must offer BOTH mechanisms we compare (dspark vs in-model MTP).
    if [[ -n "$HELP_ALL" ]]; then
        for sm in dspark kimi_k3_mtp; do
            check_choice --spec-method "$sm"
        done
    fi
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
