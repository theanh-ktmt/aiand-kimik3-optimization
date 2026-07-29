#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# common.sh — shared config + helpers for Kimi-K3 optimization benchmarks
#
# Every server script under servers/ sources this file. It centralizes:
#   * the model / hardware defaults (8x B300, vLLM 0.27.x, vllm/vllm-openai:kimi-k3)
#   * the *mandatory* benchmark-harness flags that must be identical across
#     every configuration (most importantly --no-enable-prefix-caching)
#   * the recipe-mandated Kimi-K3 Blackwell block (k3_base_args / k3_env_defaults)
#   * the DSpark speculative-decoding JSON builder (dspark_config)
#   * server launch + readiness + cleanup helpers (reused from InferenceX)
#
# It does NOT define any optimization-specific flag — those live in the
# individual servers/*.sh scripts so each one is self-documenting.
#
# Source of truth for the base config:
#   https://recipes.vllm.ai/moonshotai/Kimi-K3
#   (vllm-project/recipes -> models/moonshotai/Kimi-K3.yaml)
#
# THE BASE PRESET (the team's docker launch command) — everything here is already
# ON for every config in this repo, so it is NOT something to "optimize into":
#   env  VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1  VLLM_ALLREDUCE_USE_FLASHINFER=1
#        VLLM_ENGINE_READY_TIMEOUT_S=3600  VLLM_USE_V2_MODEL_RUNNER=1
#        VLLM_USE_RUST_FRONTEND=1
#   args --trust-remote-code --load-format fastsafetensors --moe-backend auto
#        --gpu-memory-utilization 0.95 --tensor-parallel-size 8 --kv-cache-dtype fp8
#        --attention-config '{"mla_prefill_backend":"TRTLLM_RAGGED",
#                             "use_prefill_query_quantization":true}'
#        --enable-auto-tool-choice --tool-call-parser kimi_k3 --reasoning-parser kimi_k3
#
# For a preset item the useful experiment is to SKIP it or use a DIFFERENT VALUE,
# never to re-enable it. Two deliberate deviations from the preset:
#   * --enable-prefix-caching  -> we force it OFF (team decision, see below)
#   * --max-model-len 1048576  -> we cap at 16384 so KV capacity is not the
#     variable under test; the recipe itself says to adjust it per benchmark
# ---------------------------------------------------------------------------
set -uo pipefail

# --- Resolve repo root & third-party InferenceX -----------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT
INFERENCEX_DIR="${INFERENCEX_DIR:-$REPO_ROOT/third_party/InferenceX}"
export INFERENCEX_DIR

if [[ ! -f "$INFERENCEX_DIR/benchmarks/benchmark_lib.sh" ]]; then
    echo "ERROR: InferenceX not found at $INFERENCEX_DIR"
    echo "       Run: git submodule update --init --recursive"
    exit 1
fi
# Reuse InferenceX helpers: wait_for_server_ready, start/stop_gpu_monitor, etc.
# shellcheck source=/dev/null
source "$INFERENCEX_DIR/benchmarks/benchmark_lib.sh"

# --- Model / hardware defaults ---------------------------------------------
# Kimi-K3: 2.8T total / 16-of-896 routed experts active, MXFP4 weights +
# MXFP8 activations, 93 layers (1 dense + 69 KDA + 24 Gated MLA), 1M context.
# ~1.4 TB of weights => ~175 GB/GPU on 8x B300 (288 GB each). Fits one node.
export MODEL="${MODEL:-moonshotai/Kimi-K3}"
export MODEL_PATH="${MODEL_PATH:-}"        # local pre-staged dir, optional
export SERVE_MODEL="${MODEL_PATH:-$MODEL}"
# DSpark draft model for speculative decoding (Inferact-trained, per the recipe).
export DRAFT_MODEL="${DRAFT_MODEL:-Inferact/Kimi-K3-DSpark}"
export PORT="${PORT:-8888}"
export TP="${TP:-8}"                        # 8x B300 single replica
export NUM_GPUS="${NUM_GPUS:-8}"            # used by aggregator for per-GPU tput

# Benchmark-harness invariants (apply to EVERY config; not optimizations).
# The recipe serves 1M context; for benchmarking we cap max-model-len so KV cache
# is not the bottleneck and every scenario still fits (8k ISL + 1k OSL + buffer).
# The recipe explicitly says: "Adjust max-model-len for different benchmark
# scenarios for best performance."
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

# Weight loader. `fastsafetensors` is what the recipe asks for AND what both of
# InferenceX's own Kimi-K3 B300 scripts actually run
# (third_party/InferenceX/benchmarks/single_node/{speedbench,agentic}/kimik3_fp4_b300_vllm.sh),
# so it is proven on this image — even though it is NOT in the documented
# --load-format list from `vllm serve --help=all` (auto, pt, safetensors,
# instanttensor, npcache, dummy, tensorizer, runai_streamer,
# runai_streamer_sharded, bitsandbytes, sharded_state, mistral, modelexpress).
# The help ends that list with "Other custom values can be supported via
# plugins", which is presumably how the kimi-k3 image provides it.
#
# If it ever fails to resolve, `instanttensor` is the in-tree equivalent
# ("distributed loading with pipelined prefetching and fast direct I/O"):
#     LOAD_FORMAT=instanttensor SAFETENSORS_LOAD_STRATEGY=eager bash run_all.sh
export LOAD_FORMAT="${LOAD_FORMAT:-fastsafetensors}"
export SAFETENSORS_LOAD_STRATEGY="${SAFETENSORS_LOAD_STRATEGY:-}"

# Optional tokenizer mode. This image offers `kimi_k3`, which "will always use
# the hf tokenizer but render chat prompts with Kimi K3's Python XTML encoding
# instead of a Jinja template". Leave unset (vLLM default `auto`) unless the
# smoke test shows malformed chat output — then set TOKENIZER_MODE=kimi_k3 for
# BOTH server and benchmark client (bench.sh forwards it).
export TOKENIZER_MODE="${TOKENIZER_MODE:-}"

# Where vLLM looks for / caches model weights (cloud container default).
export DOWNLOAD_DIR="${DOWNLOAD_DIR:-/workspace/models}"

# Per-config save directory. serve_main sets SAVE_DIR=results/<CONFIG> and points
# SERVER_LOG at it; this top-level value is only a fallback for direct callers.
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/results}"
SERVER_LOG="${SERVER_LOG:-$SAVE_DIR/server.log}"
export SAVE_DIR SERVER_LOG
mkdir -p "$SAVE_DIR"

# --- Mandatory flags shared by all configurations --------------------------
# IMPORTANT: --no-enable-prefix-caching is REQUIRED. With prefix caching on,
# repeated warmup / multi-turn ShareGPT prefixes are served from cache, prefill
# is skipped, and throughput is inflated. This must never be removed for any
# config.
#
# IMPORTANT DIVERGENCE TO BE AWARE OF: the published Kimi-K3 Blackwell recipe sets
# --enable-prefix-caching, and so do BOTH of InferenceX's own Kimi-K3 B300 scripts
# ("prefix caching ENABLED (production recipe) — was disabled"). We deliberately do
# NOT follow them: this harness's job is to compare configurations, and cache hits
# would mask prefill cost.
#
# This was decided explicitly (2026-07-29) after the divergence was raised — it is
# a settled choice, not an untuned default, so there is no switch to flip. The
# consequence must be stated whenever these numbers appear next to a recipe or
# InferenceX K3 result: OURS ARE A FLOOR. Config-to-config comparisons within this
# repo are unaffected, because the setting is identical everywhere.
#
# Populates the global array COMMON_SERVE_ARGS. An array (rather than echoing a
# string and re-splitting with `read -a`) is REQUIRED so that flag *values*
# containing spaces survive intact — e.g. a JSON value like
# --compilation-config '{"cudagraph_mode": "PIECEWISE"}'. Word-splitting an
# echoed string would break such a value into two tokens and hand vLLM
# truncated, invalid JSON.
common_serve_args() {
    # --served-model-name pins the API model id to $MODEL even when we launch
    # from a local weights dir ($MODEL_PATH), so the benchmark client's
    # --model "$MODEL" always matches (otherwise the server advertises the path
    # and requests 404).
    COMMON_SERVE_ARGS=(
        --host 0.0.0.0 --port "$PORT"
        --served-model-name "$MODEL"
        --trust-remote-code
        --disable-uvicorn-access-log
        --download-dir "$DOWNLOAD_DIR"
        --max-model-len "$MAX_MODEL_LEN"
        --load-format "$LOAD_FORMAT"
        --tool-call-parser kimi_k3
        --enable-auto-tool-choice
        --reasoning-parser kimi_k3
    )
    # Prefix caching is OFF, always, with no override. See the block comment above:
    # this is a team decision, not a default waiting to be tuned.
    COMMON_SERVE_ARGS+=(--no-enable-prefix-caching)
    # Guard against a stray env var creating silently non-comparable results.
    if [[ -n "${PREFIX_CACHING:-}" && "${PREFIX_CACHING}" != "0" ]]; then
        echo "ERROR: PREFIX_CACHING is set to '${PREFIX_CACHING}', but prefix caching" >&2
        echo "       is a hard invariant of this harness and cannot be enabled." >&2
        echo "       Unset it and re-run." >&2
        return 1
    fi
    if [[ -n "$SAFETENSORS_LOAD_STRATEGY" ]]; then
        COMMON_SERVE_ARGS+=(--safetensors-load-strategy "$SAFETENSORS_LOAD_STRATEGY")
    fi
    if [[ -n "$TOKENIZER_MODE" ]]; then
        COMMON_SERVE_ARGS+=(--tokenizer-mode "$TOKENIZER_MODE")
    fi
    # Explicit success. Without this the function's exit status is that of the
    # last conditional, so an unset TOKENIZER_MODE would make it "fail" — and
    # launch_vllm now checks the return value.
    return 0
}

# --- Recipe-mandated Kimi-K3 base block ------------------------------------
# These come from the recipe's own `base_args` + `hardware_overrides.blackwell`
# + `hardware_overrides.nvidia`, i.e. they are the *published starting point*
# for B300, not optimizations we invented. Every servers/*.sh splices them in as
# "${K3_BASE_ARGS[@]}" so each script stays self-documenting and any single opt
# can override one of them by repeating the flag later in argv (last wins).
#
#   --gpu-memory-utilization 0.95   recipe base_args
#   --moe-backend auto              recipe base_args
#   --no-enable-flashinfer-autotune recipe hardware_overrides.nvidia
#   --kv-cache-dtype fp8            recipe hardware_overrides.blackwell
#   --attention-config ...          recipe hardware_overrides.blackwell
#       mla_prefill_backend TRTLLM_RAGGED  = TRT-LLM ragged MLA prefill kernels
#       use_prefill_query_quantization     = REQUIRED whenever KV cache is FP8
#
# NOTE on the JSON defaults below: do NOT write them as ${VAR:-{"a":1}} — bash's
# ${...} parser stops at the first '}' inside the default, so an *override* leaks
# a stray '}' into the value. Always default with an explicit `[[ -z ]]` test.
K3_ATTENTION_CONFIG_DEFAULT='{"mla_prefill_backend":"TRTLLM_RAGGED","use_prefill_query_quantization":true}'
k3_base_args() {
    local attn_cfg="${ATTENTION_CONFIG:-}"
    [[ -z "$attn_cfg" ]] && attn_cfg="$K3_ATTENTION_CONFIG_DEFAULT"
    K3_BASE_ARGS=(
        --gpu-memory-utilization "${GPU_MEM_UTIL:-0.95}"
        --moe-backend "${MOE_BACKEND:-auto}"
        --no-enable-flashinfer-autotune
        --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}"
        --attention-config "$attn_cfg"
    )
}

# Recipe-mandated env for Blackwell. Every value uses ${VAR:-default} so an opt
# script can flip one by exporting it *before* calling serve_main.
#   VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION  fuses the LatentMoE tail (K3-specific)
#   VLLM_ALLREDUCE_USE_FLASHINFER          FlashInfer allreduce kernels
#   VLLM_ENGINE_READY_TIMEOUT_S            3600 — a 1.4 TB load is slow
#
# The last four come from InferenceX's own Kimi-K3 B300 scripts rather than the
# recipe yaml: NCCL_DMABUF_ENABLE=0 and PYTHONNOUSERSITE=1 are part of their
# "Kimi-K3 production serving environment", and VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900
# is there because the Rust frontend's 5s default closes pooled keep-alive
# sockets mid-benchmark (they lost a whole run to that race).
k3_env_defaults() {
    export VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION="${VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION:-1}"
    export VLLM_ALLREDUCE_USE_FLASHINFER="${VLLM_ALLREDUCE_USE_FLASHINFER:-1}"
    export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
    # Model Runner v2 + Rust frontend are part of the BASE PRESET (they are in the
    # team's docker launch command). They are therefore NOT screened at all — not
    # on (already on) and not off either, since flipping a setting the team has
    # committed to would spend a ~1.4 TB launch on a decision nobody will revisit.
    export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
    export VLLM_USE_RUST_FRONTEND="${VLLM_USE_RUST_FRONTEND:-1}"
    export NCCL_DMABUF_ENABLE="${NCCL_DMABUF_ENABLE:-0}"
    export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
    export VLLM_HTTP_TIMEOUT_KEEP_ALIVE="${VLLM_HTTP_TIMEOUT_KEEP_ALIVE:-900}"
}

# --- Speculative decoding: TWO methods are available for Kimi-K3 ------------
# `vllm serve --help=all` shows --spec-method accepting BOTH:
#
#   dspark        the recipe's choice: a separate Inferact-trained draft model
#                 ($DRAFT_MODEL), with its own verify attention backend,
#                 draft-sampling and rejection-sampling methods.
#   kimi_k3_mtp   an in-model MTP head, the same shape of mechanism GLM-5.2 used.
#                 The recipe never mentions it, but the engine supports it — and
#                 comparing the two is the whole point of run_spec_sweep.sh.
#
# Builders below; both emit a --speculative-config JSON.
#
#   dspark_config       [num_speculative_tokens]   # default 7 (recipe default)
#   kimi_k3_mtp_config  [num_speculative_tokens]   # default 3
#
# Overridable per script via SPEC_ATTN_BACKEND / DRAFT_SAMPLE_METHOD /
# REJECTION_SAMPLE_METHOD. Extra JSON keys can be appended with SPEC_EXTRA_JSON
# (e.g. a num_speculative_tokens_per_batch_size schedule).
#
# NOTE on draft_sample_method / rejection_sample_method: the recipe yaml sets them
# (probabilistic / block) and so do we, but InferenceX's baseline K3 script OMITS
# both and ships them only in a separate
# `..._probabilistic_sample_method_block_rejection_sample_method.sh` variant. So
# "recipe default" and "InferenceX default" differ here; set
# DRAFT_SAMPLE_METHOD= / REJECTION_SAMPLE_METHOD= to empty-string-free values or
# edit dspark_config if you need the bare form.
dspark_config() {
    local n="${1:-7}"
    local draft="${DRAFT_MODEL_PATH:-$DRAFT_MODEL}"
    local extra="${SPEC_EXTRA_JSON:-}"
    printf '{"model":"%s","num_speculative_tokens":%s,"method":"dspark","attention_backend":"%s","draft_sample_method":"%s","rejection_sample_method":"%s"%s}' \
        "$draft" "$n" \
        "${SPEC_ATTN_BACKEND:-FLASHINFER_MLA}" \
        "${DRAFT_SAMPLE_METHOD:-probabilistic}" \
        "${REJECTION_SAMPLE_METHOD:-block}" \
        "${extra:+,$extra}"
}

# spec_config — dispatcher used by servers/spec.sh so a single script covers the
# whole speculative-decoding sweep.
#
#   SPEC_METHOD=dspark|kimi_k3_mtp|none   SPEC_TOKENS=<n>
#
# Prints nothing for "none" (the caller then omits --speculative-config entirely).
spec_config() {
    local method="${1:-${SPEC_METHOD:-dspark}}" n="${2:-${SPEC_TOKENS:-3}}"
    case "$method" in
        dspark)      dspark_config "$n" ;;
        kimi_k3_mtp) kimi_k3_mtp_config "$n" ;;
        none)        : ;;
        *) echo "ERROR: unknown SPEC_METHOD '$method' (want dspark|kimi_k3_mtp|none)" >&2
           return 1 ;;
    esac
}

# In-model MTP head. No draft model to download, no draft_sample_method — the
# extra keys DSpark needs do not apply here, so keep the JSON minimal.
kimi_k3_mtp_config() {
    local n="${1:-3}"
    local extra="${SPEC_EXTRA_JSON:-}"
    printf '{"method":"kimi_k3_mtp","num_speculative_tokens":%s%s}' \
        "$n" "${extra:+,$extra}"
}

# --- Server lifecycle -------------------------------------------------------
SERVER_PID=""

_descendants() {
    local pid="$1" child
    for child in $(pgrep -P "$pid" 2>/dev/null || true); do
        echo "$child"; _descendants "$child"
    done
}

cleanup_server() {
    [[ -z "$SERVER_PID" ]] && return 0
    local descendants; descendants=$(_descendants "$SERVER_PID")
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    local pid; for pid in $descendants; do kill -9 "$pid" 2>/dev/null || true; done
    # Wait for GPU memory to drain before the next config launches.
    local waited=0
    while [[ $waited -lt 180 ]]; do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)
        [[ -z "$used" || "$used" -lt 2000 ]] && break
        sleep 3; waited=$((waited + 3))
    done
    SERVER_PID=""
}

# Ensure weights are available.
#   * If MODEL_PATH points to an explicit local dir, download into it when empty.
#   * Otherwise rely on vLLM's --download-dir ($DOWNLOAD_DIR, e.g. the cloud
#     container's /workspace/models cache): vLLM loads from there if present and
#     fetches into it if not, so no separate pre-download is needed.
#   * Same for the DSpark draft model (DRAFT_MODEL_PATH), which every
#     speculative-decoding config needs in addition to the 1.4 TB target.
ensure_model() {
    if [[ -n "$MODEL_PATH" ]]; then
        if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
            echo "=== MODEL_PATH ($MODEL_PATH) empty, downloading $MODEL ==="
            hf download "$MODEL" --local-dir "$MODEL_PATH"
        fi
    else
        echo "=== Using vLLM --download-dir cache: $DOWNLOAD_DIR (model: $SERVE_MODEL) ==="
    fi
    if [[ -n "${DRAFT_MODEL_PATH:-}" ]]; then
        if [[ ! -d "$DRAFT_MODEL_PATH" || -z "$(ls -A "$DRAFT_MODEL_PATH" 2>/dev/null)" ]]; then
            echo "=== DRAFT_MODEL_PATH ($DRAFT_MODEL_PATH) empty, downloading $DRAFT_MODEL ==="
            hf download "$DRAFT_MODEL" --local-dir "$DRAFT_MODEL_PATH"
        fi
    fi
}

# Bounded readiness wait. Unlike InferenceX's wait_for_server_ready (which loops
# forever until healthy), this gives up after SERVER_STARTUP_TIMEOUT seconds so a
# hung / crash-looping launch fails the run instead of blocking. Returns non-zero
# on timeout or if the server process dies.
#
# Default is 3600s (60 min), NOT the 20 min used for GLM-5.2: Kimi-K3 is a 2.8T
# MXFP4 checkpoint (~1.4 TB) and the recipe itself sets
# VLLM_ENGINE_READY_TIMEOUT_S=3600 for exactly this reason.
SERVER_STARTUP_TIMEOUT="${SERVER_STARTUP_TIMEOUT:-3600}"
export SERVER_STARTUP_TIMEOUT

wait_for_health() {
    local timeout="$SERVER_STARTUP_TIMEOUT" interval=5 waited=0
    # Wait for the log file to appear (container startup may delay it).
    while [[ ! -f "$SERVER_LOG" ]]; do
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "Server died before creating log."; return 1; }
        sleep 1; waited=$((waited + 1)); (( waited >= timeout )) && { echo "Timed out waiting for log."; return 1; }
    done
    # Stream the log while we poll /health.
    tail -f -n +1 "$SERVER_LOG" & local tail_pid=$!
    while true; do
        if curl --output /dev/null --silent --fail "http://0.0.0.0:$PORT/health"; then
            kill "$tail_pid" 2>/dev/null; return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Server process died before becoming healthy."; kill "$tail_pid" 2>/dev/null; return 1
        fi
        if (( waited >= timeout )); then
            echo "Server not healthy within ${timeout}s; giving up."; kill "$tail_pid" 2>/dev/null; return 1
        fi
        sleep "$interval"; waited=$((waited + interval))
    done
}

# launch_vllm <config-name> <extra serve args...>
# Combines common flags + caller's optimization flags, starts the server,
# waits for /health (bounded by SERVER_STARTUP_TIMEOUT), and leaves SERVER_PID set.
launch_vllm() {
    local config_name="$1"; shift
    ensure_model
    nvidia-smi || true

    local -a args
    common_serve_args || return 1
    args=("${COMMON_SERVE_ARGS[@]}" "$@")

    # Build a copy-pasteable, safely-quoted command string.
    local cmd_str a
    cmd_str="vllm serve $(printf '%q' "$SERVE_MODEL")"
    for a in "${args[@]}"; do cmd_str+=" $(printf '%q' "$a")"; done

    echo "=============================================================="
    echo "  Launching config: $config_name"
    echo "  $cmd_str"
    echo "=============================================================="

    # Record the exact command + relevant env into the head of the server log,
    # so each log is self-describing and reproducible.
    {
        echo "# =========================================================="
        echo "# config : $config_name"
        echo "# date   : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "# model  : $SERVE_MODEL"
        echo "# draft  : ${DRAFT_MODEL_PATH:-$DRAFT_MODEL}"
        echo "# env    :" \
             "VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=${VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION:-<unset>}" \
             "VLLM_ALLREDUCE_USE_FLASHINFER=${VLLM_ALLREDUCE_USE_FLASHINFER:-<unset>}" \
             "VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-<unset>}" \
             "VLLM_USE_RUST_FRONTEND=${VLLM_USE_RUST_FRONTEND:-<unset>}" \
             "VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-<unset>}" \
             "VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-<unset>}" \
             "VLLM_ALL2ALL_BACKEND=${VLLM_ALL2ALL_BACKEND:-<unset>}" \
             "UCX_TLS=${UCX_TLS:-<unset>}"
        echo "# command:"
        echo "$cmd_str"
        echo "# =========================================================="
        echo
    } > "$SERVER_LOG"

    vllm serve "$SERVE_MODEL" "${args[@]}" >> "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!

    if ! wait_for_health; then
        echo "ERROR: server failed to become healthy for config $config_name (timeout=${SERVER_STARTUP_TIMEOUT}s)"
        cleanup_server
        return 1
    fi
    echo "Server ready (config=$config_name pid=$SERVER_PID)"
}

# Chat-template kwargs used by the eval + smoke test to suppress reasoning
# output. Kimi-K3 has a reasoning parser (--reasoning-parser kimi_k3); the exact
# template key to turn thinking off is model-specific, so we pass the two common
# spellings. Unknown keys are ignored by the Jinja template, so this is safe;
# override with K3_CHAT_TEMPLATE_KWARGS if K3 uses a different one.
if [[ -z "${K3_CHAT_TEMPLATE_KWARGS:-}" ]]; then
    K3_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false, "thinking": false}'
fi
export K3_CHAT_TEMPLATE_KWARGS

# _eval_gen_kwargs — the --gen_kwargs JSON for lm-eval (thinking off).
_eval_gen_kwargs() {
    if [[ -n "${EVAL_GEN_KWARGS:-}" ]]; then printf '%s' "$EVAL_GEN_KWARGS"
    else printf '{"chat_template_kwargs": %s}' "$K3_CHAT_TEMPLATE_KWARGS"; fi
}

# run_mmlu_pro — MMLU-Pro accuracy check against the live server with lm-eval,
# hitting the OpenAI /v1/chat/completions endpoint. Two requirements baked in:
#   * results are saved (--output_path + --log_samples)
#   * THINKING IS OFF — K3 is a reasoning model; for MMLU-Pro we disable it via
#     --gen_kwargs chat_template_kwargs so outputs are the answer, not CoT.
# Tunables: MMLU_PRO_TASK (default mmlu_pro), EVAL_CONC (default 64),
#           EVAL_GEN_KWARGS (override the gen_kwargs JSON).
run_mmlu_pro() {
    local out="$SAVE_DIR/mmlu_pro"
    mkdir -p "$out"
    local task="${MMLU_PRO_TASK:-mmlu_pro}"
    local conc="${EVAL_CONC:-64}"
    local base="http://0.0.0.0:$PORT/v1/chat/completions"
    local gen_kwargs; gen_kwargs="$(_eval_gen_kwargs)"
    export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

    # Ensure lm-eval is available (recent version: supports mmlu_pro + JSON gen_kwargs).
    python3 -c "import lm_eval" 2>/dev/null || pip install -q "lm-eval[api]" 2>/dev/null || true

    echo "=============================================================="
    echo "  MMLU-Pro eval: config=$CONFIG task=$task conc=$conc thinking=OFF"
    echo "  gen_kwargs: $gen_kwargs"
    echo "  results -> $out"
    echo "=============================================================="

    # tokenized_requests=False -> client doesn't need the HF tokenizer (offline-safe).
    python3 -m lm_eval \
        --model local-chat-completions \
        --tasks "$task" \
        --output_path "$out" \
        --apply_chat_template \
        --log_samples \
        --gen_kwargs "$gen_kwargs" \
        --model_args "model=$MODEL,base_url=$base,api_key=$OPENAI_API_KEY,num_concurrent=$conc,tokenized_requests=False,max_retries=5,timeout=1800" \
        2>&1 | tee "$out/lm_eval.log"
    return "${PIPESTATUS[0]}"
}

# run_mmmu_pro — vision quality check. Unlike GLM-5.2 (text-only), Kimi-K3 is
# NATIVELY MULTIMODAL (MoonViT-V2 encoder), so the image benchmark applies and is
# part of the quality gate. Best-effort: lm-eval's API-backed multimodal support
# is thinner than its text path, so this is opt-in (RUN_MMMU=1) and a failure
# here does not invalidate the MMLU-Pro verdict.
#
# NOTE: never run this against a config that passes --language-model-only — that
# flag drops the vision encoder and every image request will fail.
run_mmmu_pro() {
    local out="$SAVE_DIR/mmmu_pro"
    mkdir -p "$out"
    local task="${MMMU_PRO_TASK:-mmmu_pro}"
    local conc="${EVAL_CONC:-32}"
    local base="http://0.0.0.0:$PORT/v1/chat/completions"
    local gen_kwargs; gen_kwargs="$(_eval_gen_kwargs)"
    export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

    python3 -c "import lm_eval" 2>/dev/null || pip install -q "lm-eval[api]" 2>/dev/null || true

    echo "=============================================================="
    echo "  MMMU-Pro eval (vision): config=$CONFIG task=$task conc=$conc"
    echo "  results -> $out"
    echo "=============================================================="

    python3 -m lm_eval \
        --model local-chat-completions \
        --tasks "$task" \
        --output_path "$out" \
        --apply_chat_template \
        --log_samples \
        --gen_kwargs "$gen_kwargs" \
        --model_args "model=$MODEL,base_url=$base,api_key=$OPENAI_API_KEY,num_concurrent=$conc,tokenized_requests=False,max_images=7,max_retries=5,timeout=1800" \
        2>&1 | tee "$out/lm_eval.log"
    return "${PIPESTATUS[0]}"
}

# smoke_test — sanity-check the live server BEFORE benchmarking: send a few
# sample chat requests and print prompt + response to stdout, so you can eyeball
# that the server answers correctly (and the chat template / DSpark work).
# Thinking is off for concise, easy-to-verify answers.
# Skip with SMOKE_TEST=0; abort the run on total failure with SMOKE_TEST_STRICT=1.
# Add an image request (K3 is multimodal) with SMOKE_TEST_MM=1 — do NOT enable it
# for --language-model-only configs.
smoke_test() {
    local base="http://0.0.0.0:$PORT/v1/chat/completions"
    local prompts=(
        "What is the capital of France? Answer in one word."
        "Compute 17 * 23. Give only the number."
        "Reply with exactly this word: OK"
    )
    echo "=============================================================="
    echo "  SMOKE TEST ($CONFIG): ${#prompts[@]} sample requests -> $base"
    echo "=============================================================="
    local ok=0 i=0 p payload resp
    for p in "${prompts[@]}"; do
        i=$((i + 1))
        echo ">>> [$i] prompt: $p"
        payload=$(MODEL="$MODEL" PROMPT="$p" CTK="$K3_CHAT_TEMPLATE_KWARGS" python3 -c '
import json, os
print(json.dumps({
    "model": os.environ["MODEL"],
    "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
    "max_tokens": 256,
    "temperature": 0,
    "chat_template_kwargs": json.loads(os.environ["CTK"]),
}))')
        resp=$(curl -sS -m 180 -X POST "$base" \
            -H 'Content-Type: application/json' -d "$payload" 2>&1)
        if printf '%s' "$resp" | python3 -c '
import json, sys
raw = sys.stdin.read()
try:
    d = json.loads(raw)
except Exception:
    print("    ERROR: non-JSON response:", raw[:300]); sys.exit(2)
if "choices" not in d:
    print("    ERROR: no choices; server said:", json.dumps(d)[:300]); sys.exit(2)
ch = (d.get("choices") or [{}])[0]
msg = ch.get("message", {}) or {}
content = msg.get("content")
reasoning = msg.get("reasoning_content")
usage = d.get("usage", {}) or {}
if reasoning:
    print("    reasoning:", reasoning[:160] + ("..." if len(reasoning) > 160 else ""))
print("    response:", content if content not in (None, "") else "<empty>")
print("    finish:", ch.get("finish_reason"), "| completion_tokens:", usage.get("completion_tokens"))
sys.exit(0 if content not in (None, "") else 3)
'; then
            ok=$((ok + 1))
        else
            echo "    (request $i did not return usable content)"
        fi
    done

    if [[ "${SMOKE_TEST_MM:-0}" == "1" ]]; then
        i=$((i + 1))
        echo ">>> [$i] prompt: <image> Read all the text in the image."
        payload=$(MODEL="$MODEL" IMG="${SMOKE_IMAGE_URL:-https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png}" python3 -c '
import json, os
print(json.dumps({
    "model": os.environ["MODEL"],
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": os.environ["IMG"]}},
        {"type": "text", "text": "Read all the text in the image."},
    ]}],
    "max_tokens": 512,
    "temperature": 0,
}))')
        resp=$(curl -sS -m 300 -X POST "$base" -H 'Content-Type: application/json' -d "$payload" 2>&1)
        if printf '%s' "$resp" | python3 -c '
import json, sys
try:
    d = json.loads(sys.stdin.read())
except Exception as e:
    print("    ERROR: non-JSON response"); sys.exit(2)
c = ((d.get("choices") or [{}])[0].get("message") or {}).get("content")
print("    response:", (c[:300] + "...") if c else "<empty>")
sys.exit(0 if c else 3)
'; then ok=$((ok + 1)); else echo "    (image request failed — expected if --language-model-only)"; fi
    fi

    echo "  SMOKE TEST: $ok/$i requests returned content"
    if [[ "$ok" -eq 0 ]]; then
        echo "  WARNING: server produced no usable output on any smoke request." >&2
        [[ "${SMOKE_TEST_STRICT:-0}" == "1" ]] && return 1
    fi
    return 0
}

# serve_main — entrypoint every servers/*.sh calls after defining:
#     CONFIG       config label (e.g. opt09_moe_deepgemm_mega)
#     BENCH_MODE   mtp | nonmtp  (InferenceX semantics: mtp adds --use-chat-template
#                  on the client, and additionally runs the ShareGPT lane)
#     SERVE_ARGS   bash array of the optimization-specific serve flags
#
# Default: launch the server, wait for health, stay in the foreground so you
# can benchmark it from another shell (matches the manual workflow).
# Set RUN_BENCH=1 (run.sh does this) to launch -> sweep -> tear down end-to-end.
# Set RUN_EVAL=1 to launch -> MMLU-Pro (+ optional MMMU-Pro) -> tear down.
serve_main() {
    k3_env_defaults

    # Everything for this config lands in results/<CONFIG>/.
    SAVE_DIR="$REPO_ROOT/results/$CONFIG"
    SERVER_LOG="$SAVE_DIR/server.log"
    export SAVE_DIR SERVER_LOG
    mkdir -p "$SAVE_DIR"

    trap cleanup_server EXIT INT TERM
    start_gpu_monitor --output "$SAVE_DIR/gpu.csv" 2>/dev/null || true
    launch_vllm "$CONFIG" "${SERVE_ARGS[@]}" || exit 1

    # Sanity-check the server before spending time on a full sweep.
    if [[ "${SMOKE_TEST:-1}" != "0" ]]; then
        if ! smoke_test; then
            echo "ERROR: smoke test failed (SMOKE_TEST_STRICT=1); tearing down." >&2
            stop_gpu_monitor 2>/dev/null || true
            cleanup_server
            exit 1
        fi
    fi

    if [[ "${RUN_EVAL:-0}" == "1" ]]; then
        run_mmlu_pro
        [[ "${RUN_MMMU:-0}" == "1" ]] && run_mmmu_pro
        stop_gpu_monitor 2>/dev/null || true
        cleanup_server
    elif [[ "${RUN_BENCH:-0}" == "1" ]]; then
        CONFIG="$CONFIG" BENCH_MODE="$BENCH_MODE" SWEEP="${SWEEP:-full}" \
        DATASETS="${DATASETS:-}" PORT="$PORT" \
            bash "$REPO_ROOT/bench/bench.sh" --config "$CONFIG" --mode "$BENCH_MODE"
        stop_gpu_monitor 2>/dev/null || true
        cleanup_server
    else
        echo
        echo "Server is UP on port $PORT  (config=$CONFIG, bench mode=$BENCH_MODE)."
        echo "Benchmark it from another shell with:"
        echo "  CONFIG=$CONFIG bash bench/bench.sh --config $CONFIG --mode $BENCH_MODE --sweep subset"
        echo "Press Ctrl-C to stop the server."
        wait "$SERVER_PID"
    fi
}
