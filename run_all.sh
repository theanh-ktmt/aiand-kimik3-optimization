#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_all.sh — run the whole optimization campaign in the prescribed order:
#   1. baseline      -> SUBSET sweep  (reference curve; set BASELINE_SWEEP=full for the final)
#   2. each opt      -> SUBSET sweep (OSL 1024 only, conc 1/16/128) for screening
#   3. (later) build final config and re-run FULL via run_final.sh.
#
# Each config is launched, benchmarked, and torn down before the next one.
# Results land in results/<config>/ and results/<config>.csv; a combined
# results/all.csv is written at the end for pasting into the sheet.
#
# ---------------------------------------------------------------------------
# !! COST WARNING, READ THIS BEFORE LAUNCHING THE FULL CAMPAIGN !!
# Kimi-K3 is a 2.8T MXFP4 checkpoint (~1.4 TB). Cold-loading it takes tens of
# minutes per config — the recipe itself sets VLLM_ENGINE_READY_TIMEOUT_S=3600.
# With ~35 configs, *startup alone* can dominate the wall clock and swamp the
# benchmark time. Practical advice:
#   * Keep the weights on fast local NVMe and use --load-format fastsafetensors
#     (already the default here) — this is the single biggest lever on total time.
#   * Screen in stages with --only, deciding the parallelism group (opt01) first,
#     then the attention/MoE groups, then DSpark. Most opt05/06/07 configs are
#     only worth running if DP8EP beat TP8.
#   * For a group that shares one server config (e.g. the DSpark token sweep),
#     consider launching the server once by hand and re-running bench/bench.sh
#     against it instead of paying the reload each time.
# ---------------------------------------------------------------------------
#
# Usage:
#   bash run_all.sh                 # all configs (baseline + opts) subset
#   bash run_all.sh --only opt04a_moe_deepgemm_mega opt09_flashinfer_sampler
#   SKIP_EXISTING=1 bash run_all.sh # resume: skip configs already done
#                                   #   (those with results/<cfg>.csv)
#   BASELINE_SWEEP=full bash run_all.sh     # run baseline as a full reference sweep
#   FULL_TIMEOUT=28800 bash run_all.sh      # raise the full-sweep cap
#   CONFIG_TIMEOUT=0 bash run_all.sh        # disable per-config caps entirely
#
# Resilience: every config already fails fast on a stuck *startup*
# (SERVER_STARTUP_TIMEOUT in common.sh, default 3600s) and the campaign continues
# on any failure. Additionally:
#   * A HARD per-config wall-clock cap covers post-startup hangs (stuck benchmark
#     cell / teardown). It is PER SWEEP: SUBSET_TIMEOUT (default 7200s = 2h) for
#     screening configs, FULL_TIMEOUT (default 21600s = 6h) for full sweeps.
#     Both are much larger than the GLM harness's because the 1.4 TB load is
#     inside the cap. Set CONFIG_TIMEOUT to force one cap for all (0 = disable).
#     On expiry the run is killed (SIGTERM, then SIGKILL after 60s) and skipped.
#   * After every config (success, failure, or timeout) reap_gpu force-kills any
#     stray vLLM process and VERIFIES the GPU is free before the next config,
#     so an orphaned/killed server never OOMs the next run.
# ---------------------------------------------------------------------------
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
[[ -f "$REPO_ROOT/.env" ]] && { set -a; source <(tr -d '\r' < "$REPO_ROOT/.env"); set +a; }
BASELINE_SWEEP="${BASELINE_SWEEP:-subset}"
OPT_SWEEP="${OPT_SWEEP:-subset}"
export DATASET="${DATASET:-sharegpt}"
SUBSET_TIMEOUT="${SUBSET_TIMEOUT:-7200}"    # 2h per screening (subset) config
FULL_TIMEOUT="${FULL_TIMEOUT:-21600}"       # 6h per full-sweep config
CONFIG_TIMEOUT="${CONFIG_TIMEOUT:-}"        # optional: one cap for ALL configs (0=disable)

# Make sure NO vLLM process is left holding the GPU before the next config.
# The launcher's cmdline is 'vllm serve ...', but the memory is held by the
# engine/worker processes whose titles are 'VLLM::EngineCore' / 'VllmWorker'
# (set via setproctitle) — so we match both, escalate to SIGKILL, and poll
# nvidia-smi until memory actually drains.
reap_gpu() {
    echo ">>> reap_gpu: ensuring no vLLM process holds the GPU"
    pkill -TERM -f 'vllm serve' 2>/dev/null || true
    pkill -TERM VLLM            2>/dev/null || true
    sleep 5
    local waited=0 used
    while [[ $waited -lt 300 ]]; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)
        if [[ -z "$used" || "$used" -lt 2000 ]]; then
            echo ">>> reap_gpu: GPU clear (max used=${used:-NA} MiB)"
            return 0
        fi
        echo ">>> reap_gpu: GPU still busy (${used} MiB) — pkill -9 vLLM"
        pkill -9 -f 'vllm serve' 2>/dev/null || true
        pkill -9 VLLM            2>/dev/null || true
        pkill -9 -f 'vllm'       2>/dev/null || true
        sleep 5; waited=$((waited + 5))
    done
    echo "WARN: GPU still >2000 MiB used after reap — next config may OOM." >&2
}

# Screening order. Decide the parallelism choice (opt01) first: the DP8EP-only
# opts (a2a / EPLB / DBO) are grouped after it, and are only worth running if
# DP8EP actually beat TP8.
ALL_CONFIGS=(
    baseline
    opt01a_tp8ep opt01b_dp8ep opt01c_tp4dp2ep
    opt02a_maxseqs128 opt02b_batched_tokens opt02c_gpumem097
    opt03a_attn_flashinfer_mla opt03b_attn_flashmla
    opt03c_mla_prefill_flashinfer opt03d_kv_bf16 opt03e_hybrid_kv
    opt04a_moe_deepgemm_mega opt04b_moe_flashinfer_trtllm
    opt04c_moe_cutlass opt04d_moe_triton opt04e_moe_tail_fusion_off
    opt05a_a2a_nvlink_one_sided opt05b_a2a_nvlink_two_sided
    opt05c_a2a_deepep_v2 opt05d_a2a_deepep_low_latency
    opt05e_a2a_deepep_high_throughput opt05f_a2a_allgather_reducescatter
    opt06_eplb opt07_dbo
    opt08_dspark1 opt08_dspark3 opt08_dspark5 opt08_dspark9
    opt08_dspark_greedy opt08_dspark_disable_bs64
    opt09_flashinfer_sampler
    opt10a_cudagraph_full_piecewise opt10b_cudagraph_decode_only
    opt11a_model_runner_v2 opt11b_rust_frontend opt11c_v2_and_rust
    opt12_language_model_only
    opt13_no_flashinfer_allreduce
    ref_nospec
)

CONFIGS=("${ALL_CONFIGS[@]}")
if [[ "${1:-}" == "--only" ]]; then
    shift; CONFIGS=("$@")
fi

for cfg in "${CONFIGS[@]}"; do
    sweep="$OPT_SWEEP"
    [[ "$cfg" == "baseline" ]] && sweep="$BASELINE_SWEEP"
    # Resume: skip a config that already completed (run.sh writes results/<cfg>.csv
    # only after a successful sweep+aggregate, so it's a reliable completion marker).
    if [[ "${SKIP_EXISTING:-0}" == "1" && -f "$REPO_ROOT/results/$cfg.csv" ]]; then
        echo "==================== SKIP $cfg (results/$cfg.csv exists) ===================="
        continue
    fi
    # Pick the cap: explicit CONFIG_TIMEOUT override, else per-sweep.
    if [[ -n "$CONFIG_TIMEOUT" ]]; then to="$CONFIG_TIMEOUT"
    elif [[ "$sweep" == "full" ]]; then to="$FULL_TIMEOUT"
    else to="$SUBSET_TIMEOUT"; fi

    echo "==================== $cfg (sweep=$sweep, cap=${to}s) ===================="
    rc=0
    if [[ "$to" -gt 0 ]]; then
        timeout --signal=TERM --kill-after=60 "$to" \
            bash "$REPO_ROOT/run.sh" "$cfg" "$sweep" "$DATASET" || rc=$?
        if [[ "$rc" == "124" ]]; then
            echo "WARN: $cfg exceeded ${to}s cap — killed; skipping." >&2
        elif [[ "$rc" != "0" ]]; then
            echo "WARN: $cfg failed (rc=$rc); continuing to next config." >&2
        fi
    else
        bash "$REPO_ROOT/run.sh" "$cfg" "$sweep" "$DATASET" || { rc=$?; echo "WARN: $cfg failed (rc=$rc); continuing." >&2; }
    fi
    # Always leave a clean GPU for the next config.
    reap_gpu
done

echo "==================== combined CSV ===================="
python3 "$REPO_ROOT/aggregate.py" "$REPO_ROOT/results" \
    --baseline baseline --out "$REPO_ROOT/results/all.csv"
echo "Combined: results/all.csv"
# (Per-config results are already in W&B; each run.sh call synced its config.)
