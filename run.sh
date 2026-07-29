#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run.sh - end-to-end runner for ONE configuration:
#   launch server -> InferenceX sweep -> tear down -> aggregate to CSV
#   -> sync everything (metrics + CSV + JSONs + logs) to W&B.
#
# Which lanes run is decided by the config's BENCH_MODE (see bench/bench.sh):
#   nonmtp -> random dataset, raw prompts
#   mtp    -> random dataset + chat template, AND ShareGPT + chat template
# Override with DATASETS="random" / DATASETS="sharegpt".
#
# Usage:
#   bash run.sh baseline full               # full sweep (conc 1..128)
#   bash run.sh opt05_linear_flashinfer     # subset sweep (conc 1,16,128) for trials
#   CONFIG_LABEL=spec_dspark_5 SPEC_METHOD=dspark SPEC_TOKENS=5 bash run.sh spec
#
# Positional args:
#   $1  config name = servers/<name>.sh   (required)
#   $2  sweep: full | subset              (default: subset)
#
# Durability: W&B is the off-box store for these on-demand cloud runs.
#   WANDB=0  disable W&B sync (else needs WANDB_API_KEY; see .env)
# ---------------------------------------------------------------------------
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Load secrets / settings (WANDB_API_KEY, WANDB_PROJECT, ...) if present.
# .env is git-ignored - never commit credentials. Strip CRs so a Windows-edited
# (CRLF) .env doesn't append '\r' to values (which silently corrupts the API key).
[[ -f "$REPO_ROOT/.env" ]] && { set -a; source <(tr -d '\r' < "$REPO_ROOT/.env"); set +a; }

NAME="${1:?usage: run.sh <config> [full|subset]}"
SWEEP="${2:-subset}"
export DATASETS="${DATASETS:-}"     # empty = derive from the config BENCH_MODE
NAME="${NAME%.sh}"; NAME="${NAME#servers/}"
SCRIPT="$REPO_ROOT/servers/$NAME.sh"
[[ -f "$SCRIPT" ]] || { echo "ERROR: no such config: $SCRIPT"; exit 1; }

# CONFIG_LABEL decouples the results label from the script name, so ONE
# parameterized script can produce many labelled runs. run_spec_sweep.sh uses it:
#   CONFIG_LABEL=spec_dspark_5 SPEC_METHOD=dspark SPEC_TOKENS=5 bash run.sh spec
# Every servers/*.sh reads CONFIG as ${CONFIG:-<own name>}, so the exported label
# wins and results land in results/<label>/.
LABEL="${CONFIG_LABEL:-$NAME}"
export CONFIG="$LABEL"

# --- Upfront W&B readiness check -------------------------------------------
# Warn NOW (before a possibly hours-long sweep) if results won't reach W&B,
# so you can fix .env / install wandb instead of discovering it at the end.
if [[ "${WANDB:-1}" != "0" ]]; then
    if [[ -z "${WANDB_API_KEY:-}" && ! -f "$HOME/.netrc" ]]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "!! WARNING: WANDB_API_KEY not set and no ~/.netrc login."
        echo "!!   Results will NOT sync to W&B. Put the key in .env (run.sh"
        echo "!!   loads it), 'export WANDB_API_KEY=...', or set WANDB=0 to silence."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    elif ! python3 -c "import wandb" 2>/dev/null; then
        echo "NOTE: wandb not importable yet - run.sh will pip-install it before sync."
    else
        echo "W&B ready: key loaded (len ${#WANDB_API_KEY}), project=${WANDB_PROJECT:-aiand-kimik3-mxfp4}."
    fi
fi

# Make sure the ShareGPT dataset is on disk BEFORE spending an hour loading 1.4 TB
# of weights, so a missing dataset fails in seconds rather than after the server
# is up. (bench.sh would fetch it too, but by then the cost is already sunk.)
if [[ "$DATASETS" != "random" ]]; then
    bash "$REPO_ROOT/bench/get_sharegpt.sh" || { echo "ERROR: ShareGPT unavailable"; exit 1; }
fi

echo "### RUN $NAME as '$LABEL' (sweep=$SWEEP datasets=${DATASETS:-auto}) ###"
RUN_BENCH=1 SWEEP="$SWEEP" DATASETS="$DATASETS" bash "$SCRIPT"

echo "### AGGREGATE $LABEL ###"
CSV="$REPO_ROOT/results/$LABEL.csv"
python3 "$REPO_ROOT/aggregate.py" "$REPO_ROOT/results/$LABEL" --config "$LABEL" --out "$CSV"
echo "CSV: results/$LABEL.csv"

# --- Durability: push results off this (ephemeral) box ---------------------
if [[ "${WANDB:-1}" != "0" ]]; then
    echo "### W&B SYNC $NAME ###"
    python3 -c "import wandb" 2>/dev/null || pip install -q wandb 2>/dev/null || true
    python3 "$REPO_ROOT/wandb_sync.py" --config "$LABEL" --sweep "$SWEEP"; rc=$?
    case "$rc" in
        0) echo "W&B: synced $LABEL OK" ;;
        3) echo "!! W&B: SKIPPED $LABEL (results NOT in W&B - see banner above)" ;;
        *) echo "!! W&B: FAILED $LABEL (rc=$rc - see error above; results NOT in W&B)" ;;
    esac
fi
