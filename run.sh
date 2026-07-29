#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run.sh - end-to-end runner for ONE configuration:
#   launch server -> ShareGPT sweep -> tear down -> aggregate to CSV
#   -> sync everything (metrics + CSV + JSONs + logs) to W&B.
#
# Usage:
#   bash run.sh baseline                    # full sweep (OSL 1024 + 256, conc 1..128)
#   bash run.sh opt04a_moe_deepgemm_mega    # subset sweep (conc 1,16,128) for trials
#   bash run.sh baseline full random        # cross-check on the random dataset
#
# Positional args:
#   $1  config name = servers/<name>.sh   (required)
#   $2  sweep: full | subset              (default: subset)
#   $3  dataset: sharegpt | random        (default: sharegpt)
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

NAME="${1:?usage: run.sh <config> [full|subset] [sharegpt|random]}"
SWEEP="${2:-subset}"
export DATASET="${3:-${DATASET:-sharegpt}}"
NAME="${NAME%.sh}"; NAME="${NAME#servers/}"
SCRIPT="$REPO_ROOT/servers/$NAME.sh"
[[ -f "$SCRIPT" ]] || { echo "ERROR: no such config: $SCRIPT"; exit 1; }

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
if [[ "$DATASET" == "sharegpt" ]]; then
    bash "$REPO_ROOT/bench/get_sharegpt.sh" || { echo "ERROR: ShareGPT unavailable"; exit 1; }
fi

echo "### RUN $NAME (sweep=$SWEEP dataset=$DATASET) ###"
RUN_BENCH=1 SWEEP="$SWEEP" DATASET="$DATASET" bash "$SCRIPT"

echo "### AGGREGATE $NAME ###"
CSV="$REPO_ROOT/results/$NAME.csv"
python3 "$REPO_ROOT/aggregate.py" "$REPO_ROOT/results/$NAME" --config "$NAME" --out "$CSV"
echo "CSV: results/$NAME.csv"

# --- Durability: push results off this (ephemeral) box ---------------------
if [[ "${WANDB:-1}" != "0" ]]; then
    echo "### W&B SYNC $NAME ###"
    python3 -c "import wandb" 2>/dev/null || pip install -q wandb 2>/dev/null || true
    python3 "$REPO_ROOT/wandb_sync.py" --config "$NAME" --sweep "$SWEEP"; rc=$?
    case "$rc" in
        0) echo "W&B: synced $NAME OK" ;;
        3) echo "!! W&B: SKIPPED $NAME (results NOT in W&B - see banner above)" ;;
        *) echo "!! W&B: FAILED $NAME (rc=$rc - see error above; results NOT in W&B)" ;;
    esac
fi
