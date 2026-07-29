#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# get_sharegpt.sh — fetch the ShareGPT V3 dataset used by the benchmark sweep.
#
# The harness benchmarks Kimi-K3 on ShareGPT (real multi-turn chat prompts)
# rather than the synthetic random dataset, because speculative decoding
# acceptance length is only representative on natural, chat-formatted text.
#
# Writes:  $SHAREGPT_PATH  (default <repo>/datasets/ShareGPT_V3_unfiltered_cleaned_split.json)
# Idempotent: exits immediately if the file is already present and non-trivial.
#
# Usage:
#   bash bench/get_sharegpt.sh
#   SHAREGPT_PATH=/workspace/datasets/sharegpt.json bash bench/get_sharegpt.sh
# ---------------------------------------------------------------------------
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SHAREGPT_PATH="${SHAREGPT_PATH:-$REPO_ROOT/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
SHAREGPT_REPO="${SHAREGPT_REPO:-anon8231489123/ShareGPT_Vicuna_unfiltered}"
SHAREGPT_FILE="${SHAREGPT_FILE:-ShareGPT_V3_unfiltered_cleaned_split.json}"
URL="${SHAREGPT_URL:-https://huggingface.co/datasets/$SHAREGPT_REPO/resolve/main/$SHAREGPT_FILE}"

# Already there? (>10 MB is a decent "not a truncated/error page" check.)
if [[ -f "$SHAREGPT_PATH" ]]; then
    sz=$(stat -c %s "$SHAREGPT_PATH" 2>/dev/null || stat -f %z "$SHAREGPT_PATH" 2>/dev/null || echo 0)
    if [[ "$sz" -gt 10000000 ]]; then
        echo "ShareGPT already present: $SHAREGPT_PATH ($((sz / 1024 / 1024)) MB)"
        exit 0
    fi
    echo "WARN: $SHAREGPT_PATH exists but is only $sz bytes — refetching."
fi

mkdir -p "$(dirname "$SHAREGPT_PATH")"
echo "Fetching ShareGPT V3 -> $SHAREGPT_PATH"

# Prefer the HF CLI (handles auth / mirrors / resume); fall back to curl.
if command -v hf >/dev/null 2>&1; then
    tmpdir="$(dirname "$SHAREGPT_PATH")/.hf_sharegpt"
    if hf download "$SHAREGPT_REPO" "$SHAREGPT_FILE" --repo-type dataset --local-dir "$tmpdir"; then
        mv -f "$tmpdir/$SHAREGPT_FILE" "$SHAREGPT_PATH" && rm -rf "$tmpdir"
    fi
fi
if [[ ! -s "$SHAREGPT_PATH" ]]; then
    curl -fL --retry 3 -o "$SHAREGPT_PATH" "$URL" || {
        echo "ERROR: could not download ShareGPT from $URL" >&2
        echo "       Pre-stage it manually and set SHAREGPT_PATH=/path/to/file.json" >&2
        exit 1
    }
fi

# Validate it really is the JSON array of conversations.
python3 - "$SHAREGPT_PATH" <<'PY' || exit 1
import json, sys
p = sys.argv[1]
with open(p, encoding="utf-8") as f:
    d = json.load(f)
if not isinstance(d, list) or not d:
    print(f"ERROR: {p} is not a non-empty JSON list", file=sys.stderr); sys.exit(1)
n = sum(1 for c in d if len(c.get("conversations", [])) >= 2)
print(f"OK: {len(d)} records, {n} usable (>=2 turns) -> {p}")
PY
