#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_spec_sweep.sh — sweep the speculative-decoding token count.
#
# The number of speculative tokens dominates performance on this model, and the
# optimum is not knowable in advance: more draft tokens buy more accepted tokens
# per verify step at low concurrency, but each step costs proportionally more
# compute and VRAM, so the optimum moves DOWN as batch size grows. This script
# measures the whole curve instead of guessing a point.
#
# It mirrors InferenceX's own Kimi-K3 acceptance-length collector
# (benchmarks/single_node/speedbench/kimik3_fp4_b300_vllm.sh, MTP_LIST="1..8"):
# one server per token count, acceptance read from /metrics. Difference: we also
# capture throughput/TTFT/TPOT per concurrency, and we sweep BOTH mechanisms.
#
# Usage:
#   bash run_spec_sweep.sh                       # dspark 1..8 + kimi_k3_mtp 1..8
#   SPEC_METHODS=dspark bash run_spec_sweep.sh    # one mechanism only
#   SPEC_TOKENS_LIST="1 3 5 7" bash run_spec_sweep.sh
#   SWEEP=full bash run_spec_sweep.sh             # full concurrency grid (expensive)
#   SKIP_EXISTING=1 bash run_spec_sweep.sh        # resume
#   SPEC_DRAFT_METHODS="probabilistic greedy" SPEC_TOKENS_LIST=3 \
#     SPEC_METHODS=dspark bash run_spec_sweep.sh  # draft-sampling A/B at the peak
#
# Env:
#   SPEC_METHODS      default "dspark kimi_k3_mtp"   (add/remove mechanisms)
#   SPEC_TOKENS_LIST  default "1 2 3 4 5 6 7 8"
#   SPEC_DRAFT_METHODS  default "" — optional EXTRA axis, dspark only:
#                     space-separated draft_sample_method values, e.g.
#                     "probabilistic greedy". Empty means "use the recipe default
#                     (probabilistic)" and adds no launches. Greedy drafting is
#                     cheaper per draft token and often raises acceptance on
#                     low-temperature traffic, so it is worth one pass once the
#                     token count is settled.
#   INCLUDE_NONE      0 (default) — the no-drafting floor already has its own
#                     config, servers/ref_nonmtp.sh; set 1 to fold it in here
#   SWEEP             subset (default) | full
#   MAX_NUM_SEQS      passed through to servers/spec.sh (default 32)
#
# !! COST: every point is a separate server launch, and Kimi-K3 cold-loads ~1.4 TB.
# The default grid is 2 mechanisms x 8 token counts = 16 launches. Start
# with SPEC_TOKENS_LIST="1 3 5 7" (9 launches) to find the shape, then refine
# around the peak. Set LOAD_FORMAT/SAFETENSORS_LOAD_STRATEGY to cut load time.
# ---------------------------------------------------------------------------
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
[[ -f "$REPO_ROOT/.env" ]] && { set -a; source <(tr -d '\r' < "$REPO_ROOT/.env"); set +a; }

SPEC_METHODS="${SPEC_METHODS:-dspark kimi_k3_mtp}"
SPEC_TOKENS_LIST="${SPEC_TOKENS_LIST:-1 2 3 4 5 6 7 8}"
INCLUDE_NONE="${INCLUDE_NONE:-0}"
SWEEP="${SWEEP:-subset}"

for m in $SPEC_METHODS; do
    case "$m" in dspark|kimi_k3_mtp) ;; *) echo "ERROR: bad SPEC_METHODS entry '$m'" >&2; exit 1 ;; esac
done
for n in $SPEC_TOKENS_LIST; do
    [[ "$n" =~ ^[0-9]+$ ]] || { echo "ERROR: bad SPEC_TOKENS_LIST entry '$n'" >&2; exit 1; }
done
SPEC_DRAFT_METHODS="${SPEC_DRAFT_METHODS:-}"
for d in $SPEC_DRAFT_METHODS; do
    case "$d" in probabilistic|greedy) ;;
        *) echo "ERROR: bad SPEC_DRAFT_METHODS entry '$d' (want probabilistic|greedy)" >&2; exit 1 ;;
    esac
done

# Build the run list: (label, method, tokens, draft_sample_method)
LABELS=(); METHODS=(); TOKENS=(); DRAFTS=()
if [[ "$INCLUDE_NONE" == "1" ]]; then
    LABELS+=("spec_none"); METHODS+=("none"); TOKENS+=("0"); DRAFTS+=("")
fi
for m in $SPEC_METHODS; do
    for n in $SPEC_TOKENS_LIST; do
        if [[ -z "$SPEC_DRAFT_METHODS" || "$m" != "dspark" ]]; then
            # No extra axis (or kimi_k3_mtp, which has no draft_sample_method).
            LABELS+=("spec_${m}_${n}"); METHODS+=("$m"); TOKENS+=("$n"); DRAFTS+=("")
        else
            for d in $SPEC_DRAFT_METHODS; do
                LABELS+=("spec_${m}_${n}_${d}"); METHODS+=("$m"); TOKENS+=("$n"); DRAFTS+=("$d")
            done
        fi
    done
done

echo "=============================================================="
echo "  Speculative-decoding sweep: ${#LABELS[@]} server launches"
echo "  methods=[$SPEC_METHODS]  tokens=[$SPEC_TOKENS_LIST]  none=$INCLUDE_NONE"
echo "  sweep=$SWEEP"
echo "  labels: ${LABELS[*]}"
echo "=============================================================="

for i in "${!LABELS[@]}"; do
    label="${LABELS[$i]}"; method="${METHODS[$i]}"; ntok="${TOKENS[$i]}"; draft="${DRAFTS[$i]}"
    if [[ "${SKIP_EXISTING:-0}" == "1" && -f "$REPO_ROOT/results/$label.csv" ]]; then
        echo "==================== SKIP $label (results/$label.csv exists) ===================="
        continue
    fi
    echo "==================== $label (method=$method tokens=$ntok${draft:+ draft=$draft}) ===================="
    CONFIG_LABEL="$label" SPEC_METHOD="$method" SPEC_TOKENS="$ntok" \
    ${draft:+DRAFT_SAMPLE_METHOD="$draft"} \
        bash "$REPO_ROOT/run.sh" spec "$SWEEP" \
        || echo "WARN: $label failed; continuing to the next point." >&2
    # Reuse run_all.sh's GPU reaper so a stuck server never OOMs the next launch.
    pkill -TERM -f 'vllm serve' 2>/dev/null || true
    pkill -TERM VLLM 2>/dev/null || true
    sleep 5
    waited=0
    while [[ $waited -lt 300 ]]; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)
        [[ -z "$used" || "$used" -lt 2000 ]] && break
        pkill -9 -f 'vllm' 2>/dev/null || true; sleep 5; waited=$((waited + 5))
    done
done

echo "==================== combined spec CSV ===================="
dirs=()
for label in "${LABELS[@]}"; do
    [[ -d "$REPO_ROOT/results/$label" ]] && dirs+=("$REPO_ROOT/results/$label")
done
if [[ ${#dirs[@]} -eq 0 ]]; then
    echo "ERROR: no spec results found — every launch failed?" >&2; exit 1
fi
python3 "$REPO_ROOT/aggregate.py" "${dirs[@]}" --out "$REPO_ROOT/results/spec_sweep.csv"
echo "Combined: results/spec_sweep.csv"

echo "==================== acceptance-length curve ===================="
# The point of the sweep: acceptance length vs token count, per mechanism and
# per dataset lane. This is what tells you WHERE the optimum is, and the ShareGPT
# column is the trustworthy one (random-token acceptance is an artifact).
python3 - "$REPO_ROOT/results/spec_sweep.csv" <<'PY'
import csv, re, sys, collections
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
def num(x):
    try: return float(x)
    except (TypeError, ValueError): return None
# (method, tokens) -> {dataset: {conc: (accept, tput)}}
data = collections.defaultdict(dict)
for r in rows:
    m = re.match(r"spec_(none|dspark|kimi_k3_mtp)(?:_(\d+))?(?:_(probabilistic|greedy))?$",
                 r["Config"] or "")
    if not m: continue
    meth = m.group(1) + (f"/{m.group(3)}" if m.group(3) else "")
    key = (meth, int(m.group(2)) if m.group(2) else 0)
    data[key][(r["Dataset"], r["Conc"])] = (num(r.get("Accept len")), num(r.get("Output tok/s")))

if not data:
    print("  (no spec_* rows — nothing to plot)"); sys.exit(0)
concs = sorted({c for v in data.values() for (_, c) in v}, key=lambda x: int(x))
for ds in ("sharegpt", "random"):
    present = [(k, v) for k, v in data.items() if any(d == ds for (d, _) in v)]
    if not present: continue
    print(f"\n  dataset={ds}   (AL = mean accepted tokens per draft step; tput = output tok/s)")
    hdr = f"    {'method':26} {'tok':>3} " + " ".join(f"{'c'+c:>18}" for c in concs)
    print(hdr); print("    " + "-" * (len(hdr) - 4))
    for (meth, ntok), v in sorted(present, key=lambda kv: (kv[0][0], kv[0][1])):
        cells = []
        for c in concs:
            al, tp = v.get((ds, c), (None, None))
            cells.append(f"{'AL '+format(al,'.2f') if al else 'AL   - ':>9}/{format(tp,'.0f') if tp else '-':>8}")
        print(f"    {meth:26} {ntok:>3} " + " ".join(f"{x:>18}" for x in cells))
print("\n  Read it as: pick the token count whose ShareGPT throughput peaks at the")
print("  concurrency you care about. A high AL with flat throughput means drafting")
print("  is accurate but the verify step has become the bottleneck.")
PY
