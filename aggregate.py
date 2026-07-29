#!/usr/bin/env python3
"""Aggregate benchmark result JSONs into a CSV.

Walks one or more result directories, reads every benchmark_serving result
JSON, and emits a tab/comma-separated table you can paste straight into the
tracking Google Sheet.

Tracked metrics (simplified set):
    Output throughput (tok/s), Out tok/s/GPU,
    TTFT mean / median / P90 (ms),
    TPOT mean / median / P90 (ms)

Identity (config, mode, ISL, OSL, conc) is parsed from the result filename
produced by bench/bench.sh, which has two forms:
    ShareGPT: <config>__<mode>_sharegpt_osl<OSL>_conc<CONC>.json
    Random:   <config>__<mode>_isl<ISL>_osl<OSL>_conc<CONC>.json
where <mode> is spec | nospec (Kimi-K3 uses DSpark speculative decoding; the
legacy mtp|nonmtp spelling is still accepted so GLM-era files parse too).

For ShareGPT the input length is not fixed by the harness — it comes from the
dataset — so the ISL column reads "sharegpt" and the real value is reported in
"Mean ISL tok" (derived from total_input_tokens / completed).

Usage:
    python3 aggregate.py results/                       # everything under results/
    python3 aggregate.py results/baseline --out baseline.csv
    python3 aggregate.py results/ --sep $'\t'           # TSV for direct paste
    python3 aggregate.py results/ --baseline baseline   # add "vs baseline" tput %
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

MODES = "spec|nospec|mtp|nonmtp"
FNAME_RE = re.compile(
    rf"^(?P<config>.+?)__(?P<mode>{MODES})_"
    r"(?:sharegpt|isl(?P<isl>\d+))_osl(?P<osl>\d+)_conc(?P<conc>\d+)$"
)

COLUMNS = [
    "Config", "Mode", "ISL", "OSL", "Conc",
    "Completed", "Mean ISL tok", "Output tok/s", "Total tok/s", "Out tok/s/GPU",
    "TTFT Mean", "TTFT Median", "TTFT P90",
    "TPOT Mean", "TPOT Median", "TPOT P90",
]


def _f(d, key):
    v = d.get(key)
    return round(float(v), 2) if isinstance(v, (int, float)) else ""


def parse_file(path: Path, num_gpus: int):
    stem = path.name[:-len(".json")] if path.name.endswith(".json") else path.stem
    m = FNAME_RE.match(stem)
    with open(path, encoding="utf-8") as f:
        d = json.load(f)

    # Identity: prefer filename; fall back to JSON fields where possible.
    if m:
        config, mode = m["config"], m["mode"]
        # No isl group -> ShareGPT, where input length is dataset-driven.
        isl = int(m["isl"]) if m["isl"] else "sharegpt"
        osl, conc = int(m["osl"]), int(m["conc"])
    else:
        config, mode = stem, ""
        isl = osl = ""
        conc = int(d.get("max_concurrency") or 0)

    out_tput = d.get("output_throughput")
    out_per_gpu = round(float(out_tput) / num_gpus, 2) if isinstance(out_tput, (int, float)) else ""

    # Real mean input length — the only way to know what ShareGPT actually served.
    completed = d.get("completed")
    total_in = d.get("total_input_tokens")
    mean_isl = (round(float(total_in) / completed, 1)
                if isinstance(total_in, (int, float)) and isinstance(completed, int) and completed
                else "")

    return {
        "Config": config, "Mode": mode, "ISL": isl, "OSL": osl, "Conc": conc,
        "Completed": completed if completed is not None else "",
        "Mean ISL tok": mean_isl,
        "Output tok/s": _f(d, "output_throughput"),
        "Total tok/s": _f(d, "total_token_throughput"),
        "Out tok/s/GPU": out_per_gpu,
        "TTFT Mean": _f(d, "mean_ttft_ms"),
        "TTFT Median": _f(d, "median_ttft_ms"),
        "TTFT P90": _f(d, "p90_ttft_ms"),
        "TPOT Mean": _f(d, "mean_tpot_ms"),
        "TPOT Median": _f(d, "median_tpot_ms"),
        "TPOT P90": _f(d, "p90_tpot_ms"),
    }


def _isl_sort_key(v):
    """Sort numeric ISLs numerically; put dataset-driven ('sharegpt') first."""
    return (0, 0) if not isinstance(v, int) else (1, v)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="result directory/directories to scan")
    ap.add_argument("--out", help="write CSV here (default: stdout)")
    ap.add_argument("--sep", default=",", help="field separator (use $'\\t' for TSV)")
    ap.add_argument("--num-gpus", type=int, default=8, help="GPUs per replica (default 8)")
    ap.add_argument("--config", help="only include rows whose Config == this")
    ap.add_argument("--baseline",
                    help="config name to use as the throughput reference; "
                         "adds a 'vs Baseline' column (per ISL/OSL/Conc cell)")
    args = ap.parse_args()

    rows = []
    for d in args.dirs:
        for path in sorted(Path(d).rglob("*.json")):
            if ".pytorch." in path.name:   # skip the pytorch-format sidecar
                continue
            try:
                row = parse_file(path, args.num_gpus)
            except Exception as exc:  # noqa: BLE001
                print(f"WARN: skipping {path}: {exc}", file=sys.stderr)
                continue
            if args.config and row["Config"] != args.config:
                continue
            rows.append(row)

    if not rows:
        print("No result JSONs found.", file=sys.stderr)
        sys.exit(1)

    # Stable, human-friendly ordering.
    rows.sort(key=lambda r: (str(r["Config"]), str(r["Mode"]),
                             _isl_sort_key(r["ISL"]),
                             int(r["OSL"] or 0), int(r["Conc"] or 0)))

    columns = list(COLUMNS)
    if args.baseline:
        base = {}
        for r in rows:
            if r["Config"] == args.baseline:
                base[(r["Mode"], r["ISL"], r["OSL"], r["Conc"])] = r["Output tok/s"]
        columns.append("vs Baseline")
        for r in rows:
            ref = base.get((r["Mode"], r["ISL"], r["OSL"], r["Conc"]))
            cur = r["Output tok/s"]
            if isinstance(ref, (int, float)) and ref and isinstance(cur, (int, float)):
                r["vs Baseline"] = f"{(cur / ref - 1) * 100:+.1f}%"
            else:
                r["vs Baseline"] = ""

    out = open(args.out, "w", newline="", encoding="utf-8") if args.out else sys.stdout
    try:
        w = csv.DictWriter(out, fieldnames=columns, delimiter=args.sep)
        w.writeheader()
        w.writerows(rows)
    finally:
        if args.out:
            out.close()
            print(f"Wrote {len(rows)} rows -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
