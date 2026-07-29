#!/usr/bin/env python3
"""Run InferenceX's benchmark_serving.py against a ShareGPT dataset.

WHY THIS EXISTS
---------------
The team's method is InferenceX's, and it must stay InferenceX's so numbers
remain comparable:

    non-MTP  -> random dataset, raw prompts
    MTP      -> random dataset, prompts wrapped in the chat template

For Kimi-K3 we additionally want the MTP configs measured on **ShareGPT wrapped
in the chat template**, because speculative-decoding acceptance length on
synthetic random tokens is an artifact (random input drives degenerate, often
highly repetitive output, so acceptance can be biased in either direction).

The vendored InferenceX client is random-only in this revision — its
``--dataset-name`` is literally ``choices=["random"]`` and anything else raises
"Unknown dataset". Rather than swap in a different benchmark client (which would
change the metrics code, the warmup semantics and the result schema all at once),
this shim keeps **InferenceX's client entirely intact** and replaces only the
prompt source:

  1. imports ``benchmark_serving`` as a module,
  2. rebuilds InferenceX's OWN argument parser by exec'ing the parser block from
     its ``__main__`` section, so every flag, default and semantic is upstream's
     rather than a copy that can drift,
  3. monkey-patches ``sample_random_requests`` with a ShareGPT sampler that calls
     upstream's ``_apply_chat_template`` for the wrapping,
  4. hands the namespace to upstream's ``main()``.

Everything downstream — request dispatch, warmup, metric calculation, percentile
handling, result JSON layout — is byte-identical to a normal InferenceX run. The
only difference between a random cell and a ShareGPT cell is which prompts went in.

CAVEAT (read before comparing lanes)
------------------------------------
The chat wrapping is upstream's ``tokenizer.apply_chat_template`` (Jinja), used
for BOTH lanes. Kimi-K3 also has a non-Jinja rendering path
(``--tokenizer-mode kimi_k3``, "Kimi K3's Python XTML encoding"); upstream has a
precedent for this (``--dsv4`` + ``encoding_dsv4.py``) but no K3 equivalent. If
the Jinja template turns out to be wrong for K3, the error is *systematic across
both lanes*, so config-to-config comparisons stay valid while absolute acceptance
length would not. ``TOKENIZER_MODE`` (env, honoured by bench.sh) is the escape
hatch; check the smoke test output first.

Usage — same CLI as InferenceX's benchmark_serving.py, plus:
    --dataset-name sharegpt --dataset-path <ShareGPT_V3_...json>
    --sharegpt-output-len N        fixed output length (recommended with --ignore-eos)
    --sharegpt-min-input-len N     drop prompts shorter than this (default 4)
    --sharegpt-max-input-len N     drop prompts longer than this (default 4096)
"""
import argparse
import json
import os
import random
import re
import sys
import textwrap
from pathlib import Path

import numpy as np


def _locate_inferencex() -> Path:
    """Directory holding InferenceX's benchmark_serving.py."""
    env = os.environ.get("INFERENCEX_DIR")
    root = Path(env) if env else Path(__file__).resolve().parent.parent / "third_party" / "InferenceX"
    d = root / "utils" / "bench_serving"
    if not (d / "benchmark_serving.py").is_file():
        sys.exit(f"ERROR: benchmark_serving.py not found under {d}\n"
                 f"       Run: git submodule update --init --recursive")
    return d


def _build_upstream_parser(bs):
    """Rebuild InferenceX's own argparse parser.

    Its parser is constructed inline inside ``if __name__ == "__main__":``, so it
    cannot simply be imported. We take that block's source, drop the two driver
    statements at the end, and exec it with the module's globals — giving the real
    parser with the real defaults, which stays in sync with the submodule instead
    of being duplicated here.
    """
    src = Path(bs.__file__).read_text(encoding="utf-8")
    marker = 'if __name__ == "__main__":'
    if marker not in src:
        sys.exit("ERROR: could not find the __main__ block in InferenceX's "
                 "benchmark_serving.py — its layout changed; update this shim.")
    block = textwrap.dedent(src[src.index(marker) + len(marker):])
    block = re.sub(r"(?m)^args\s*=\s*parser\.parse_args\(\).*$", "", block)
    block = re.sub(r"(?m)^main\(args\).*$", "", block)
    g = dict(bs.__dict__)
    exec(compile(block, "<inferencex-parser>", "exec"), g)  # noqa: S102
    parser = g.get("parser")
    if parser is None:
        sys.exit("ERROR: exec'ing InferenceX's __main__ block produced no "
                 "'parser' — its layout changed; update this shim.")
    return parser


def _widen_parser(parser):
    """Allow --dataset-name sharegpt and add our ShareGPT filtering knobs."""
    for action in parser._actions:                       # noqa: SLF001
        if action.dest == "dataset_name" and action.choices:
            action.choices = sorted(set(action.choices) | {"sharegpt"})
        # Upstream restricts --tokenizer-mode to auto/slow/mistral/custom, which
        # excludes Kimi-K3's XTML mode. Widen it so TOKENIZER_MODE can be used.
        if action.dest == "tokenizer_mode" and action.choices:
            action.choices = sorted(set(action.choices) | {"kimi_k3", "hf"})
    grp = parser.add_argument_group("sharegpt filtering (this shim)")
    grp.add_argument("--sharegpt-min-input-len", type=int, default=4,
                     help="Drop prompts shorter than this many tokens (default 4).")
    grp.add_argument("--sharegpt-max-input-len", type=int, default=4096,
                     help="Drop prompts longer than this many tokens (default 4096). "
                          "Keep it under the server's --max-model-len minus the "
                          "output length.")
    return parser


def _sample_sharegpt(*, dataset_path, num_prompts, tokenizer, output_len,
                     range_ratio, use_chat_template, dsv4,
                     min_input_len, max_input_len, seed, apply_chat_template):
    """Return InferenceX's request tuples: (prompt, prompt_len, output_len, mm).

    Mirrors upstream's random sampler in the ways that matter for comparability:
      * the chat template is applied through upstream's own helper,
      * prompt_len is measured AFTER wrapping, exactly as upstream does,
      * output length is jittered with the same --random-range-ratio, so decode
        lengths vary the same way they do on the random lane.
    """
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    # Keep conversations with at least one human turn + one assistant turn.
    convs = [d["conversations"] for d in dataset
             if len(d.get("conversations", [])) >= 2]
    if not convs:
        sys.exit(f"ERROR: no usable conversations in {dataset_path}")

    rng = random.Random(seed)
    rng.shuffle(convs)

    lower = max(1, int(output_len * range_ratio)) if output_len else 0
    np_rng = np.random.RandomState(seed)

    requests, skipped = [], 0
    # One pass is normally plenty (ShareGPT V3 has ~90k usable conversations);
    # cycle if a very large --num-prompts asks for more than survive filtering.
    idx = 0
    while len(requests) < num_prompts:
        if idx >= len(convs):
            if not requests:
                sys.exit(f"ERROR: every ShareGPT prompt was filtered out "
                         f"(min={min_input_len} max={max_input_len}); widen the range")
            idx = 0            # cycle: reuse prompts rather than under-fill
        c = convs[idx]
        idx += 1

        prompt = c[0].get("value") or ""
        completion = c[1].get("value") or ""
        if not prompt:
            skipped += 1
            continue
        if use_chat_template:
            prompt = apply_chat_template(prompt, tokenizer, dsv4)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        if prompt_len < min_input_len or prompt_len > max_input_len:
            skipped += 1
            continue

        if output_len:
            olen = int(np_rng.randint(lower, output_len + 1))
        else:
            olen = len(tokenizer.encode(completion, add_special_tokens=False))
            if olen < 4:
                skipped += 1
                continue
        requests.append((prompt, prompt_len, olen, None))

    header = f'{"-" * 17}  ShareGPT Input/Output Length Statistics  {"-" * 17}'
    print(header)
    print(f' dataset    : {dataset_path}')
    print(f' prompts    : {len(requests)} kept, {skipped} skipped by filters '
          f'(min={min_input_len} max={max_input_len} chat_template={use_chat_template})')
    print(f' input_lens : min={min(r[1] for r in requests):<5d} '
          f'max={max(r[1] for r in requests):<5d} '
          f'mean={np.mean([r[1] for r in requests]):<8.2f}')
    print(f' output_lens: min={min(r[2] for r in requests):<5d} '
          f'max={max(r[2] for r in requests):<5d} '
          f'mean={np.mean([r[2] for r in requests]):<8.2f}')
    print("-" * len(header), "\n")
    return requests


def main():
    bench_dir = _locate_inferencex()
    sys.path.insert(0, str(bench_dir))
    import benchmark_serving as bs  # noqa: E402  (path must be set first)

    parser = _widen_parser(_build_upstream_parser(bs))
    args = parser.parse_args()

    if args.dataset_name != "sharegpt":
        # Nothing to patch — behave exactly like calling upstream directly.
        return bs.main(args)

    if not args.dataset_path:
        sys.exit("ERROR: --dataset-path is required with --dataset-name sharegpt")
    if not os.path.isfile(args.dataset_path):
        sys.exit(f"ERROR: ShareGPT dataset not found: {args.dataset_path}\n"
                 f"       Run: bash bench/get_sharegpt.sh")

    sharegpt_args = dict(
        dataset_path=args.dataset_path,
        output_len=args.sharegpt_output_len,
        min_input_len=args.sharegpt_min_input_len,
        max_input_len=args.sharegpt_max_input_len,
        apply_chat_template=bs._apply_chat_template,   # noqa: SLF001 - deliberate reuse
    )

    def _patched_sampler(**kw):
        """Stand in for upstream's random sampler.

        main() passes the tokenizer, use_chat_template, dsv4, num_prompts and
        range_ratio it resolved itself; we take those and ignore the
        random-specific geometry (input_len / prefix_len / worker count).
        """
        return _sample_sharegpt(
            num_prompts=kw["num_prompts"],
            tokenizer=kw["tokenizer"],
            range_ratio=kw.get("range_ratio", 1.0),
            use_chat_template=kw.get("use_chat_template", False),
            dsv4=kw.get("dsv4", False),
            seed=args.seed,
            **sharegpt_args,
        )

    bs.sample_random_requests = _patched_sampler
    # main() dispatches on dataset_name; "random" routes to the patched sampler.
    args.dataset_name = "random"
    return bs.main(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
