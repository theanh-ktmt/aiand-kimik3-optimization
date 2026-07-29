# (AI&) Kimi-K3 MXFP4 Optimizations

Benchmark harness to find the best **vLLM 0.27.x** server configuration for
**`moonshotai/Kimi-K3`** (2.8T MXFP4 MoE) on **8x NVIDIA B300**, maximizing
throughput while keeping TTFT / TPOT acceptable.

Base config follows the published recipe
(<https://recipes.vllm.ai/moonshotai/Kimi-K3>, source
`vllm-project/recipes` → `models/moonshotai/Kimi-K3.yaml`), with every flag name
and enum value **verified against `vllm serve --help=all` on the target image**
`vllm/vllm-openai:kimi-k3` — which corrected three things the recipe alone got
wrong (see [`OPTIMIZATIONS.md`](OPTIMIZATIONS.md#corrections-the-help-log-forced-on-the-first-draft)).

Method is [InferenceX](https://github.com/SemiAnalysisAI/InferenceX)'s (vendored as
a git submodule under `third_party/InferenceX`), unchanged: saturated server
(`--request-rate inf`), `--ignore-eos`, prefix caching **off**, warmup before every
cell, swept across concurrency.

InferenceX also already ships two Kimi-K3 B300 reference scripts
(`benchmarks/single_node/{speedbench,agentic}/kimik3_fp4_b300_vllm.sh`). Their serve
profile is treated here as a third source alongside the recipe, and it overrode the
recipe on several points — the full diff is in
[`OPTIMIZATIONS.md`](OPTIMIZATIONS.md#third-source-inferencex-already-ships-kimi-k3-b300-scripts).

* [`OPTIMIZATIONS.md`](OPTIMIZATIONS.md) — the optimization space, what made the
  list, and what was cut.
* [`SERVERS.md`](SERVERS.md) — one line per config, plus verified flag/enum notes.
* **Open questions** at the bottom of this file — three things only a run on the box
  can settle.

## The model, in one table

| | |
|---|---|
| Params | 2.8T total / 104B active (16 of 896 routed experts + 2 shared) |
| Layers | 93 = 1 dense + **69 KDA** + **24 Gated MLA** (hybrid attention) |
| Quant | **MXFP4 weights / MXFP8 activations** (QAT) → ~1.4 TB of weights |
| Context / vocab | 1,048,576 tokens / 160K |
| Spec decoding | **DSpark** (`Inferact/Kimi-K3-DSpark`) **or `kimi_k3_mtp`** (in-model head) |
| Modality | native multimodal (MoonViT-V2, 401M) |
| Min hardware | recipe `vram_minimum_gb: 1680`; 8x B300 = 2304 GiB → fits one node |

## Benchmark method

### Two dataset lanes, decided by the config's bench mode

| BENCH_MODE | lanes |
|---|---|
| `nonmtp` | random dataset, **raw** prompts |
| `mtp` | random dataset **+ chat template**, *and additionally* ShareGPT **+ chat template** |

The random lanes are the InferenceX method verbatim — that is what keeps the
numbers comparable to the team's existing results.

The extra **ShareGPT lane exists because acceptance length on synthetic random
tokens is an artifact**: random input drives degenerate, often highly repetitive
output, which a draft model predicts unusually well, so a random-dataset
measurement can be biased in *either* direction. ShareGPT gives
workload-representative acceptance, which is what makes the spec-decoding
comparison mean anything. The two lanes are separate rows in the CSV (`Dataset`
column); `aggregate.py` only ever compares like with like.

Both lanes run through the **same client**. `bench/sharegpt_client.py` is a thin
shim that imports InferenceX's `benchmark_serving`, rebuilds *its own* argument
parser, and swaps only the prompt source — request dispatch, warmup, metrics,
percentiles and result-JSON layout are upstream's. With `--dataset-name random` it
forwards to upstream untouched. Read that file's docstring for the caveat about
Jinja vs K3's XTML chat rendering.

### Invariants

- **Prefix caching is ALWAYS off** (`--no-enable-prefix-caching`) — otherwise
  repeated warmup / multi-turn prefixes are served from cache, prefill is skipped,
  and throughput is inflated. This is a hard invariant with no override: setting
  `PREFIX_CACHING` to anything non-zero makes the launch fail rather than silently
  produce non-comparable numbers.
  Both the recipe's Blackwell block **and** InferenceX's own K3 scripts enable it,
  so our absolute numbers are a **floor** versus theirs — say so whenever the two
  are shown together. Config-to-config comparisons inside this repo are unaffected,
  because the setting is identical everywhere.
- `--request-rate inf`, `--ignore-eos`, warmup = 2× concurrency,
  `--num-prompts` = 10× concurrency, `--random-range-ratio 0.8` (applied to
  ShareGPT output lengths too, so decode length varies the same way).
- Concurrency sweep: `1 2 4 8 16 32 64 128` (FULL) or `1 16 128` (SUBSET — 128 also
  exercises the batch-gated spec-decode disable).
- Scenarios: FULL = random `1k/1k` + `8k/1k`, ShareGPT `OSL 1024` + `256`;
  SUBSET = random `1k/1k`, ShareGPT `OSL 1024`.
- ShareGPT input length comes from the dataset (prompts filtered to ≤ 4096 tokens
  by default), so the CSV reports the real `Mean ISL tok` per cell.

### Speculative-decoding acceptance is measured, not inferred

`bench.sh` scrapes vLLM's `vllm:spec_decode_*` counters from `/metrics` before and
after each cell and writes `results/<config>/<cell>.accept.json` with the mean
acceptance length and draft acceptance rate. `aggregate.py` surfaces it as the
**`Accept len`** column. This is read-only and changes no measurement; disable with
`ACCEPT_METRICS=0`.

It is what turns "MTP helped by 12%" into "MTP accepted 3.4 of 7 draft tokens on
real chat traffic and 5.9 on random tokens" — the distinction the whole ShareGPT
lane exists to expose.

The formula (`AL = 1 + accepted/drafts`) and the counters are exactly those used by
InferenceX's own K3 acceptance-length collector, so the numbers are directly
comparable to their golden AL matrix.

Tracked metrics: **Output throughput**, **TTFT** (mean/median/P90), **TPOT**
(mean/median/P90), **Mean ISL tok**, **Accept len**.

## Layout

```
common.sh            shared config + mandatory flags + K3 recipe base block +
                     dspark_config() / kimi_k3_mtp_config() + server lifecycle
servers/*.sh         one launch script per configuration (baseline + 19 opts +
                     ref_nonmtp + 2 final placeholders = 23)
bench/bench.sh       the sweep: random lane (+ ShareGPT lane for mtp configs)
bench/sharegpt_client.py  InferenceX client with a ShareGPT prompt source
bench/get_sharegpt.sh     fetch the ShareGPT V3 dataset
run.sh               one config end-to-end: launch -> sweep -> teardown -> CSV
run_all.sh           the whole campaign: baseline + every opt SUBSET -> all.csv
run_final.sh         full bench for baseline + final1 + final2 -> final_full.csv
aggregate.py         result JSONs (+ accept sidecars) -> CSV (sheet-pasteable)
eval/quality_check.sh  MMLU-Pro (+ optional MMMU-Pro): baseline vs recommended
eval/parse_mmlu.py   lm-eval results -> accuracy comparison table + Pass?
preflight.sh         validate the box/build BEFORE a long campaign
results/<config>/    per-config outputs: *.json, *.accept.json, server.log
                     (with launch command), bench.log, gpu.csv  (git-ignored)
third_party/InferenceX  benchmark engine (submodule)
OPTIMIZATIONS.md     the optimization space, with rationale per knob
SERVERS.md           index of every server script and what it changes
```

## Setup

```bash
git clone --recurse-submodules <this-repo>
cd aiand-kimik3-optimization
# or, if already cloned:
git submodule update --init --recursive

cp .env.example .env          # put WANDB_API_KEY in it (git-ignored)
bash bench/get_sharegpt.sh    # -> datasets/ShareGPT_V3_unfiltered_cleaned_split.json
bash preflight.sh             # exit 0 = good to go
```

Run inside the **`vllm/vllm-openai:kimi-k3`** container on a B300 node. Optional:
pre-stage weights and `export MODEL_PATH=/path/to/Kimi-K3` to skip the ~1.4 TB HF
download. DSpark configs also need `Inferact/Kimi-K3-DSpark` (vLLM fetches it into
`--download-dir`, or set `DRAFT_MODEL_PATH`); `opt13_mtp_kimik3` and `ref_nonmtp`
do not.

`preflight.sh` checks, without loading the model: tools, vLLM ≥ 0.27.0, GPU count
*and aggregate VRAM vs the 1680 GiB minimum*, disk space for 1.4 TB, ShareGPT
presence, that the ShareGPT shim can still rebuild InferenceX's parser, bash/python
syntax, **every `--flag` via `vllm serve --help=<flag>`** (a definitive per-flag
answer, not a substring grep), the exact enum values passed, the
`AttentionBackendEnum` / `MambaBackendEnum` / `CUDAGraphMode` names, and that
`--spec-method` really offers both `dspark` and `kimi_k3_mtp`.

## Usage

### One config, end-to-end

```bash
bash run.sh baseline full                # full sweep, both lanes
bash run.sh opt05_linear_flashinfer      # subset sweep (default) for screening
DATASETS=sharegpt bash run.sh opt13_mtp_kimik3 full   # one lane only
```

### Manual (launch + benchmark in separate shells)

Preferred when screening several variants of one server config, because a cold
1.4 TB load is expensive — see the cost warning in `run_all.sh`.

```bash
# shell A - launch and keep the server up (command is echoed + logged)
bash servers/opt05_linear_flashinfer.sh

# shell B - benchmark the running server
CONFIG=opt05_linear_flashinfer bash bench/bench.sh \
    --config opt05_linear_flashinfer --mode mtp --sweep subset
```

### Whole campaign

```bash
bash run_all.sh                    # baseline + every opt, SUBSET each
python3 aggregate.py results --baseline baseline --out results/all.csv
```

`run_all.sh` caps each config's wall clock (2 h subset / 6 h full — the 1.4 TB load
is inside the cap), reaps stray vLLM processes between configs, and continues past
failures. `SKIP_EXISTING=1` resumes.

Once validated on the box, cut total campaign time with
`LOAD_FORMAT=instanttensor SAFETENSORS_LOAD_STRATEGY=eager bash run_all.sh`.

### Final configs + full benchmark + quality check

```bash
bash run_final.sh                                       # -> results/final_full.csv
bash eval/quality_check.sh baseline final1              # MMLU-Pro
RUN_MMMU=1 bash eval/quality_check.sh baseline final1   # + vision (MMMU-Pro)
```

Before each sweep a **smoke test** sends a few chat requests and prints
prompt + response, so you can confirm the server answers sensibly first.
`SMOKE_TEST=0` disables it; `SMOKE_TEST_STRICT=1` aborts on total failure;
`SMOKE_TEST_MM=1` adds an image request (not for `opt18_language_model_only`).

**If the smoke test shows malformed chat output**, set `TOKENIZER_MODE=kimi_k3`
(forwarded to both server and client) — that mode renders prompts with K3's Python
XTML encoding instead of a Jinja template.

## Durability (Weights & Biases)

Each config becomes **one W&B run** (so configs overlay in the dashboard), pushed
when the config finishes: per-concurrency curves per scenario lane (throughput /
TTFT / TPOT / acceptance length, **x-axis = concurrency**), the full results table,
and the raw JSONs / `server.log` / `bench.log` / `gpu.csv` / CSV as an artifact.

Runs automatically from `run.sh` / `run_all.sh`. `.env` is auto-loaded; `wandb` is
auto-installed if missing. Disable with `WANDB=0`. Project defaults to
`aiand-kimik3-mxfp4` (override `WANDB_PROJECT`). Nothing in `results/` is committed
to git — W&B is the store.

## Logs

Each config writes everything to `results/<config>/`:

- `server.log` — vLLM output; the **first lines record the exact `vllm serve`
  command and relevant env vars** used to launch it.
- `bench.log` — full console output of the sweep, including the per-cell
  input/output length statistics.
- `gpu.csv` — per-second GPU power/clocks/util (InferenceX `start_gpu_monitor`).
- `*.json` — one benchmark result per (lane, scenario, concurrency).
- `*.accept.json` — spec-decode counter deltas for that cell.

## Workflow

1. Benchmark **baseline** (FULL sweep) — the reference curve.
2. Screen each optimization with a **SUBSET** sweep and compare to baseline.
   Decide parallelism (`opt01`/`opt02`) first — `opt09`/`opt10`/`opt11` are only
   worth running if DP8EP beat TP8.
3. Settle the spec-decoding question with `ref_nonmtp` vs `opt13_mtp_kimik3` vs
   `baseline`/`opt12`/`opt14`, reading `Accept len` alongside throughput.
4. Collect the winners into `servers/final1.sh` / `final2.sh` (replace every
   `[SCREEN]` line) and run `run_final.sh` (FULL).
5. Run `eval/quality_check.sh` on each final config vs baseline.

## Open questions (only a run on the box can settle these)

1. **Is the MXFP4 checkpoint actually multimodal?** The HF model card describes a
   MoonViT-V2 vision encoder, but InferenceX's K3 script header says "NO
   `--language-model-only` (text-only checkpoint)". If it is text-only, `opt18` is
   pointless and MMMU-Pro is N/A. Check for a vision tower in the served
   `config.json` before spending a config on it.
2. **Thinking on or off for the quality gate?** K3 is a thinking model — its
   reasoning parser defaults `enable_thinking=True`, and InferenceX collects K3
   acceptance for `thinking=on` only. This harness forces thinking **off** for
   MMLU-Pro (inherited from the GLM campaign). That may understate K3. Decide before
   quoting accuracy; override with `K3_CHAT_TEMPLATE_KWARGS` / `EVAL_GEN_KWARGS`.
~~3. Prefix caching on or off?~~ **Settled 2026-07-29: OFF, no override.** Both
   reference sources enable it, so every absolute number here is a floor versus
   theirs; that must be stated whenever the numbers are shown side by side.

## Known caveats

- **DSpark draft model.** `Inferact/Kimi-K3-DSpark` must be downloadable (or
  pre-staged at `DRAFT_MODEL_PATH`), or every config except `ref_nonmtp` and
  `opt13_mtp_kimik3` fails at startup.
- **Chat rendering.** Both lanes wrap prompts with InferenceX's
  `tokenizer.apply_chat_template` (Jinja). K3 also has a non-Jinja path
  (`--tokenizer-mode kimi_k3`). If Jinja is wrong for K3 the error is *systematic
  across both lanes*, so config-to-config comparisons stay valid while absolute
  acceptance length would not.
- **Single-node DEP is exploratory.** The recipe lists `multi_node_dep` with
  `strategy_min_gpus: 16`.
- **Pre-release model.** The recipe is marked `nightly_required: true` with an
  estimated VRAM footprint, so names can still move. `preflight.sh` is the check.
