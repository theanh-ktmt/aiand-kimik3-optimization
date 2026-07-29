# (AI&) Kimi-K3 MXFP4 Optimizations

Benchmark harness to find the best **vLLM 0.27.x** server configuration for
**`moonshotai/Kimi-K3`** (2.8T MXFP4 MoE) on **8x NVIDIA B300**, maximizing
throughput while keeping TTFT / TPOT acceptable.

Base config follows the published recipe:
<https://recipes.vllm.ai/moonshotai/Kimi-K3>
(source: `vllm-project/recipes` → `models/moonshotai/Kimi-K3.yaml`).
Container image: **`vllm/vllm-openai:kimi-k3`**.

Method follows [InferenceX](https://github.com/SemiAnalysisAI/InferenceX)
(vendored as a git submodule under `third_party/InferenceX`): saturated server
(`--request-rate inf`), `--ignore-eos`, prefix caching **off**, warmup before
every cell, swept across concurrency.

**See [`OPTIMIZATIONS.md`](OPTIMIZATIONS.md)** for the full optimization space and
why it differs from the GLM-5.2 campaign, and [`SERVERS.md`](SERVERS.md) for the
one-line-per-config index.

## The model, in one table

| | |
|---|---|
| Params | 2.8T total / 104B active (16 of 896 routed experts + 2 shared) |
| Layers | 93 = 1 dense + **69 KDA** + **24 Gated MLA** (hybrid attention) |
| Quant | **MXFP4 weights / MXFP8 activations** (QAT) → ~1.4 TB of weights |
| Context / vocab | 1,048,576 tokens / 160K |
| Spec decoding | **DSpark** — separate draft model `Inferact/Kimi-K3-DSpark` |
| Modality | native multimodal (MoonViT-V2, 401M) |
| Min hardware | recipe `vram_minimum_gb: 1680`; 8x B300 = 2304 GiB → fits one node |

## Layout

```
common.sh            shared config + mandatory flags + K3 recipe base block +
                     dspark_config() + server lifecycle helpers
servers/*.sh         one launch script per configuration (baseline + 13 opt groups)
bench/bench.sh       InferenceX-method sweep (ShareGPT or random; spec / nospec)
bench/get_sharegpt.sh  fetch the ShareGPT V3 dataset
run.sh               one config end-to-end: launch -> sweep -> teardown -> CSV
run_all.sh           the whole campaign: baseline + every opt SUBSET -> all.csv
aggregate.py         result JSONs -> CSV (sheet-pasteable)
servers/final1.sh    proposed config #1 (TP8);  final2.sh = #2 (DP8EP)   [placeholders]
run_final.sh         full bench for baseline + final1 + final2 -> final_full.csv
eval/quality_check.sh  MMLU-Pro (+ optional MMMU-Pro): baseline vs recommended
eval/parse_mmlu.py   lm-eval results -> accuracy comparison table + Pass?
preflight.sh         validate the box/build BEFORE a long campaign
results/<config>/    per-config outputs: *.json, server.log (with launch command),
                     bench.log, gpu.csv  (git-ignored)
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
```

Run inside the **`vllm/vllm-openai:kimi-k3`** container on a B300 node. Optional:
pre-stage weights and `export MODEL_PATH=/path/to/Kimi-K3` to skip the (~1.4 TB)
HF download. Every speculative-decoding config additionally needs
`Inferact/Kimi-K3-DSpark`; vLLM fetches it into `--download-dir` on first use, or
set `DRAFT_MODEL_PATH` to a pre-staged copy.

Fetch the benchmark dataset once:

```bash
bash bench/get_sharegpt.sh          # -> datasets/ShareGPT_V3_unfiltered_cleaned_split.json
```

**Run the preflight first** — it validates the environment without launching the
model (tools, vLLM ≥ 0.27.0, GPU count *and aggregate VRAM vs the 1680 GiB
minimum*, disk space for 1.4 TB, ShareGPT presence, `vllm bench serve` flags,
InferenceX submodule, and crucially that every `--flag` the configs use exists in
*this* `vllm serve --help=all`, plus the attention-backend names):

```bash
bash preflight.sh        # exit 0 = good to go; FAIL lines list anything missing
```

If a flag FAILs (e.g. `--language-model-only`, `--attention-config`,
`--moe-backend deep_gemm_mega_moe` may be newer than your exact build), either
update vLLM / use the `kimi-k3` image, or skip that optimization.

## Benchmark method

### Dataset: ShareGPT (default)

The sweep runs on **ShareGPT V3** — real multi-turn chat prompts — not synthetic
random tokens. This is a deliberate change from the GLM-5.2 harness and it exists
because **the headline optimization for K3 is speculative decoding**. DSpark's
acceptance length depends on how predictable the text is; random tokens are
unpredictable by construction, so a random-dataset sweep understates every
spec-decode config and makes the whole `opt08*` group meaningless. ShareGPT gives
representative acceptance behaviour, so the DSpark token-count sweep measures
something real.

Consequence worth knowing: **input length is dataset-driven, not fixed.** vLLM's
ShareGPT sampler keeps prompts ≤ 1024 tokens and prompt+output ≤ 2048, so the
scenarios vary output length instead of input length, and `aggregate.py` reports
the real mean input length per cell in the `Mean ISL tok` column.

`DATASET=random` still works and reproduces the GLM-style `1k/1k` + `8k/1k`
scenarios via InferenceX — use it as a prefill-heavy cross-check, since ShareGPT
cannot produce an 8k-input scenario.

> **Why two clients?** InferenceX's vendored `benchmark_serving.py` is
> random-only in this revision (`--dataset-name` is literally
> `choices=["random"]`). ShareGPT runs therefore use vLLM's own `vllm bench serve`
> — the upstream that InferenceX's copy was forked from. It emits the same result
> JSON schema, so `aggregate.py` / `wandb_sync.py` are unchanged, and the
> InferenceX *method* is preserved exactly.

### Invariants

- **Prefix caching is always off** (`--no-enable-prefix-caching`) — otherwise
  repeated warmup / multi-turn prefixes are served from cache, prefill is
  skipped, and throughput is inflated. Note the recipe's Blackwell block *does*
  enable prefix caching, so absolute numbers here are a **floor** versus a
  production deployment. The deviation is applied uniformly to every config.
- `--request-rate inf`, `--ignore-eos`, warmup = 2× concurrency,
  `--num-prompts` = 10× concurrency.
- Scenarios: **FULL** = ShareGPT with `OSL 1024` + `OSL 256`;
  **SUBSET** = `OSL 1024` only (faster screening loop).
- Concurrency sweep: `1 2 4 8 16 32 64 128` (FULL) or `1 16 128` (SUBSET — 128
  also exercises the batch-size-gated DSpark disable).
- **spec vs nospec** differ on the client by exactly one thing: `spec` posts to
  `/v1/chat/completions` so the server applies the Kimi-K3 chat template (DSpark
  was trained on chat-formatted input; raw prompts silently tank acceptance
  length); `nospec` posts to `/v1/completions`. The baseline runs DSpark(7), so
  it is benchmarked in `spec` mode.

Tracked metrics (simplified): **Output throughput**, **TTFT** (mean/median/P90),
**TPOT** (mean/median/P90), plus **Mean ISL tok** for ShareGPT cells.

## Logs

Each config writes everything to `results/<config>/`:

- `server.log` — vLLM server output; the **first lines record the exact
  `vllm serve` command and relevant env vars** used to launch it.
- `bench.log` — full console output of the benchmark sweep.
- `gpu.csv` — per-second GPU power/clocks/util (via InferenceX `start_gpu_monitor`).
- `*.json` — one benchmark result per (scenario, concurrency).

## Durability (Weights & Biases)

Each config becomes **one W&B run** (so configs overlay in the dashboard),
pushed at the end of the config: the per-concurrency curves (throughput / TTFT /
TPOT, **x-axis = concurrency**), the full results table, and the raw JSONs /
`server.log` / `bench.log` / `gpu.csv` / CSV as a run artifact.

Runs automatically from `run.sh` / `run_all.sh`. Setup on the box:

```bash
cp .env.example .env      # then put your WANDB_API_KEY in .env (git-ignored)
```

`.env` is auto-loaded; `wandb` is auto-installed if missing. Disable with `WANDB=0`.
Project defaults to `aiand-kimik3-mxfp4` (override `WANDB_PROJECT`). Nothing in
`results/` is committed to git — W&B is the store.

## Usage

### One config, end-to-end

```bash
bash run.sh baseline full                  # baseline, full ShareGPT sweep
bash run.sh opt04a_moe_deepgemm_mega       # an optimization, subset sweep (default)
bash run.sh baseline full random           # prefill-heavy cross-check (1k/1k + 8k/1k)
# -> results/<config>/*.json  and  results/<config>.csv
```

### Manual (launch + benchmark in separate shells)

Preferred when screening several configs that share one server, because a cold
1.4 TB load is expensive — see the cost warning in `run_all.sh`.

```bash
# shell A - launch and keep the server up (command is echoed + logged)
bash servers/opt09_flashinfer_sampler.sh

# shell B - benchmark the running server
CONFIG=opt09_flashinfer_sampler bash bench/bench.sh \
    --config opt09_flashinfer_sampler --mode spec --sweep subset
```

### Whole campaign

```bash
bash run_all.sh                    # baseline + every opt, SUBSET each
python3 aggregate.py results --baseline baseline --out results/all.csv
```

`run_all.sh` caps each config's wall clock (2 h subset / 6 h full — the 1.4 TB
load is inside the cap), reaps stray vLLM processes between configs, and
continues past failures. `SKIP_EXISTING=1` resumes.

### Final configs + full benchmark + quality check

```bash
bash run_final.sh                                  # -> results/final_full.csv
bash eval/quality_check.sh baseline final1         # MMLU-Pro, baseline vs final1
RUN_MMMU=1 bash eval/quality_check.sh baseline final1   # + vision (MMMU-Pro)
```

Before each config's sweep, a **smoke test** sends a few sample chat requests to
the freshly-launched server and prints prompt + response, so you can confirm it
answers correctly before committing to the sweep. `SMOKE_TEST=0` disables it;
`SMOKE_TEST_STRICT=1` aborts the run if nothing usable comes back;
`SMOKE_TEST_MM=1` adds an image request (K3 is multimodal — but not for
`opt12_language_model_only`).

`quality_check.sh` launches each config, runs lm-eval against
`/v1/chat/completions` **with thinking disabled**, tears down, and prints a table
with the accuracy delta and a Pass? verdict (fails if the candidate regresses
more than 1.0 accuracy point; tune with `--threshold`). Results are saved under
`results/<config>/mmlu_pro/` (and `mmmu_pro/`) and compared into
`results/quality_check.csv`. Tunables: `EVAL_CONC` (default 64),
`MMLU_PRO_TASK`, `MMMU_PRO_TASK`, `EVAL_GEN_KWARGS`, `K3_CHAT_TEMPLATE_KWARGS`.

## Workflow

1. Benchmark **baseline** (FULL sweep) — the reference curve.
2. For each optimization, run a **SUBSET** sweep (`opt*`) and compare to baseline.
   Decide the parallelism group (`opt01*`) first — the `opt05/06/07` configs are
   only worth running if DP8EP beat TP8.
3. Collect the winners into `servers/final1.sh` / `final2.sh` (replace every
   `[SCREEN]` line) and run `run_final.sh` (FULL).
4. Run `eval/quality_check.sh` on each final config vs baseline.

## Known caveats

- **DSpark draft model.** `Inferact/Kimi-K3-DSpark` must be downloadable (or
  pre-staged at `DRAFT_MODEL_PATH`), or every config except `ref_nospec` fails at
  startup.
- **Single-node DEP is exploratory.** The recipe lists `multi_node_dep` with
  `strategy_min_gpus: 16`, so `opt01b` / `opt05*` / `opt06` / `opt07` on one node
  are outside the validated envelope — validate on a subset sweep before trusting
  them.
- **Pre-release model.** The recipe is marked `nightly_required: true` with an
  estimated VRAM footprint, so flag names and backend names are more likely than
  usual to have moved. `preflight.sh` is the check for that.
