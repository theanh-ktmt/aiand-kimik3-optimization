# Server scripts

25 scripts: baseline + 20 optimizations + `spec.sh` (the parameterized
speculative-decoding sweep) + a non-MTP reference + 2 final placeholders. Trimmed
to the knobs with real expected value — see
[`OPTIMIZATIONS.md`](OPTIMIZATIONS.md) for what was cut and why.

Every script carries:

* the mandatory harness flags from `common.sh` —
  `--no-enable-prefix-caching`, `--trust-remote-code`, `--max-model-len 16384`,
  `--load-format fastsafetensors`, `--disable-uvicorn-access-log`,
  `--tool-call-parser kimi_k3`, `--enable-auto-tool-choice`,
  `--reasoning-parser kimi_k3`
* the recipe-mandated Kimi-K3 base block `"${K3_BASE_ARGS[@]}"` (from
  `k3_base_args`) —
  `--gpu-memory-utilization 0.95`, `--moe-backend auto`,
  `--no-enable-flashinfer-autotune`, `--kv-cache-dtype fp8`,
  `--attention-config '{"mla_prefill_backend":"TRTLLM_RAGGED","use_prefill_query_quantization":true}'`
  plus env `VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1`,
  `VLLM_ALLREDUCE_USE_FLASHINFER=1`, `VLLM_ENGINE_READY_TIMEOUT_S=3600`

A config overrides a base value either by setting the matching variable before
`k3_base_args` (`MOE_BACKEND`, `KV_CACHE_DTYPE`, `ATTENTION_CONFIG`,
`GPU_MEM_UTIL`) or by repeating the flag after the base block (last wins).

Run one directly to launch + hold the server:  `bash servers/<name>.sh`
Run one end-to-end (launch + bench + CSV):      `bash run.sh <name> [full|subset]`

## Bench mode decides which dataset lanes run

`BENCH_MODE` follows the InferenceX method:

| BENCH_MODE | lanes |
|---|---|
| `nonmtp` | random dataset, **raw** prompts |
| `mtp` | random dataset **+ chat template**, *and additionally* ShareGPT **+ chat template** |

Everything is `mtp` except `ref_nonmtp`, because the Day-0 baseline itself runs
speculative decoding. The two `mtp` lanes are separate rows in the CSV
(`Dataset` column) and are never averaged together.

## Shorthand used below

* `DSpark(n)` = `--speculative-config` with
  `{"model":"Inferact/Kimi-K3-DSpark","num_speculative_tokens":n,"method":"dspark","attention_backend":"FLASHINFER_MLA","draft_sample_method":"probabilistic","rejection_sample_method":"block"}`
  (built by `dspark_config`).
* `K3MTP(n)` = `{"method":"kimi_k3_mtp","num_speculative_tokens":n}`
  (built by `kimi_k3_mtp_config`) — the **in-model** MTP head.

| # | Script | Optimization | Key flags added / changed | Bench |
|---|--------|--------------|---------------------------|-------|
| 0 | `baseline.sh` | **Baseline (Day 0)** — published recipe | `--tensor-parallel-size 8 --max-num-seqs 32` + DSpark(7) | mtp |
| 1 | `opt01_tp8ep.sh` | Parallelism: TP8 + EP | `--enable-expert-parallel --enable-ep-weight-filter` | mtp |
| 2 | `opt02_dp8ep.sh` | Parallelism: DP8 + EP | `--data-parallel-size 8 --enable-expert-parallel --enable-ep-weight-filter` (base for 9/10/11) | mtp |
| 3 | `opt03_hyperparams.sh` | Batching | `--max-num-batched-tokens 16384 --max-num-seqs 512 --max-cudagraph-capture-size 512` | mtp |
| 4 | `opt04_perf_mode_throughput.sh` | High-level mode | `--performance-mode throughput` | mtp |
| 5 | `opt05_linear_flashinfer.sh` | **KDA / linear kernel (69 of 93 layers)** | `--mamba-backend FLASHINFER` (default is TRITON) | mtp |
| 6 | `opt06_attn_flashmla.sh` | MLA kernel (24 layers) | `--attention-backend FLASHMLA` | mtp |
| 19 | `opt19_mla_prefill_flashinfer.sh` | MLA **prefill** kernel — settles a source conflict | `mla_prefill_backend: FLASHINFER` (recipe says `TRTLLM_RAGGED`, InferenceX uses `FLASHINFER`) | mtp |
| 7 | `opt07_moe_deepgemm_mega.sh` | MoE backend | `MOE_BACKEND=deep_gemm_mega_moe` + `VLLM_USE_DEEP_GEMM=1` | mtp |
| 8 | `opt08_hybrid_kv.sh` | **Hybrid KV manager** (KDA + MLA) | `--no-disable-hybrid-kv-cache-manager` | mtp |
| 9 | `opt09_a2a_nvlink_one_sided.sh` | (DP8EP) All2All | `--all2all-backend flashinfer_nvlink_one_sided` | mtp |
| 10 | `opt10_eplb.sh` | (DP8EP) Expert load balancing | `--enable-eplb --eplb-config '{…,"num_redundant_experts":${EPLB_REDUNDANT:-32}}' --expert-placement-strategy` | mtp |
| 11 | `opt11_dbo.sh` | (DP8EP) Dual-batch overlap | `--enable-dbo --dbo-decode-token-threshold … --dbo-prefill-token-threshold …` | mtp |
| — | `spec.sh` | **Speculative-decoding sweep** (parameterized) | `SPEC_METHOD` = dspark / kimi_k3_mtp / none, times `SPEC_TOKENS=n`; driven by `run_spec_sweep.sh`, one label per point | mtp (nonmtp when none) |
| 14 | `opt14_spec_disable_bs64.sh` | Batch-gated spec decoding | DSpark(7) for batch 1–64, off above 64 | mtp |
| 15 | `opt15_async_scheduling.sh` | Scheduler | `--async-scheduling` | mtp |
| 16 | `opt16_cudagraph_decode_only.sh` | CUDA graph | `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --max-num-seqs 128 --max-cudagraph-capture-size 128` | mtp |
| 17 | `opt17_no_v2_runner_rust.sh` | Engine + frontend **OFF** (both are in the base preset) | `VLLM_USE_V2_MODEL_RUNNER=0` + `VLLM_USE_RUST_FRONTEND=0` | mtp |
| 20 | `opt20_no_tail_fusion.sh` | LatentMoE tail fusion **OFF** | `VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=0` | mtp |
| 21 | `opt21_no_flashinfer_allreduce.sh` | FlashInfer allreduce **OFF** | `VLLM_ALLREDUCE_USE_FLASHINFER=0` | mtp |
| 22 | `opt22_gpumem090.sh` | GPU memory fraction | `--gpu-memory-utilization 0.90` (preset is 0.95; InferenceX uses 0.90) | mtp |
| 18 | `opt18_language_model_only.sh` | Skip multimodal path | `--language-model-only` | mtp |
| — | `ref_nonmtp.sh` | **The base preset verbatim** — no drafting, no `--max-num-seqs` | (preset only) | **nonmtp** |
| — | `final1.sh` | **Proposed #1 (TP8)** — placeholder | KDA FLASHINFER + hybrid KV + async sched + batching + DSpark(3) + Model Runner v2 | mtp |
| — | `final2.sh` | **Proposed #2 (DP8EP)** — placeholder | as final1 but DP8EP + `flashinfer_nvlink_one_sided` + `deep_gemm_mega_moe` | mtp |

`final1.sh` / `final2.sh` are **placeholders**: every line marked `[SCREEN]` must
be replaced with the actual winner from `results/all.csv` before the numbers are
presented as a recommendation.

Run the final FULL reference benchmark (baseline + both finals → 3-way CSV):
`bash run_final.sh` → `results/final_full.csv`.

## The three-way speculative-decoding comparison

This is the question the campaign exists to answer, and it needs three configs:

| | mechanism | draft weights |
|---|---|---|
| `ref_nonmtp` (or `spec_none`) | none | — |
| `spec_kimi_k3_mtp_<n>` | in-model MTP head (`kimi_k3_mtp`) | none |
| `baseline` / `spec_dspark_<n>` / `opt14` | external draft model (`dspark`) | `Inferact/Kimi-K3-DSpark` |

Run it with `bash run_spec_sweep.sh` — it sweeps the token count for both
mechanisms and prints the acceptance-length-vs-throughput curve. `spec_none` is the
strictest floor (identical config, drafting removed); `ref_nonmtp` is the base
preset itself.

Read them against the **`Accept len`** column in the CSV (mean accepted tokens
per draft step, scraped from `/metrics` around each cell and stored in
`results/<config>/<cell>.accept.json`), not throughput alone — a config can win
on throughput while accepting fewer tokens, and that means something different.
Disable the scraping with `ACCEPT_METRICS=0`.

## Quality check (MMLU-Pro + MMMU-Pro)

Any server script also supports an accuracy run instead of a throughput sweep:

```bash
RUN_EVAL=1 bash servers/final1.sh              # launch -> lm-eval mmlu_pro -> teardown
RUN_EVAL=1 RUN_MMMU=1 bash servers/final1.sh   # ... and the vision task
bash eval/quality_check.sh baseline final1     # run both + print comparison table
```

Runs lm-eval `mmlu_pro` against `/v1/chat/completions` with **thinking disabled**
(`--gen_kwargs` `chat_template_kwargs`) and `--apply_chat_template`; results (with
`--log_samples`) land in `results/<config>/mmlu_pro/`. `EVAL_CONC` (default 64)
sets eval concurrency; `MMLU_PRO_TASK` / `MMMU_PRO_TASK` / `EVAL_GEN_KWARGS` /
`K3_CHAT_TEMPLATE_KWARGS` override the task and generation kwargs.

Unlike GLM-5.2 (text-only), **Kimi-K3 is natively multimodal**, so MMMU-Pro
applies — opt in with `RUN_MMMU=1`. `quality_check.sh` refuses to run it against
`opt18_language_model_only`, which disables multimodal inputs.

## Flag / value notes (all verified against `vllm serve --help=all` on the target image)

- **`--load-format fastsafetensors`**: not in the documented value list from
  `--help=all` (`auto, pt, safetensors, instanttensor, npcache, dummy, tensorizer,
  runai_streamer, runai_streamer_sharded, bitsandbytes, sharded_state, mistral,
  modelexpress`), but the help ends that list with "Other custom values can be
  supported via plugins" — and **both** of InferenceX's own Kimi-K3 B300 scripts
  run it, so it resolves on this image. `instanttensor` is the in-tree fallback.
- **`--mamba-backend`**: `MambaBackendEnum = TRITON (default), FLASHINFER, CPU`.
  This is the knob for the KDA layers. Related: `--mamba-cache-dtype`,
  `--mamba-ssm-cache-dtype {auto,bfloat16,float16,float32}`,
  `--enable-mamba-cache-stochastic-rounding`.
- **`--attention-backend`** exists as a real flag (also `--attention-config.backend`
  and `backend_per_kind` for per-attention-kind selection). Non-sparse MLA
  backends available: `FLASHINFER_MLA`, `FLASHMLA`, `CUTLASS_MLA`, `TRITON_MLA`,
  `FLASH_ATTN_MLA`. The `*_SPARSE` variants are for sparse-MLA models, not K3.
- **`--moe-backend`**: `aiter, auto, cutlass, deep_gemm, deep_gemm_mega_moe,
  emulation, flashinfer_b12x, flashinfer_cutedsl, flashinfer_cutlass,
  flashinfer_trtllm, flydsl, hpc, humming, marlin, triton, triton_unfused`.
  `marlin` is a Hopper-only override in the recipe — not for B300.
- **`--all2all-backend`**: `allgather_reducescatter, deepep_high_throughput,
  deepep_low_latency, deepep_v2, flashinfer_all2allv,
  flashinfer_nvlink_one_sided, flashinfer_nvlink_two_sided, mori_high_throughput,
  mori_low_latency, naive, nixl_ep, pplx`. The recipe names
  `flashinfer_nvlink_one_sided` for NVLink (single node) and `deepep_v2` for RDMA
  (cross-node, needs `UCX_TLS="rc,cuda_copy"`).
- **`--spec-method`** offers both `dspark` and `kimi_k3_mtp` — the basis of opt13.
- **FP8 KV cache**: whenever `--kv-cache-dtype fp8` is set, the recipe requires
  `--attention-config '{"use_prefill_query_quantization":true}'`. The base block
  does this.
- **Spec "disable by batch size"** is a `num_speculative_tokens_per_batch_size`
  schedule of `(start, end, num_spec)` tuples — no standalone flag exists.
- **`cudagraph_mode`**: `NONE, PIECEWISE, FULL, FULL_DECODE_ONLY,
  FULL_AND_PIECEWISE`.
- **`--tokenizer-mode kimi_k3`** renders chat prompts with K3's Python XTML
  encoding *instead of a Jinja template*. Left unset by default; set
  `TOKENIZER_MODE=kimi_k3` (server **and** client — `bench.sh` forwards it) if the
  smoke test shows malformed chat output.

Adjust values inline or via the env vars shown (`EPLB_REDUNDANT`, `SPEC_TOKENS`,
`MAMBA_BACKEND`, `ATTN_BACKEND`, `A2A_BACKEND`, `PERF_MODE`, `CUDAGRAPH_MODE`,
`SPEC_METHOD` / `SPEC_TOKENS`, the DBO thresholds, `FINAL1_SPEC_METHOD` /
`FINAL1_SPEC`), sweep, keep the winners.

`CONFIG_LABEL` lets one script produce many labelled runs — that is how
`run_spec_sweep.sh` gets a separate results dir and W&B run per token count out of a
single `spec.sh`:

```bash
CONFIG_LABEL=spec_dspark_5 SPEC_METHOD=dspark SPEC_TOKENS=5 bash run.sh spec subset
```

Server startup is bounded by `SERVER_STARTUP_TIMEOUT` (default **3600s = 60 min**,
matching the recipe's own `VLLM_ENGINE_READY_TIMEOUT_S`). Loading ~1.4 TB of MXFP4
weights is genuinely slow — see the cost warning at the top of `run_all.sh`.

## Cross-check: InferenceX already ships Kimi-K3 B300 scripts

The vendored submodule contains two reference scripts for this exact
model + hardware, and they are worth reading before trusting anything here:

* `third_party/InferenceX/benchmarks/single_node/speedbench/kimik3_fp4_b300_vllm.sh`
  — the golden **acceptance-length (AL)** matrix collector: sweeps DSpark 1..8 on
  SPEED-Bench and computes `AL = 1 + accepted/drafts` from `/metrics`. Our
  `Accept len` column uses the identical formula and counters.
* `third_party/InferenceX/benchmarks/single_node/agentic/kimik3_fp4_b300_vllm.sh`
  — agentic trace replay on the same production serve profile.

Where they differ from the recipe yaml, this repo now follows **them** for
`mla_prefill_backend` (as `opt19`), `--load-format`, `--disable-uvicorn-access-log`,
`--max-cudagraph-capture-size`, the `--max-num-seqs 512` candidate, and the
`NCCL_DMABUF_ENABLE` / `PYTHONNOUSERSITE` / `VLLM_HTTP_TIMEOUT_KEEP_ALIVE`
environment. Three of their choices are deliberately NOT adopted:

| Their choice | Ours | Why |
|---|---|---|
| `--enable-prefix-caching` | **always off, no override** | cache hits mask prefill cost, which defeats a configuration comparison. Settled 2026-07-29; a non-zero `PREFIX_CACHING` now fails the launch. |
| `--gpu-memory-utilization 0.90` | 0.95 (recipe) | 0.95 is the recipe base_args value; drop to 0.90 if a big `--max-num-seqs` OOMs. |
| `VLLM_USE_RUST_FRONTEND=1` always on | only in `opt17` | we want to measure what it is worth before adopting it. |
