# Kimi-K3 optimization space (8x B300, single node)

What can be tuned for **`moonshotai/Kimi-K3`** on one 8x B300 node, which knobs
made the screening list, and which were cut.

Grounded in two sources, not guesswork:
* the published recipe — `vllm-project/recipes` → `models/moonshotai/Kimi-K3.yaml`
  (rendered at <https://recipes.vllm.ai/moonshotai/Kimi-K3>)
* **`vllm serve --help=all` from the actual target image** (`vllm/vllm-openai:kimi-k3`),
  which is how every flag name, enum value and default below was verified — and how
  three errors in the recipe-derived first draft were caught (see "Corrections").

## 0. What the model is, and why the knobs differ from GLM-5.2

| | GLM-5.2-FP8 | **Kimi-K3** |
|---|---|---|
| Size | ~few-hundred-B MoE | **2.8T total**, 16-of-896 routed experts + 2 shared, 104B active |
| Weights | FP8 | **MXFP4 weights / MXFP8 activations** (QAT) → ~1.4 TB |
| Layers | uniform | **93: 1 dense + 69 KDA + 24 Gated MLA** (hybrid attention) |
| Attention | sparse MLA (DSA) | **Kimi Delta Attention (KDA) + Gated MLA + Attention Residuals** |
| MoE | standard | **Stable LatentMoE** (3072 expert hidden, 3584 latent) |
| Spec decoding | MTP (in-model heads) | **DSpark** (external draft model) **or `kimi_k3_mtp`** (in-model head) |
| Modality | text-only | **native multimodal** (MoonViT-V2, 401M) |
| Context / vocab | — | 1M tokens / 160K |
| vLLM | 0.23.x | **≥ 0.27.0**, image `vllm/vllm-openai:kimi-k3` |
| VRAM | fits easily | recipe minimum **1680 GiB**; 8x B300 = 2304 GiB → fits one node |

Five consequences drive the sweep:

1. **74% of the layers are KDA**, served through vLLM's Mamba/SSM path, whose
   default backend is `TRITON`. Every attention knob familiar from GLM only
   touches the other 24 layers. This is the largest untouched surface in the model
   and has no GLM analogue.
2. **There are TWO speculative-decoding mechanisms**, not one. `--spec-method`
   accepts both `dspark` (the recipe's external Inferact draft model) and
   `kimi_k3_mtp` (an in-model MTP head). The recipe never mentions the second.
3. **The recipe caps `--max-num-seqs 32`** *because* DSpark needs extra VRAM. That
   cap, not the kernels, is the ceiling at high concurrency.
4. **Attention is hybrid**, which makes the **hybrid KV cache manager** a
   first-class knob: constant-size KDA state and growing MLA cache get separate
   budgets instead of worst-casing every layer.
5. **The vision encoder is in the default path** but our traffic is text, so
   `--language-model-only` is a real, recipe-sanctioned option.

## Corrections the help log forced on the first draft

| First draft (from the recipe) | Reality in the image |
|---|---|
| `--load-format fastsafetensors` | Not in the documented value list (`auto, pt, safetensors, instanttensor, npcache, dummy, …`) — but the help ends with "Other custom values can be supported via plugins", and **both** InferenceX Kimi-K3 B300 scripts run it, so it resolves here. Kept as the default; `instanttensor` is the in-tree fallback. |
| `VLLM_ATTENTION_BACKEND` env | `--attention-backend` is a real **flag** (plus `--attention-config.backend` and `backend_per_kind` for per-attention-kind selection). Flags are preflight-verifiable and land in `server.log`; the env var is not. |
| "`--language-model-only` drops MoonViT-V2 and frees VRAM" | The help says only: "disables all multimodal inputs by setting all modality limits to 0. Equivalent to `--limit-mm-per-prompt` 0 for every modality." It removes multimodal profiling and scheduler branches; it does **not** promise the encoder weights are skipped. Any VRAM saving must be measured. |
| "add `compile_mm_encoder: false`" | Already `False` by default — the proposal was a no-op. Dropped. |

## The screening list (19 configs)

Ordered as `run_all.sh` runs them.

### 1–2. Parallelism  (`opt01_tp8ep`, `opt02_dp8ep`)

TP8+EP and DP8+EP. With 896 routed experts, sharding beats replicating. Both carry
**`--enable-ep-weight-filter`**, which only works with EP and pays for itself
immediately: per the help, each rank then "only reads its own expert shard from
disk, which can drastically reduce storage I/O for MoE models with per-expert
weight tensors" — on a 1.4 TB checkpoint that is the difference between a tolerable
and an intolerable campaign wall-clock.

Caveat: the recipe lists `multi_node_dep` with `strategy_min_gpus: 16`, so DP+EP on
one node is **exploratory**. `opt09`/`opt10`/`opt11` build on it and are only worth
running if `opt02` beat `opt01`.

### 3–4. Batching  (`opt03_hyperparams`, `opt04_perf_mode_throughput`)

`opt03` = `--max-num-seqs 512` + `--max-num-batched-tokens 16384` +
`--max-cudagraph-capture-size 512`. The 32-sequence cap is the
highest-expected-value hyperparameter to challenge, and 512 is not a guess: it is
what **InferenceX's own Kimi-K3 B300 SPEED-Bench script runs by default, with
DSpark on**. 16384 matches the recipe's TEP prefill profile. The capture-size pin
exists because "a 93-layer 2.8T model makes capturing vLLM's full 2048-wide ladder
prohibitively slow" (their words).

`opt04` = `--performance-mode throughput`, a single high-level switch this build
exposes: "larger CUDA graphs, more aggressive batching, throughput-oriented
kernels". It deliberately overlaps `opt03` and `opt16` — if one switch matches the
hand-tuned configs, the hand-tuning is not worth maintaining.

### 5. KDA / linear-attention kernel  (`opt05_linear_flashinfer`)  ← **likely the biggest win**

`--mamba-backend TRITON → FLASHINFER`. `MambaBackendEnum = TRITON (default),
FLASHINFER, CPU`. This is the kernel for the 69 KDA layers, i.e. **74% of the
model**, currently on the portable-but-slow default.

Adjacent knobs, left out to keep this a single-variable test:
`--mamba-cache-dtype`, `--mamba-ssm-cache-dtype {auto,bfloat16,float16,float32}`,
`--enable-mamba-cache-stochastic-rounding`, and
`--kernel-config '{"linear_backend": …}'` (defaults to `'auto'`; the valid values
are not listed in `--help`, so probe before use).

### 6 + 19. MLA kernels  (`opt06_attn_flashmla`, `opt19_mla_prefill_flashinfer`)

`opt06` = `--attention-backend FLASHMLA`, the **decode** kernel for the 24 Gated MLA
layers. Alternatives in this build: `FLASHINFER_MLA` (what the recipe pins for the
DSpark verify step), `CUTLASS_MLA`, `TRITON_MLA`, `FLASH_ATTN_MLA`. The `*_SPARSE`
variants are for sparse-MLA models (DeepSeek DSA / GLM-5.2) — wrong for K3 gated MLA.

`opt19` = the **prefill** kernel, and it exists because the two credible sources
**disagree**: the recipe yaml says `mla_prefill_backend: TRTLLM_RAGGED`, while both
InferenceX Kimi-K3 B300 scripts say `FLASHINFER` ("MLA prefill runs on FlashInfer
per the production recipe"). One of them is stale; a subset sweep settles it. Watch
TTFT, not output throughput.

### 7. MoE backend  (`opt07_moe_deepgemm_mega`)

`auto → deep_gemm_mega_moe`. The recipe recommends it explicitly for any
expert-parallel deployment and hardcodes it in its own decode profile.

### 8. Hybrid KV cache manager  (`opt08_hybrid_kv`)

`--no-disable-hybrid-kv-cache-manager`. More KV capacity → bigger batch →
throughput, and the recipe enables it on both prefill and decode workers in its P/D
profile.

### 9–11. Expert-parallel communication and balancing (DP8EP only)

* `opt09_a2a_nvlink_one_sided` — `flashinfer_nvlink_one_sided`, the recipe's
  explicit NVLink recommendation and what its decode profile hardcodes.
* `opt10_eplb` — `--enable-eplb` (32 redundant experts) plus
  `--expert-placement-strategy`. 112 experts/GPU at EP8 is ~7× a GLM-class model,
  so routing skew has much more room to create a straggler rank, and one straggler
  stalls the whole all2all.
* `opt11_dbo` — dual-batch overlap: hide the all2all behind the other
  micro-batch's GEMMs. With 896 experts there is a lot to hide.

### 12–14. Speculative decoding  ← the question the campaign exists to answer

| config | mechanism |
|---|---|
| `ref_nonmtp` | none |
| `opt13_mtp_kimik3` | **in-model MTP head** (`kimi_k3_mtp`) — no draft weights, no extra VRAM |
| `baseline` (7) / `opt12_dspark3` (3) | external DSpark draft model |
| `opt14_spec_disable_bs64` | DSpark(7) for batch 1–64, **off** above 64 |

`opt14` encodes the core trade-off: spec decoding is a latency win at small batch
and a throughput **tax** once the GPU is saturated, because every rejected draft
token is compute a real request could have used.

Read all four against the **`Accept len`** column (mean accepted tokens per draft
step, scraped from `/metrics` per cell), not throughput alone.

### 15–18. Runtime

* `opt15_async_scheduling` — "avoids gaps in GPU utilization"; default is
  engine-decides, so pinning it on is free to try. Matters most here because 93
  layers + spec decoding makes each engine step short.
* `opt16_cudagraph_decode_only` — `FULL_DECODE_ONLY`, what the recipe's own decode
  workers use. Valid modes: `NONE, PIECEWISE, FULL, FULL_DECODE_ONLY,
  FULL_AND_PIECEWISE`.
* `opt17_v2_runner_rust` — Model Runner v2 + Rust frontend, both of which the
  recipe says "fully support this model and can be enabled if needed". Bundled into
  one config; split only if the bundle regresses.
* `opt18_language_model_only` — text-only serving. See the correction above for
  what it actually does.

## Third source: InferenceX already ships Kimi-K3 B300 scripts

The vendored submodule contains reference scripts for this exact model + hardware —
`benchmarks/single_node/speedbench/kimik3_fp4_b300_vllm.sh` (golden
acceptance-length matrix, DSpark 1..8) and
`benchmarks/single_node/agentic/kimik3_fp4_b300_vllm.sh` (agentic replay). Their
serve profile is the practical ground truth, and it differs from the recipe yaml in
ways that changed this repo:

| Setting | Recipe yaml | InferenceX K3 scripts | This repo |
|---|---|---|---|
| `mla_prefill_backend` | `TRTLLM_RAGGED` | `FLASHINFER` | baseline follows the recipe; `opt19` tests theirs |
| `--load-format` | `fastsafetensors` | `fastsafetensors` | `fastsafetensors` (proven to resolve) |
| `--max-num-seqs` | 32 (DSpark VRAM) | **512** / `2*CONC` | `opt03` tests 512 |
| `--max-cudagraph-capture-size` | — | pinned to `--max-num-seqs` | same, in `opt03` / `opt16` |
| `--gpu-memory-utilization` | 0.95 | 0.90 | 0.95, with 0.90 as the OOM fallback |
| `--disable-uvicorn-access-log` | — | yes | yes (in `common.sh`) |
| prefix caching | enabled | enabled | **off** by default; `PREFIX_CACHING=1` reproduces theirs |
| `VLLM_USE_RUST_FRONTEND` | "can be enabled" | always 1 | only in `opt17`, so we can price it |
| extra env | — | `NCCL_DMABUF_ENABLE=0`, `PYTHONNOUSERSITE=1`, `VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900` | adopted in `k3_env_defaults` |
| DSpark JSON | + `draft_sample_method` / `rejection_sample_method` | omits both (separate variant script has them) | we send the recipe form |
| EP | — | explicitly **not wired** (`EP_SIZE>1` rejected) | `opt01`/`opt02` are ours, and exploratory |

Their acceptance-length formula is `AL = 1 + accepted/drafts` from
`vllm:spec_decode_num_accepted_tokens_total` / `vllm:spec_decode_num_drafts_total`.
Our `Accept len` column uses the identical counters and formula, so the two are
directly comparable.

Also worth knowing: their K3 header calls the checkpoint **text-only** ("NO
`--language-model-only` (text-only checkpoint)") and treats K3 as a *thinking*
model, collecting acceptance only for `thinking=on`. Both bear on `opt18` and on the
quality gate — see the open questions in README.

## Cut from the sweep (and why)

Everything here is reachable by editing one value in an existing script, so nothing
is lost — only the cost of a separate 1.4 TB server load.

| Cut | Why |
|---|---|
| `opt01c_tp4dp2ep` (TP4×DP2) | TP8 and DP8 bracket the space; a hybrid only matters if both extremes are close |
| `--gpu-memory-utilization 0.97` | ~2 pts of headroom, real OOM risk at conc=128, low upside next to `--max-num-seqs` |
| `--kv-cache-dtype auto` (bf16) | doubles KV footprint; expected to lose, and FP8 KV is what the recipe validated |
| MoE `flashinfer_trtllm` / `cutlass` / `triton` | the recipe names one recommended backend; the rest are fallbacks, reachable via `MOE_BACKEND=` |
| 5 of 6 all2all backends | one node is all-NVLink, so the recipe's NVLink pick is the only strong candidate; reachable via `A2A_BACKEND=` |
| `VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=0` | the recipe turns it on for every NVIDIA target; disabling it is a debugging tool, not a candidate |
| `VLLM_ALLREDUCE_USE_FLASHINFER=0` | same |
| `VLLM_USE_FLASHINFER_SAMPLER=1` | plausible (160K vocab) but small next to the kernel/batching knobs |
| DSpark 1 / 5 / 9, `draft_sample_method greedy` | 3 vs 7 vs batch-gated brackets the curve; extra points are `SPEC_TOKENS=`/`DRAFT_SAMPLE_METHOD=` runs |
| `cudagraph_mode FULL_AND_PIECEWISE` | reachable via `CUDAGRAPH_MODE=` |
| Model Runner v2 and Rust frontend separately | bundled in `opt17` |

## Out of scope

- **Prefix caching.** The recipe's Blackwell block sets `--enable-prefix-caching`;
  the harness forces it **off** so prefill is really measured. Absolute numbers here
  are a **floor** versus production. Applied uniformly, so comparisons hold.
- **`--load-format instanttensor` / `--safetensors-load-strategy`** — affect
  weight-load time, not steady-state throughput. Campaign-level settings
  (`LOAD_FORMAT=`, `SAFETENSORS_LOAD_STRATEGY=`), not swept variables.
- **Hopper-only overrides** (`--moe-backend marlin`, `--disable-custom-all-reduce`)
  and all **AMD/ROCm** knobs (`VLLM_ROCM_USE_AITER`, `AITER_SITUV2_A8W4`,
  `AITER_BF16_FP8_MOE_BOUND`, `VLLM_USE_BREAKABLE_CUDAGRAPH`) — wrong hardware.
- **P/D disaggregation** (`pd_cluster`) and `multi_node_*` strategies — need ≥2
  nodes.
- **`--mm-encoder-tp-mode`, `cudagraph_mm_encoder`, `--limit-mm-per-prompt`** — only
  meaningful for image traffic, which this harness does not benchmark. Revisit if
  the workload becomes multimodal.
- **`--optimization-level 3`** (default 2) and `--kv-cache-dtype-skip-layers` — both
  interesting (the latter could keep KDA state at bf16 while MLA runs fp8), but
  second-round material once the first-order knobs have landed.
- **`--max-model-len 1048576`** — the recipe's production value. We cap at 16384 so
  KV capacity is not the variable under test; the recipe itself says to adjust it
  per benchmark scenario.
