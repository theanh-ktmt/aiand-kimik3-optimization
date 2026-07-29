# Kimi-K3 optimization space (8x B300, single node)

What can be tuned for **`moonshotai/Kimi-K3`** on one 8x B300 node, and how it
differs from the GLM-5.2-FP8 campaign. Every entry is grounded in the published
recipe (`vllm-project/recipes` → `models/moonshotai/Kimi-K3.yaml`,
rendered at <https://recipes.vllm.ai/moonshotai/Kimi-K3>) or in the model card;
nothing here is invented.

## 0. What the model is, and why the knobs differ from GLM-5.2

| | GLM-5.2-FP8 | **Kimi-K3** |
|---|---|---|
| Size | ~few-hundred-B MoE | **2.8T total**, 16-of-896 routed experts + 2 shared, 104B active |
| Weights | FP8 | **MXFP4 weights / MXFP8 activations** (QAT) |
| Layers | uniform | **93: 1 dense + 69 KDA + 24 Gated MLA** (hybrid attention) |
| Attention | sparse MLA (DSA) | **Kimi Delta Attention (KDA) + Gated MLA + Attention Residuals** |
| MoE | standard | **Stable LatentMoE** (3072 expert hidden, 3584 latent) |
| Spec decoding | **MTP** (in-model heads) | **DSpark** (separate `Inferact/Kimi-K3-DSpark` draft model) |
| Modality | text-only | **native multimodal** (MoonViT-V2, 401M) |
| Context | — | 1M tokens, vocab 160K |
| vLLM | 0.23.x | **≥ 0.27.0**, image `vllm/vllm-openai:kimi-k3` |
| VRAM | fits easily | **~1.4 TB of weights**; recipe minimum 1680 GiB aggregate (8x B300 = 2304 GiB) |

Four consequences drive the whole sweep:

1. **Spec decoding is a different mechanism.** No `{"method":"mtp"}`. It is
   `{"method":"dspark","model":"Inferact/Kimi-K3-DSpark", …}` with its own
   `attention_backend`, `draft_sample_method` and `rejection_sample_method` — so
   there are more axes to sweep than GLM's single `num_speculative_tokens`.
2. **The recipe caps `--max-num-seqs 32`** *because* DSpark needs extra VRAM.
   That cap, not the kernels, is the ceiling on the high-concurrency end of the
   curve. Challenging it is the highest-expected-value experiment in the set.
3. **Attention is hybrid** (69 recurrent-state KDA + 24 growing-KV MLA layers),
   which makes the **hybrid KV cache manager** a first-class knob — it has no
   equivalent in the GLM campaign.
4. **The vision encoder is loaded by default** but our benchmark traffic is 100%
   text, so `--language-model-only` is a real, recipe-sanctioned optimization.

## 1. Parallelism  (`opt01*`)

| Config | What | Why |
|---|---|---|
| `opt01a_tp8ep` | TP8 + `--enable-expert-parallel` | 896 experts: sharding beats replicating |
| `opt01b_dp8ep` | DP8 attention + EP | throughput-oriented; base for opt05/06/07 |
| `opt01c_tp4dp2ep` | TP4 × DP2 + EP | halves TP collective width, keeps all2all small |

Baseline is pure TP8 (`single_node_tp`, the recipe's `default_strategy`).
Caveat: the recipe lists `multi_node_dep` with `strategy_min_gpus: 16`, so DP+EP
on a single node is **exploratory** — validate on a subset sweep first.

## 2. Scheduler hyperparameters  (`opt02*`)

| Config | Change |
|---|---|
| `opt02a_maxseqs128` | `--max-num-seqs` 32 → 128 |
| `opt02b_batched_tokens` | `--max-num-batched-tokens 16384` (+ seqs 128) |
| `opt02c_gpumem097` | `--gpu-memory-utilization` 0.95 → 0.97 |

The recipe's own AMD override already runs `--max-num-seqs 128`, which is direct
evidence that 32 is a VRAM-conservative default rather than a hard limit. On
B300 (288 GiB/GPU) there should be headroom. 16384 batched tokens matches the
recipe's TEP prefill profile.

## 3. Attention & KV cache  (`opt03*`)

| Config | Change | Note |
|---|---|---|
| `opt03a_attn_flashinfer_mla` | `VLLM_ATTENTION_BACKEND=FLASHINFER_MLA` | same family the DSpark verify step uses |
| `opt03b_attn_flashmla` | `VLLM_ATTENTION_BACKEND=FLASHMLA` | DeepSeek FlashMLA kernels |
| `opt03c_mla_prefill_flashinfer` | `mla_prefill_backend` `TRTLLM_RAGGED` → `FLASHINFER` | isolates the prefill kernel; watch TTFT |
| `opt03d_kv_bf16` | `--kv-cache-dtype fp8` → `auto`, drop `use_prefill_query_quantization` | expected to lose; quality reference |
| `opt03e_hybrid_kv` | `--no-disable-hybrid-kv-cache-manager` | **most K3-specific knob in the sweep** |

`opt03e` matters because KDA layers hold a constant-size recurrent state while
Gated MLA layers grow with context. The hybrid manager sizes those separately
instead of budgeting every layer for the worst case → more KV capacity → bigger
batch → throughput. The recipe enables it on **both** prefill and decode workers
in its P/D profile, so it is well-trodden for this architecture.

Note the recipe's FP8-KV rule: whenever KV cache is FP8 you must also pass
`--attention-config '{"use_prefill_query_quantization":true}'`. The baseline does.

## 4. MoE backend  (`opt04*`)

| Config | Value |
|---|---|
| baseline | `--moe-backend auto` (recipe `base_args`) |
| `opt04a_moe_deepgemm_mega` | `deep_gemm_mega_moe` + `VLLM_USE_DEEP_GEMM=1` |
| `opt04b_moe_flashinfer_trtllm` | `flashinfer_trtllm` |
| `opt04c_moe_cutlass` | `cutlass` |
| `opt04d_moe_triton` | `triton` (floor / fallback) |
| `opt04e_moe_tail_fusion_off` | `VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=0` |

The recipe recommends `deep_gemm_mega_moe` for **any** DEP environment and
hardcodes it in its own decode profile, so 4a is the leading hypothesis.
`marlin` is a Hopper-only override in the recipe and is **not** included.
`opt04e` quantifies what K3's LatentMoE tail fusion is actually worth.

## 5. All2All communication  (`opt05*`, requires EP)

| Config | Backend | Note |
|---|---|---|
| `opt05a_a2a_nvlink_one_sided` | `flashinfer_nvlink_one_sided` | **recipe's NVLink recommendation**; expected winner intra-node |
| `opt05b_a2a_nvlink_two_sided` | `flashinfer_nvlink_two_sided` | extra handshake, may pipeline better |
| `opt05c_a2a_deepep_v2` | `deepep_v2` + `UCX_TLS="rc,cuda_copy"` | recipe's RDMA choice; the gap vs 5a prices a future 2-node scale-out |
| `opt05d/e` | `deepep_low_latency` / `deepep_high_throughput` | decode- vs prefill-oriented DeepEP |
| `opt05f` | `allgather_reducescatter` | zero-setup fallback / safety net |

## 6. Expert load balancing — EPLB  (`opt06`)

`--enable-eplb --eplb-config '{"window_size":1000,"step_interval":3000,"num_redundant_experts":32}'`

896 routed experts = 112 experts/GPU at EP8, ~7× more than a GLM-5.2-class
model. Routing skew therefore has far more room to create a straggler rank, and
one straggler stalls the entire all2all. Tune via `EPLB_REDUNDANT`.

## 7. Dual-batch overlap — DBO  (`opt07`)

`--enable-dbo --dbo-decode-token-threshold 32 --dbo-prefill-token-threshold 512`

Overlaps one micro-batch's MoE all2all with the other's GEMMs. The bigger the
all2all, the more there is to hide — and with 896 experts there is a lot.

## 8. Speculative decoding — DSpark  (`opt08*`)  ← the headline knob

Baseline: `{"model":"Inferact/Kimi-K3-DSpark","num_speculative_tokens":7,"method":"dspark","attention_backend":"FLASHINFER_MLA","draft_sample_method":"probabilistic","rejection_sample_method":"block"}`

| Config | Change |
|---|---|
| `opt08_dspark1` / `3` / `5` / `9` | `num_speculative_tokens` sweep around the default 7 |
| `opt08_dspark_greedy` | `draft_sample_method` `probabilistic` → `greedy` |
| `opt08_dspark_disable_bs64` | DSpark(7) for batch 1–64, **off** above 64 |
| `ref_nospec` | no speculative decoding at all (+ `--max-num-seqs 128`) |

`opt08_dspark_disable_bs64` encodes the core trade-off: spec decoding is a
latency win at small batch and a throughput **tax** once the GPU is saturated,
because every rejected draft token is compute a real request could have used.
Expressed as a `num_speculative_tokens_per_batch_size` schedule of
`(start, end, num_spec)` tuples — there is no standalone `disable_by_batch_size`.

**This is also why the dataset is ShareGPT, not random tokens** — see README.

## 9. Sampler  (`opt09`)

`VLLM_USE_FLASHINFER_SAMPLER=1`. K3's vocabulary is 160K, so per-step sampler
work is ~25% larger than a 128K-vocab model's, and spec decoding invokes the
sampler once per draft token.

## 10. CUDA graphs  (`opt10*`)

| Config | Mode |
|---|---|
| `opt10a_cudagraph_full_piecewise` | `FULL_AND_PIECEWISE` + capture sizes to 128 |
| `opt10b_cudagraph_decode_only` | `FULL_DECODE_ONLY` — what the recipe's own decode workers use |

93 layers means a lot of kernel-launch overhead to amortise, and DSpark makes
decode steps small and frequent — exactly the regime where capture pays.

## 11. Runtime/engine  (`opt11*`)

| Config | Change |
|---|---|
| `opt11a_model_runner_v2` | `VLLM_USE_V2_MODEL_RUNNER=1` |
| `opt11b_rust_frontend` | `VLLM_USE_RUST_FRONTEND=1` |
| `opt11c_v2_and_rust` | both |

The recipe says both "fully support this model and can be enabled if needed" —
validated but off by default, i.e. free candidates. They attack different layers
(engine step vs HTTP front end), so `opt11c` checks that they compose.

## 12. Skip the vision encoder  (`opt12`)

`--language-model-only`. **No GLM-5.2 equivalent.** Drops MoonViT-V2 and the
multimodal preprocessing path. Legitimate only if the deployment really is
text-only — and it makes MMMU-Pro impossible, so it is excluded from the vision
quality gate.

## 13. Collectives  (`opt13`)

`VLLM_ALLREDUCE_USE_FLASHINFER=0` (the recipe sets 1). At TP8 across 93 layers
there are many allreduces per token; this checks whether FlashInfer really beats
vLLM's custom all-reduce on B300.

## Deliberately out of scope

- **Prefix caching.** The recipe's Blackwell block sets
  `--enable-prefix-caching`; the harness forces it **off** so prefill is really
  measured. Every number here is therefore a floor versus production. This is the
  one intentional deviation from the recipe and it is applied uniformly.
- **`--load-format fastsafetensors`** — kept on everywhere. It changes weight-load
  time (which matters a lot for 1.4 TB) but not steady-state throughput, so it is
  a default, not a swept variable.
- **Hopper-only overrides** (`--moe-backend marlin`,
  `--disable-custom-all-reduce`, `--no-enable-flashinfer-autotune` as a Hopper
  fix) and all **AMD/ROCm** knobs (`VLLM_ROCM_USE_AITER`, `AITER_SITUV2_A8W4`,
  `AITER_BF16_FP8_MOE_BOUND`, `VLLM_USE_BREAKABLE_CUDAGRAPH`) — wrong hardware.
- **P/D disaggregation** (`pd_cluster`) and `multi_node_*` strategies — need
  ≥2 nodes. The `opt05c` RDMA data point is the cheapest proxy for what they'd
  cost.
- **`--mm-encoder-tp-mode data`** — only meaningful for image traffic, which this
  harness does not benchmark. Revisit if the workload becomes multimodal.
- **`--max-model-len 1048576`** — the recipe's production value. We cap at 16384
  so KV capacity is not the variable under test; the recipe itself says to adjust
  it per benchmark scenario.
