# Server scripts

Every script under `servers/` launches one vLLM configuration. All of them carry:

* the mandatory harness flags from `common.sh` —
  `--no-enable-prefix-caching`, `--trust-remote-code`, `--max-model-len 16384`,
  `--load-format fastsafetensors`, `--tool-call-parser kimi_k3`,
  `--enable-auto-tool-choice`, `--reasoning-parser kimi_k3`
* the recipe-mandated Kimi-K3 base block `"${K3_BASE_ARGS[@]}"` (from
  `k3_base_args` in `common.sh`) —
  `--gpu-memory-utilization 0.95`, `--moe-backend auto`,
  `--no-enable-flashinfer-autotune`, `--kv-cache-dtype fp8`,
  `--attention-config '{"mla_prefill_backend":"TRTLLM_RAGGED","use_prefill_query_quantization":true}'`
  plus env `VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1`,
  `VLLM_ALLREDUCE_USE_FLASHINFER=1`, `VLLM_ENGINE_READY_TIMEOUT_S=3600`

...plus their own optimization-specific flags shown below. A config overrides a
base value either by exporting the matching variable before `k3_base_args`
(`MOE_BACKEND`, `KV_CACHE_DTYPE`, `ATTENTION_CONFIG`, `GPU_MEM_UTIL`) or by
repeating the flag after the base block (last occurrence wins).

Run one directly to launch + hold the server:  `bash servers/<name>.sh`
Run one end-to-end (launch + bench + CSV):      `bash run.sh <name> [full|subset]`

Bench mode is `spec` (client posts to `/v1/chat/completions`, so the server
applies the K3 chat template) for everything except `ref_nospec`, because the
Day-0 baseline itself runs DSpark speculative decoding.

`DSpark(n)` below is shorthand for
`--speculative-config '{"model":"Inferact/Kimi-K3-DSpark","num_speculative_tokens":n,"method":"dspark","attention_backend":"FLASHINFER_MLA","draft_sample_method":"probabilistic","rejection_sample_method":"block"}'`
(built by `dspark_config` in `common.sh`).

| # | Script | Optimization | Key flags added / changed | Bench |
|---|--------|--------------|---------------------------|-------|
| 0 | `baseline.sh` | **Baseline (Day 0)** — published recipe | `--tensor-parallel-size 8 --max-num-seqs 32` + DSpark(7) | spec |
| 1a | `opt01a_tp8ep.sh` | Parallelism: TP8 + EP | `--enable-expert-parallel` | spec |
| 1b | `opt01b_dp8ep.sh` | Parallelism: DP8 + EP | `--data-parallel-size 8 --enable-expert-parallel` (base for 5/6/7) | spec |
| 1c | `opt01c_tp4dp2ep.sh` | Parallelism: TP4 × DP2 + EP | `--tensor-parallel-size 4 --data-parallel-size 2 --enable-expert-parallel` | spec |
| 2a | `opt02a_maxseqs128.sh` | Hyperparams | `--max-num-seqs 128` (recipe caps at 32 for DSpark VRAM) | spec |
| 2b | `opt02b_batched_tokens.sh` | Hyperparams | `--max-num-batched-tokens 16384 --max-num-seqs 128` | spec |
| 2c | `opt02c_gpumem097.sh` | Hyperparams | `GPU_MEM_UTIL=0.97` + `--max-num-seqs 128` | spec |
| 3a | `opt03a_attn_flashinfer_mla.sh` | Attention backend | `VLLM_ATTENTION_BACKEND=FLASHINFER_MLA` | spec |
| 3b | `opt03b_attn_flashmla.sh` | Attention backend | `VLLM_ATTENTION_BACKEND=FLASHMLA` | spec |
| 3c | `opt03c_mla_prefill_flashinfer.sh` | MLA prefill kernel | `ATTENTION_CONFIG` `mla_prefill_backend: FLASHINFER` | spec |
| 3d | `opt03d_kv_bf16.sh` | KV dtype | `KV_CACHE_DTYPE=auto`, no `use_prefill_query_quantization` | spec |
| 3e | `opt03e_hybrid_kv.sh` | **Hybrid KV manager** (KDA + MLA) | `--no-disable-hybrid-kv-cache-manager` | spec |
| 4a | `opt04a_moe_deepgemm_mega.sh` | MoE backend | `MOE_BACKEND=deep_gemm_mega_moe` + `VLLM_USE_DEEP_GEMM=1` | spec |
| 4b | `opt04b_moe_flashinfer_trtllm.sh` | MoE backend | `MOE_BACKEND=flashinfer_trtllm` | spec |
| 4c | `opt04c_moe_cutlass.sh` | MoE backend | `MOE_BACKEND=cutlass` | spec |
| 4d | `opt04d_moe_triton.sh` | MoE backend | `MOE_BACKEND=triton` | spec |
| 4e | `opt04e_moe_tail_fusion_off.sh` | LatentMoE tail fusion | `VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=0` | spec |
| 5a | `opt05a_a2a_nvlink_one_sided.sh` | (DP8EP) All2All | `--all2all-backend flashinfer_nvlink_one_sided` (recipe's NVLink pick) | spec |
| 5b | `opt05b_a2a_nvlink_two_sided.sh` | (DP8EP) All2All | `--all2all-backend flashinfer_nvlink_two_sided` | spec |
| 5c | `opt05c_a2a_deepep_v2.sh` | (DP8EP) All2All | `--all2all-backend deepep_v2` + `UCX_TLS=rc,cuda_copy` | spec |
| 5d | `opt05d_a2a_deepep_low_latency.sh` | (DP8EP) All2All | `--all2all-backend deepep_low_latency` | spec |
| 5e | `opt05e_a2a_deepep_high_throughput.sh` | (DP8EP) All2All | `--all2all-backend deepep_high_throughput` | spec |
| 5f | `opt05f_a2a_allgather_reducescatter.sh` | (DP8EP) All2All | `--all2all-backend allgather_reducescatter` (fallback) | spec |
| 6 | `opt06_eplb.sh` | (DP8EP) EPLB — 896 experts | `--enable-eplb --eplb-config '{…,"num_redundant_experts":${EPLB_REDUNDANT:-32}}'` | spec |
| 7 | `opt07_dbo.sh` | (DP8EP) DBO | `--enable-dbo --dbo-decode-token-threshold … --dbo-prefill-token-threshold …` | spec |
| 8a | `opt08_dspark1.sh` | DSpark | `num_speculative_tokens=1` | spec |
| 8b | `opt08_dspark3.sh` | DSpark | `num_speculative_tokens=3` | spec |
| 8c | `opt08_dspark5.sh` | DSpark | `num_speculative_tokens=5` | spec |
| 8d | `opt08_dspark9.sh` | DSpark | `num_speculative_tokens=9` | spec |
| 8e | `opt08_dspark_greedy.sh` | DSpark | `draft_sample_method: greedy` | spec |
| 8f | `opt08_dspark_disable_bs64.sh` | DSpark | DSpark(7) for batch 1–64, off above 64 (`num_speculative_tokens_per_batch_size`) | spec |
| 9 | `opt09_flashinfer_sampler.sh` | Sampler (160K vocab) | `VLLM_USE_FLASHINFER_SAMPLER=1` | spec |
| 10a | `opt10a_cudagraph_full_piecewise.sh` | CUDA graph | `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE",…}'` + `--max-num-seqs 128` | spec |
| 10b | `opt10b_cudagraph_decode_only.sh` | CUDA graph | `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` | spec |
| 11a | `opt11a_model_runner_v2.sh` | Engine | `VLLM_USE_V2_MODEL_RUNNER=1` | spec |
| 11b | `opt11b_rust_frontend.sh` | Frontend | `VLLM_USE_RUST_FRONTEND=1` | spec |
| 11c | `opt11c_v2_and_rust.sh` | Engine + frontend | both of the above | spec |
| 12 | `opt12_language_model_only.sh` | **Skip vision encoder** | `--language-model-only` (MoonViT-V2 off) | spec |
| 13 | `opt13_no_flashinfer_allreduce.sh` | Collectives | `VLLM_ALLREDUCE_USE_FLASHINFER=0` | spec |
| — | `ref_nospec.sh` | Reference | no `--speculative-config`; `--max-num-seqs 128` | **nospec** |
| — | `final1.sh` | **Proposed config #1 (TP8)** — placeholder | hybrid KV + 16384 batched tokens + seqs 128 + DSpark(3) + Model Runner v2 | spec |
| — | `final2.sh` | **Proposed config #2 (DP8EP)** — placeholder | as final1 but `--data-parallel-size 8 --enable-expert-parallel --all2all-backend flashinfer_nvlink_one_sided` + `deep_gemm_mega_moe` | spec |

`final1.sh` / `final2.sh` are **placeholders**: every line marked `[SCREEN]` in
them must be replaced with the actual winner from `results/all.csv` before the
numbers are presented as a recommendation.

Run the final FULL reference benchmark (baseline + both finals → 3-way CSV):
`bash run_final.sh` → `results/final_full.csv`.

## Quality check (MMLU-Pro + MMMU-Pro)

Any server script also supports an accuracy run instead of a throughput sweep:

```bash
RUN_EVAL=1 bash servers/final1.sh              # launch -> lm-eval mmlu_pro -> teardown
RUN_EVAL=1 RUN_MMMU=1 bash servers/final1.sh   # ... and the vision task
bash eval/quality_check.sh baseline final1     # run both + print comparison table
```

Runs lm-eval `mmlu_pro` directly against `/v1/chat/completions` with **thinking
disabled** (`--gen_kwargs` `chat_template_kwargs`) and `--apply_chat_template`;
results (with `--log_samples`) land in `results/<config>/mmlu_pro/`.
`EVAL_CONC` (default 64) sets eval concurrency; `MMLU_PRO_TASK` /
`MMMU_PRO_TASK` / `EVAL_GEN_KWARGS` / `K3_CHAT_TEMPLATE_KWARGS` override the
task and generation kwargs.

Unlike GLM-5.2 (text-only), **Kimi-K3 is natively multimodal**, so MMMU-Pro
applies — opt in with `RUN_MMMU=1`. `quality_check.sh` refuses to run it against
`opt12_language_model_only`, which has no vision encoder.

## Notes on flag / value names

- **MoE backend** (`--moe-backend`): the recipe uses `auto` as the base and
  recommends `deep_gemm_mega_moe` for expert-parallel deployments. `marlin` is a
  **Hopper-only** override in the recipe and is not used here.
- **All2All** (`--all2all-backend`): the recipe names `flashinfer_nvlink_one_sided`
  for NVLink (single node — what we have) and `deepep_v2` for RDMA (cross-node,
  needs `UCX_TLS="rc,cuda_copy"`). `deepep_*` variants require DeepEP in the
  image; `allgather_reducescatter` always works and is the fallback.
- **Attention** (`VLLM_ATTENTION_BACKEND`): K3 is hybrid KDA + Gated MLA, so this
  selects the kernel for the MLA layers only. `preflight.sh` prints the
  `AttentionBackendEnum` names this build actually exposes — use that list if
  `FLASHINFER_MLA` / `FLASHMLA` have been renamed.
- **FP8 KV cache**: whenever `--kv-cache-dtype fp8` is set, the recipe requires
  `--attention-config '{"use_prefill_query_quantization":true}'`. The base block
  does this; `opt03d` drops both together.
- **DSpark "disable by batch size"** is expressed as a
  `num_speculative_tokens_per_batch_size` schedule of `(start, end, num_spec)`
  tuples — there is no standalone `disable_by_batch_size` flag.
- **EPLB**: configured via a single `--eplb-config` JSON
  (`window_size`, `step_interval`, `num_redundant_experts`, …).

These are screening configs: adjust the values inline (or via the env vars shown,
e.g. `EPLB_REDUNDANT`, the DBO thresholds, `FINAL1_SPEC`), sweep, keep the winners.

Server startup is bounded by `SERVER_STARTUP_TIMEOUT` (default **3600s = 60 min**,
matching the recipe's own `VLLM_ENGINE_READY_TIMEOUT_S`); a launch that doesn't
pass `/health` in time fails the run instead of hanging. Loading ~1.4 TB of MXFP4
weights is genuinely slow — see the cost warning at the top of `run_all.sh`.
