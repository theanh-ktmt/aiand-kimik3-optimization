#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 3 — Scheduler hyperparameters: lift the sequence cap and the prefill chunk.
#
# The recipe caps --max-num-seqs at 32 *specifically because DSpark needs extra
# VRAM* ("Speculative decoding needs additional VRAM, so cap concurrent
# sequences"). That cap — not the kernels — is the hard ceiling at the conc=128 end
# of the sweep, so it is the highest-expected-value hyperparameter to challenge.
#
# 512 is not a guess: it is what InferenceX's own Kimi-K3 B300 SPEED-Bench script
# runs (MAX_NUM_SEQS default 512, with DSpark on), and their agentic script uses
# 2*CONC. Their production profile also runs --gpu-memory-utilization 0.90 rather
# than the recipe 0.95, so if this OOMs, drop GPU_MEM_UTIL before dropping seqs.
# Fall back through 256 / 128 / 64 rather than abandoning the knob.
#
# --max-cudagraph-capture-size is pinned to the sequence cap because, in
# InferenceX's words, "a 93-layer 2.8T model makes capturing vLLM's full 2048-wide
# ladder prohibitively slow".
# 16384 batched tokens matches the recipe own TEP prefill profile.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="${CONFIG:-opt03_hyperparams}"
BENCH_MODE="mtp"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-16384}"
    --max-num-seqs "${MAX_NUM_SEQS:-512}"
    --max-cudagraph-capture-size "${MAX_NUM_SEQS:-512}"
    --speculative-config "$(dspark_config 7)"
)
serve_main
