#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 5 — KDA / linear-attention kernel: --mamba-backend TRITON -> FLASHINFER.
#
# LIKELY THE BIGGEST SINGLE WIN IN THE CAMPAIGN, and it has no GLM-5.2 analogue.
#
# Kimi-K3 has 93 layers: 1 dense + 69 KDA (Kimi Delta Attention — a linear /
# recurrent-state mechanism, served through vLLM Mamba SSM path) + 24 Gated MLA.
# So **74% of the layers** run on the Mamba backend, whose default is
# MambaBackendEnum.TRITON (confirmed: MambaBackendEnum = TRITON, FLASHINFER, CPU).
# Every other attention knob in this repo only touches the 24 MLA layers.
#
# Related knobs, deliberately not swept here so this stays a single-variable test:
#   --mamba-cache-dtype / --mamba-ssm-cache-dtype   SSM state precision
#   --enable-mamba-cache-stochastic-rounding        stability for long sequences
#   --kernel-config '{"linear_backend": ...}'       linear layer kernel ('auto'
#       by default; valid values are not listed in --help, so probe the enum
#       before using it)
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt05_linear_flashinfer"
BENCH_MODE="mtp"
k3_base_args
SERVE_ARGS=(
    "${K3_BASE_ARGS[@]}"
    --tensor-parallel-size 8
    --mamba-backend "${MAMBA_BACKEND:-FLASHINFER}"
    --max-num-seqs 32
    --speculative-config "$(dspark_config 7)"
)
serve_main
