#!/bin/bash
###############################################################################
#
# MIT License
#
# Copyright (c) Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################
# FastWAM on LIBERO, inside the image docker/pyt_fam builds. No nested docker.
#   --mode interleaved   four latency arms, one session, writes perf_fam_latency.csv
#   --mode arm --arm X   one latency arm; prints performance: <ms> ms
#   --mode smoke --arm X one-trial LIBERO screen on the gfx950 simulator
#   --mode frames        four LIBERO episodes, eager and GEAK at 2 steps, saved as frames
#   --mode cae           CAE-FastWAM latency at 10 steps
set -euo pipefail

MODE=""
ARM=""
ROUNDS=3
CSV="perf_fam_latency.csv"
CAE_STEPS=10

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift ;;
        --arm) ARM="$2"; shift ;;
        --rounds) ROUNDS="$2"; shift ;;
        --csv) CSV="$2"; shift ;;
        --cae-steps) CAE_STEPS="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "$MODE" ]]; then
    echo "run.sh: --mode is required" >&2
    exit 1
fi

# One render node is mounted when n_gpus is 1. Present it as device 0.
if [[ "${MAD_RUNTIME_NGPUS:-1}" == "1" ]]; then
    export HIP_VISIBLE_DEVICES=0
fi

if [[ -n "${MAD_SECRETS_HFTOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN="$MAD_SECRETS_HFTOKEN"
fi

FAM_ROOT="${FAM_ROOT:-/opt/faster-action-models}"
CACHE="${FAM_CACHE:-/myworkspace/fam_cache}"
mkdir -p "$CACHE/hf" "$CACHE/checkpoints/Wan-AI"
export HF_HOME="$CACHE/hf"
export HF_HUB_CACHE="$CACHE/hf"
export HF_CACHE_DIR="$CACHE/hf"
export DIFFSYNTH_MODEL_BASE_PATH="$CACHE/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export TOKENIZERS_PARALLELISM=false
export FAM_ROOT

hf_download() {
    if command -v hf >/dev/null 2>&1; then
        hf download "$@"
    else
        huggingface-cli download "$@"
    fi
}

FASTWAM_DIR="$CACHE/checkpoints/fastwam"
if [[ ! -f "$FASTWAM_DIR/libero_uncond_2cam224.pt" ]]; then
    hf_download yuanty/fastwam \
        libero_uncond_2cam224.pt libero_uncond_2cam224_dataset_stats.json \
        --local-dir "$FASTWAM_DIR"
fi
export MODEL_CKPT="$FASTWAM_DIR/libero_uncond_2cam224.pt"
export DATASET_STATS="$FASTWAM_DIR/libero_uncond_2cam224_dataset_stats.json"

WAN22="$CACHE/checkpoints/Wan-AI/Wan2.2-TI2V-5B"
WAN21="$CACHE/checkpoints/Wan-AI/Wan2.1-T2V-1.3B"
if [[ ! -f "$WAN22/Wan2.2_VAE.pth" || ! -f "$WAN22/models_t5_umt5-xxl-enc-bf16.pth" ]]; then
    hf_download Wan-AI/Wan2.2-TI2V-5B \
        Wan2.2_VAE.pth models_t5_umt5-xxl-enc-bf16.pth \
        --local-dir "$WAN22"
fi
if [[ ! -d "$WAN21/google/umt5-xxl" ]]; then
    hf_download Wan-AI/Wan2.1-T2V-1.3B \
        --include "google/umt5-xxl/*" \
        --local-dir "$WAN21"
fi

if [[ "$MODE" == "cae" ]]; then
    CAE_DIR="$CACHE/checkpoints/CAE-FastWAM"
    if [[ ! -f "$CAE_DIR/policy.pt" || ! -f "$CAE_DIR/action_vae.pt" ]]; then
        hf_download amd/CAE-FastWAM --local-dir "$CAE_DIR"
    fi
    export MODEL_CKPT="$CAE_DIR/policy.pt"
    export ACTION_VAE_CKPT="$CAE_DIR/action_vae.pt"
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
python3 "$HERE/fam_driver.py" \
    --mode "$MODE" \
    --arm "$ARM" \
    --rounds "$ROUNDS" \
    --csv "$CSV" \
    --cae-steps "$CAE_STEPS"
