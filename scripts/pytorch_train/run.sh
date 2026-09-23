#!/bin/bash
###############################################################################
#
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
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

export HF_TOKEN=$MAD_SECRETS_HFTOKEN

# Parse named arguments
# --batch_size: omitted → Primus/YAML defaults; integer → Hydra override;
#               auto → MAD GPU table (historical MAD numbers).
BATCH_SIZE=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_repo) MODEL_REPO="$2"; shift ;;
        --batch_size)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --batch_size requires a value (positive integer or 'auto')."
                exit 1
            fi
            BATCH_SIZE="$2"
            shift
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [[ -n "$BATCH_SIZE" && "$BATCH_SIZE" != "auto" && ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --batch_size must be a positive integer or 'auto' (got '$BATCH_SIZE')."
    exit 1
fi

echo "=hyper params start="
echo $MODEL_REPO
echo "batch_size=${BATCH_SIZE:-yaml}"
echo "=hyper params end="

datatypes=("BF16")
sequence_lengths=("256")

if [[ "$MODEL_REPO" == "pyt_train_flux" ]]; then
  model="Flux"
  datatypes=("BF16")
  tasks=("posttrain")

elif [[ "$MODEL_REPO" == "pyt_train_stable-diffusion-xl" ]]; then
  model="Stable-Diffusion-XL"
  datatypes=("BF16")
  tasks=("posttrain")

elif [[ "$MODEL_REPO" == "pyt_train_mochi-1" ]]; then
  model="Mochi-1"
  datatypes=("BF16")
  tasks=("posttrain")

elif [[ "$MODEL_REPO" == "pyt_train_hunyuan-video" ]]; then
  model="Hunyuan-video"
  datatypes=("BF16")
  tasks=("posttrain")

elif [[ "$MODEL_REPO" == "pyt_train_wan2_1-i2v" ]]; then
  model="Wan2_1-i2v"
  datatypes=("BF16")
  tasks=("posttrain")

elif [[ "$MODEL_REPO" == "pyt_train_dlrm" ]]; then
  model="DLRM"
  datatypes=("TF32" "FP32")
  tasks=("pretrain")

else
  echo "Error: Unsupported model repo '$MODEL_REPO'."
  echo "Supported: pyt_train_flux, pyt_train_stable-diffusion-xl, pyt_train_mochi-1, pyt_train_hunyuan-video, pyt_train_wan2_1-i2v, pyt_train_dlrm"
  exit 1
fi

# Run pytorch setup script
bash ./pytorch_benchmark_setup.sh -m $model

echo "Model: $model"
# Loop through all combinations
for task in "${tasks[@]}"; do
  for datatype in "${datatypes[@]}"; do
    for sequence_length in "${sequence_lengths[@]}"; do
      echo "Running: $task - $model - $datatype - $sequence_length"
      report_args=(-t "$task" -m "$model" -p "$datatype" -s "$sequence_length")
      if [[ -n "$BATCH_SIZE" ]]; then
        report_args+=(-b "$BATCH_SIZE")
      fi
      ./pytorch_benchmark_report.sh "${report_args[@]}"
    done
  done
done
