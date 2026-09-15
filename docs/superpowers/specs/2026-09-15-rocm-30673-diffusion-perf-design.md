# ROCM-30673 Diffusion Performance Fix

## Problem

MAD diffusion runs on `rocm/primus:v26.6` are reported 10–15% slower than
manual runs in the same image on MI300X/MI355X-class GPUs. The MAD wrapper sets
performance environment values that differ from the Primus defaults, including
`GPU_MAX_HW_QUEUES=8` instead of `2` and
`TORCH_NCCL_HIGH_PRIORITY=0` instead of `1`.

## Scope

Change only `scripts/pytorch_train/pytorch_benchmark_report.sh` and focused
tests. Preserve explicit caller overrides. Do not alter model configs, batch
sizes, reporting, madengine, or unrelated launchers.

## Implementation

Set the diffusion wrapper's performance defaults to the Primus values:

- `GPU_MAX_HW_QUEUES=2`
- `TORCH_NCCL_HIGH_PRIORITY=1`
- `CUDA_DEVICE_MAX_CONNECTIONS=1`
- `HSA_ENABLE_SDMA=1`
- `HSA_NO_SCRATCH_RECLAIM=1` for post-training

Use `${VAR:-default}` so user-supplied values continue to win. Add a shell test
that extracts the environment setup without launching training and verifies
both defaults and override preservation.

## Validation

1. Run the new test before implementation and confirm it fails for the expected
   environment mismatch.
2. Apply the minimal fix and confirm focused tests and shell syntax checks pass.
3. On the same 8-GPU MI350X node and exact `rocm/primus:v26.6` image, compare:
   - manual AMDiffusionBenchmark launch,
   - MAD launch with the old environment,
   - MAD launch with the fixed environment.
4. Keep image, model, batch size, GPU set, assets, and node constant. Alternate
   run order where practical and compare median `avg_fps_gpu` and
   `avg_tflops`.

The regression is resolved if the fixed MAD result is within normal run
variance of the manual result and materially recovers the reported 10–15% gap.
If it does not, retain the branch evidence and next isolate MAD's batch-size
override.

## Risks

Full benchmarks may be blocked by registry access, image availability, model
assets, Hugging Face credentials, or runtime duration. These are reported as
validation limitations rather than treated as proof of a fix.
