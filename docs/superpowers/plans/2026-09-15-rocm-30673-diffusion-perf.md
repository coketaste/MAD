# ROCM-30673 Diffusion Performance Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make MAD diffusion launches use the same performance defaults as Primus while preserving explicit caller overrides, then measure the result against a manual run on the same MI350X node and image.

**Architecture:** Keep the fix in the existing PyTorch training wrapper. A focused shell regression test evaluates the wrapper's exported defaults and override behavior without downloading models or launching GPUs. Runtime validation uses the exact JIRA image and holds node, GPU set, model, assets, and batch size constant.

**Tech Stack:** Bash, Docker, madengine CLI, AMDiffusionBenchmark, ROCm/MI350X

---

### Task 1: Add the environment regression test

**Files:**
- Create: `scripts/pytorch_train/tests/test_perf_env.sh`
- Test: `scripts/pytorch_train/tests/test_perf_env.sh`

- [ ] **Step 1: Write a test that extracts and evaluates the wrapper's performance exports**

The test must unset the five variables, evaluate the export block from
`pytorch_benchmark_report.sh`, and assert these values:

```bash
GPU_MAX_HW_QUEUES=2
TORCH_NCCL_HIGH_PRIORITY=1
CUDA_DEVICE_MAX_CONNECTIONS=1
HSA_ENABLE_SDMA=1
HSA_NO_SCRATCH_RECLAIM=1
```

It must repeat the evaluation after setting sentinel values and assert that all
sentinels remain unchanged.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
bash scripts/pytorch_train/tests/test_perf_env.sh
```

Expected: nonzero exit reporting the current `GPU_MAX_HW_QUEUES=8` or
`TORCH_NCCL_HIGH_PRIORITY=0` mismatch.

- [ ] **Step 3: Commit the failing test**

```bash
git add scripts/pytorch_train/tests/test_perf_env.sh
git commit -m "test: cover Primus diffusion environment defaults"
```

### Task 2: Align MAD defaults with Primus

**Files:**
- Modify: `scripts/pytorch_train/pytorch_benchmark_report.sh:109-117`
- Test: `scripts/pytorch_train/tests/test_perf_env.sh`

- [ ] **Step 1: Replace unconditional mismatched exports with override-preserving defaults**

Use exactly:

```bash
export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-1}"
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-2}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TORCH_NCCL_HIGH_PRIORITY="${TORCH_NCCL_HIGH_PRIORITY:-1}"
```

Keep existing unrelated settings unchanged and remove the later duplicate
post-training `HSA_NO_SCRATCH_RECLAIM=1` export.

- [ ] **Step 2: Run focused verification**

```bash
bash scripts/pytorch_train/tests/test_perf_env.sh
bash -n scripts/pytorch_train/pytorch_benchmark_report.sh
bash -n scripts/pytorch_train/run.sh
```

Expected: all commands exit zero.

- [ ] **Step 3: Commit the implementation**

```bash
git add scripts/pytorch_train/pytorch_benchmark_report.sh
git commit -m "fix: align diffusion perf environment with Primus"
```

### Task 3: Validate on the MI350X node

**Files:**
- Create locally, do not commit: benchmark logs and extracted CSV summaries

- [ ] **Step 1: Pull and inspect the exact image**

```bash
docker pull rocm/primus:v26.6
docker image inspect rocm/primus:v26.6 --format '{{.Id}}'
```

Expected: pull succeeds and records one immutable image ID for every run.

- [ ] **Step 2: Confirm runtime prerequisites**

Check eight visible `gfx950` GPUs, available HF credentials, required assets,
free disk space, and no competing GPU process. Do not print credential values.

- [ ] **Step 3: Run a matched Flux comparison**

Use all eight GPUs and the same batch size for:

1. manual `python launcher.py train_args=flux-dev`,
2. MAD wrapper with the old env (`GPU_MAX_HW_QUEUES=8`,
   `TORCH_NCCL_HIGH_PRIORITY=0`),
3. MAD wrapper with the fixed defaults.

Preserve each generated `runs_summary.csv` and complete log under a timestamped
local results directory. If runtime permits, repeat in alternating order.

- [ ] **Step 4: Compare results**

Extract `avg_fps_gpu` and `avg_tflops`, calculate percent differences against
manual, and report individual runs plus medians. The result establishes a fix
only when fixed MAD is within normal run variance of manual and materially
recovers the old-env gap.

- [ ] **Step 5: Run final repository checks**

```bash
git diff primus-v26.6...HEAD --check
git status --short --branch
bash scripts/pytorch_train/tests/test_perf_env.sh
bash -n scripts/pytorch_train/pytorch_benchmark_report.sh
```

Expected: no whitespace errors; focused tests and syntax checks exit zero;
status contains only the intentional branch commits plus pre-existing untracked
files.
