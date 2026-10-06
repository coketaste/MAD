#!/usr/bin/env python3
"""Run the blog's profiling matrix with madengine and archive every run's artifacts.

Each (model, run) pair is one `madengine run` invocation. The run's lens is selected
only through --additional-context (`tools` for madengine collectors, `model_args` for
the Primus built-in profiler), so every command in the blog is reproducible as-is.

After each run, everything madengine and Primus left in the MAD root is moved into
blog_results/<gpu>/<model>/<run>/ so runs never overwrite each other's traces.

Usage:
    python3 docs/blog/runkit/run_matrix.py --models B --runs B0a,P1
    python3 docs/blog/runkit/run_matrix.py --models A,B,C,D --runs all
    python3 docs/blog/runkit/run_matrix.py --models B --runs P1 --dry-run

Gated Hugging Face assets (model A) need MAD_SECRETS_HFTOKEN in the environment;
madengine forwards it to the container and redacts it from its logs.

Requires madengine with the profiling fixes from branch fix/profiling-validation-therock
(rocprofv3 output dir, perfetto preset, TraceLens curl, RTL empty-trace check). Validate
the collectors first with validate_profilers.py.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

MAD_ROOT = Path(__file__).resolve().parents[3]
GPU = os.environ.get("BLOG_GPU", "mi300x")
# Traces run to tens of GB; keep them off the (nearly full) root disk that holds MAD.
RESULTS_ROOT = Path(os.environ.get("BLOG_RESULTS_ROOT", "/data/ysha/primus_blog_results"))
RESULTS = RESULTS_ROOT / GPU
CACHE_IN_CONTAINER = "/myworkspace/.blog_cache"
# Primus models declare no madengine `data` entry, so Primus's own prepare hooks download
# HF weights (HF_HOME) and write converted checkpoints (DATA_PATH). Back that cache with a
# large host disk via docker_mounts; the directory must exist and be writable beforehand.
CACHE_ON_HOST = os.environ.get("BLOG_CACHE_HOST", "/data/ysha/primus_blog_cache")
ARCHIVE_IMAGE = os.environ.get("BLOG_ARCHIVE_IMAGE", "python:3.12-slim")

# Primus writes its experiment tree (logs, Kineto traces) under PRIMUS_WORKSPACE.
# Point it into run_directory, which --keep-model-dir preserves on the host.
COMMON_ENV = {
    "PRIMUS_WORKSPACE": "/myworkspace/run_directory/output",
    "DATA_PATH": f"{CACHE_IN_CONTAINER}/primus_data",
    # Megatron-Bridge's convert hook tries MOUNT_DATA_PATH, then /data, before DATA_PATH;
    # without this the converted checkpoint lands in the container layer and is redone per run.
    "MOUNT_DATA_PATH": f"{CACHE_IN_CONTAINER}/primus_data",
    "HF_HOME": f"{CACHE_IN_CONTAINER}/hf",
}

MEGATRON_P1 = (
    "--profile True --use_pytorch_profiler True "
    "--profile_step_start 10 --profile_step_end 12 "
    "--torch_profiler_with_stack False "
    "--profile_ranks [0,1,2,3,4,5,6,7]"
)

# Kernel lens: dispatch and copy timestamps only. No --hip-trace: on model D it recorded
# 20x more host API records than kernels (1.25M vs 63k), which dominated round 2's R3 cost.
KERNEL_TRACE = (
    "bash ../scripts/common/tools/rocprof_wrapper.sh --kernel-trace --memory-copy-trace "
    "--output-format json -d ./rocprof_output --"
)
_MEGATRON_TORCH = (
    "--profile True --use_pytorch_profiler True --profile_step_start 10 --profile_step_end 12 "
    "--torch_profiler_with_stack False "
)
# Kineto's window cost grows with the number of profiled ranks (model D: 7x with 8 ranks,
# 2.2x with rank 0), so only the collective report (PC) profiles every rank.
MEGATRON_PT = _MEGATRON_TORCH + "--torch_profiler_record_shapes False --profile_ranks [0]"
MEGATRON_PS = _MEGATRON_TORCH + "--torch_profiler_record_shapes True --profile_ranks [0]"
MEGATRON_PC = _MEGATRON_TORCH + "--torch_profiler_record_shapes False --profile_ranks [0,1,2,3,4,5,6,7]"
# Megatron-Bridge hard-codes with_stack=True (profiling.py:125); only record_shapes is ours.
_BRIDGE_TORCH = (
    "--profiling.use_pytorch_profiler True --profiling.profile_step_start 10 "
    "--profiling.profile_step_end 12 "
    "--logger.tensorboard_dir /myworkspace/run_directory/output/tensorboard "
)
BRIDGE_PT = _BRIDGE_TORCH + "--profiling.record_shapes False --profiling.profile_ranks [0]"
BRIDGE_PS = _BRIDGE_TORCH + "--profiling.record_shapes True --profiling.profile_ranks [0]"
BRIDGE_PC = _BRIDGE_TORCH + "--profiling.record_shapes False --profiling.profile_ranks [0,1,2,3,4,5,6,7]"

MODELS = {
    "A": {
        "name": "llama3.1_8B-torchtitan",
        "tag": "primus_train/torchtitan_MI300X_llama3.1_8B-BF16-pretrain",
        "config": "examples/torchtitan/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml",
        "base_args": "--training.steps 30",
        "short_args": "--training.steps 20",
        "p1_args": "--profiling.enable_profiling True --profiling.profile_freq 10",
        "rocprof": "rocprofv3_lightweight",
        "timeout": 7200,
    },
    "B": {
        "name": "qwen3_30B_A3B-megatron",
        "tag": "primus_train/megatron_MI300X_qwen3_30B_A3B-FP8-pretrain",
        "config": "examples/megatron/configs/MI300X/qwen3_30B_A3B-FP8-pretrain.yaml",
        "base_args": "--train_iters 30 --log_interval 1",
        "short_args": "--train_iters 20 --log_interval 1",
        "p1_args": MEGATRON_P1,
        "pt_args": MEGATRON_PT,
        "ps_args": MEGATRON_PS,
        "pc_args": MEGATRON_PC,
        "rocprof": "rocprofv3_communication",
        "extra_runs": ["K2s"],
        "timeout": 7200,
    },
    "C": {
        "name": "qwen3_32b_sft-megatron_bridge",
        "tag": "primus_train/megatron_bridge_MI300X_qwen3_32b_sft_posttrain",
        "config": "examples/megatron_bridge/configs/MI300X/qwen3_32b_sft_posttrain.yaml",
        # The config's lr_warmup_iters=50 must stay below train_iters (Megatron asserts
        # lr_warmup_steps < lr_decay_steps, and lr_decay_iters defaults to train_iters).
        "base_args": "--train_iters 30 --log_interval 1 --lr_warmup_iters 5",
        "short_args": "--train_iters 20 --log_interval 1 --lr_warmup_iters 5",
        # Primus exposes no profiler option for Megatron-Bridge, but the bridge's own
        # ProfilingConfig is reachable through overrides. Its trace handler writes to
        # logger.tensorboard_dir, which defaults to the image's Primus tree (lost with the
        # container), so redirect it into run_directory.
        "p1_args": (
            "--profiling.use_pytorch_profiler True --profiling.profile_step_start 10 "
            "--profiling.profile_step_end 12 --profiling.profile_ranks [0,1,2,3,4,5,6,7] "
            "--logger.tensorboard_dir /myworkspace/run_directory/output/tensorboard"
        ),
        "pt_args": BRIDGE_PT,
        "ps_args": BRIDGE_PS,
        "pc_args": BRIDGE_PC,
        "rocprof": "rocprofv3_lightweight",
        "timeout": 7200,
    },
    "D": {
        "name": "flux_535m-megatron_diffusion",
        "tag": "primus_train/megatron_MI300X_flux_535m_pretrain",
        "config": "examples/megatron/configs/MI300X/diffusion/flux_535m_pretrain.yaml",
        # rocm/primus:v26.7 exports NVTE_FLASH_ATTN=0 / NVTE_FUSED_ATTN=1, which Megatron's
        # default attention_backend=auto rejects; select the fused backend explicitly.
        # 43 ms steps: 30 iterations gave 7.9% baseline noise, so run 100.
        "base_args": "--train_iters 100 --log_interval 1 --attention_backend fused",
        "short_args": "--train_iters 100 --log_interval 1 --attention_backend fused",
        "p1_args": MEGATRON_P1,
        "pt_args": MEGATRON_PT,
        "ps_args": MEGATRON_PS,
        "pc_args": MEGATRON_PC,
        "rocprof": "rocprofv3_lightweight",
        "timeout": 7200,
    },
    "E": {
        "name": "wan2.1_1.3b-jax_maxdiffusion",
        "tag": "jax-maxdiffusion/maxdiffusion_MI300X_wan2.1_1.3b-pretrain",
        # Path-only copy of the Primus MI300X config (persistent compile cache, outputs
        # under run_directory); see the header of that file.
        "config": "/myworkspace/docs/blog/runkit/configs/wan2.1_1.3b-pretrain-MI300X.yaml",
        "base_args": "",
        "short_args": "",
        "p1_args": "--enable_profiler True --skip_first_n_steps_for_profiler 10 --profiler_steps 2",
        "rocprof": "rocprofv3_lightweight",
        "timeout": 14400,
    },
}

# v3 single-lens round. KT is not in `all`: on rocm/primus:v26.7, rocprofv3 kernel
# tracing costs ~280 us per dispatch (8x on the dummy model), so the GPU timeline comes
# from PT. KT and the round-2 runs (P1, K2, K2s, R3, R3c) stay runnable by name.
# Shapes cost nothing measurable on rank 0 (model D: PT 2.15x vs PS 2.18x), so PS also
# serves PT's op view; PT stays runnable by name.
RUNS = ["B0a", "B0b", "PS", "PC"]
EXTRA_RUNS = set()   # in `--runs all` only for models that list them


def build_context(model: dict, run: str) -> dict:
    args = f"--config_path {model['config']} "
    tools = []
    if run in ("B0a", "B0b"):
        args += model["base_args"]
    elif run == "P1":
        args += f"{model['base_args']} {model['p1_args']}"
    elif run == "K2":
        args += model["base_args"]
        tools = [{"name": "rocm_trace_lite"}]
    elif run == "K2s":
        args += model["base_args"]
        # RTL v0.3.3 modes are lite/default/full; `default` also times the has-signal
        # packets (e.g. RCCL kernels) that lite skips. Upstream later renamed it `standard`.
        tools = [{"name": "rocm_trace_lite_default"}]
    elif run in ("R3", "R3c"):
        # rocprofv3 traces the whole process, so cap iterations to bound trace volume.
        # TraceLens is stacked after the single collector; it runs as a post-script.
        args += model["short_args"]
        preset = "rocprofv3_communication" if run == "R3c" else model["rocprof"]
        mode = "pftrace" if preset == "rocprofv3_communication" else "rocprof"
        tools = [{"name": preset}, {"name": "tracelens", "env_vars": {"TRACELENS_MODE": mode}}]
    elif run == "KT":
        # Whole-run kernel trace; crop_rocprof.py cuts steps 10-11 on the host.
        args += model["short_args"]
        tools = [{"name": "rocprofv3_lightweight", "cmd": KERNEL_TRACE}]
    elif run in ("PT", "PS", "PC"):
        args += f"{model['base_args']} {model[run.lower() + '_args']}"
    else:
        raise ValueError(run)
    ctx = {
        "gpu_vendor": "AMD",
        "guest_os": "UBUNTU",
        "docker_env_vars": dict(COMMON_ENV),
        "docker_mounts": {CACHE_IN_CONTAINER: CACHE_ON_HOST},
        "model_args": " ".join(args.split()),
    }
    if tools:
        ctx["tools"] = tools
    return ctx


def archive(dest: Path, model: dict) -> None:
    """Move run artifacts (often root-owned) into dest via a throwaway container."""
    safe = model["tag"].replace("/", "_")
    items = [
        "run_directory", "rocprof_output", "rocm_trace_lite_output", "tracelens_output",
        "torch_profiler_output", "primus_perf_output.csv", "perf_entry.csv",
        "perf_entry.json", "perf_entry_super.csv", "perf_entry_super.json",
        "perf_super.csv", "perf_super.json", f"{safe}_*", "library_trace.csv",
    ]
    target = shlex.quote("/r/" + str(dest.relative_to(RESULTS_ROOT)))
    # madengine's trace post-script copies run_directory/rocprof_output to the MAD root, so
    # every rocprofv3 trace is archived twice (34 GB of duplicates in round 2). Drop the
    # top-level copy only when diff shows nothing that run_directory's copy lacks.
    top, inner = f"{target}/rocprof_output", f"{target}/run_directory/rocprof_output"
    dedupe = (
        f" if [ -d {top} ] && [ -d {inner} ] && ! diff -rq {top} {inner} | grep -qv '^Only in {inner}';"
        f" then rm -rf {top}; fi;"
    )
    script = (
        "set -u; cd /w; "
        + " ".join(f"for p in {i}; do [ -e \"$p\" ] && mv \"$p\" {target}/; done;" for i in items)
        + dedupe
        + f" chown -R {os.getuid()}:{os.getgid()} {target}"
    )
    subprocess.run(
        ["docker", "run", "--rm", "-v", f"{MAD_ROOT}:/w", "-v", f"{RESULTS_ROOT}:/r",
         ARCHIVE_IMAGE, "sh", "-c", script],
        check=True,
    )


def run_one(key: str, run: str, dry: bool) -> int:
    model = MODELS[key]
    dest = RESULTS / f"{key}_{model['name']}" / run
    ctx = build_context(model, run)
    cmd = [
        "madengine", "run", "--tags", model["tag"],
        "--additional-context", json.dumps(ctx),
        "--keep-model-dir", "--timeout", str(model["timeout"]),
        "-o", str(dest / "perf.csv"),
    ]
    print(f"\n=== {key} {run}: {model['tag']}\n{shlex.join(cmd)}", flush=True)
    if dry:
        return 0
    if (dest / "status.json").exists():
        print(f"skip: {dest} already has status.json", flush=True)
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "additional_context.json").write_text(json.dumps(ctx, indent=2) + "\n")
    t0 = time.time()
    with open(dest / "madengine.log", "w") as log:
        rc = subprocess.run(cmd, cwd=MAD_ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    archive(dest, model)
    status = {"model": key, "run": run, "tag": model["tag"], "rc": rc, "wall_s": round(wall, 1),
              "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (dest / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(f"--- {key} {run}: rc={rc} wall={wall/60:.1f} min", flush=True)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="A,B,C,D")
    ap.add_argument("--runs", default="all")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.dry_run and not os.access(CACHE_ON_HOST, os.W_OK):
        sys.exit(f"{CACHE_ON_HOST} must exist and be writable (set BLOG_CACHE_HOST to override)")
    # Fall back to the token saved by `hf auth login`, so runs don't depend on shell state.
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if not os.environ.get("MAD_SECRETS_HFTOKEN") and token_file.exists():
        os.environ["MAD_SECRETS_HFTOKEN"] = token_file.read_text().strip()
    if "A" in a.models.split(",") and not os.environ.get("MAD_SECRETS_HFTOKEN"):
        print("warning: model A (gated Llama 3.1 tokenizer) needs MAD_SECRETS_HFTOKEN", flush=True)
    failures = 0
    for key in a.models.split(","):
        model = MODELS[key]
        runs = RUNS if a.runs == "all" else a.runs.split(",")
        for run in runs:
            if a.runs == "all" and run in EXTRA_RUNS and run not in model.get("extra_runs", []):
                continue
            failures += run_one(key, run, a.dry_run) != 0
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
