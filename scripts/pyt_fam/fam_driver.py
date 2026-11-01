#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License. See docker/pyt_fam.ubuntu.amd.Dockerfile for the full text.
"""In-container driver for the FastWAM MAD models.

The upstream launchers start their own containers. madengine has already
started this one, so every measurement calls the Python entry point directly:

  interleaved   scripts/compare_artifacts.py's four arms, one session
  arm           one of those arms (a sweep cell)
  smoke         40 LIBERO episodes, one trial each, on the gfx950 simulator
  cae           CAE-FastWAM latency at 10 denoising steps
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

FAM = Path(os.environ.get("FAM_ROOT", "/opt/faster-action-models"))
FASTWAM = FAM / "FastWAM"
BASELINE_SRC = FAM / "src" / "fastwam-baseline"
GEAK_SRC = FAM / "src" / "fastwam-geak"
CAE_SRC = FAM / "src" / "cae-fastwam"
CONFIGS_BASELINE = FASTWAM / "configs_baseline"
CONFIGS_CAE = FASTWAM / "configs"
TASK_RUNNER = FAM / "geak" / "task" / "scripts" / "task_runner.py"
CAE_WORKER = FAM / "scripts" / "cae_fastwam_latency_worker.py"
BATCHED = FAM / "eval" / "native" / "fastwam_libero_batched.py"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CAE_TASK = (
    "libero_uncond_2cam224_1e-4_32hdim_latent_K8d16_freebits_pernorm_joint"
)

# Same arm environment as scripts/compare_artifacts.py.
ARM_SPECS = {
    "eager": (BASELINE_SRC, 10, {
        "FASTWAM_COMPILE": "0",
        "FASTWAM_MOT_COMPILE_MODE": "off",
        "FASTWAM_VAE_COMPILE": "0",
    }),
    "compile": (BASELINE_SRC, 10, {
        "FASTWAM_COMPILE": "0",
        "FASTWAM_MOT_COMPILE_MODE": "off",
        "FASTWAM_VAE_COMPILE": "0",
        "FWE_NAIVE_COMPILE": "1",
    }),
    "geak": (GEAK_SRC, 10, {"FASTWAM_KVSLAB": "0"}),
    "geak@2": (GEAK_SRC, 2, {"FASTWAM_KVSLAB": "0"}),
}
FRESH_CACHE = {"compile", "geak", "geak@2"}
INTERLEAVED = ("eager", "compile", "geak", "geak@2")


def require(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        sys.exit(f"missing {name}")
    return value


def base_env() -> dict:
    env = os.environ.copy()
    env.setdefault("HIP_VISIBLE_DEVICES", "0")
    env["MODEL_SRC_ROOT"] = str(FASTWAM)
    env["MODEL_CKPT"] = require("MODEL_CKPT")
    env["DATASET_STATS"] = os.environ.get("DATASET_STATS", "")
    env["HF_CACHE_DIR"] = require("HF_CACHE_DIR")
    env["MODEL_PYTHON"] = sys.executable
    env["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def perf_line(text: str) -> float | None:
    for line in reversed(text.splitlines()):
        if line.startswith("Perf:"):
            try:
                return float(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def measure_arm(label: str, cache_root: Path) -> float:
    src, steps, extra = ARM_SPECS[label]
    spec_name = "geak" if label == "geak@2" else label
    env = base_env()
    env.update(extra)
    env["FWE_STEPS"] = str(steps)
    env["FWE_SRC"] = str(src)
    env["FWE_CONFIG_DIR"] = str(CONFIGS_BASELINE)
    if label in FRESH_CACHE:
        cache = cache_root / label.replace("@", "_")
        (cache / "inductor").mkdir(parents=True, exist_ok=True)
        (cache / "triton").mkdir(parents=True, exist_ok=True)
        env["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
        env["TRITON_CACHE_DIR"] = str(cache / "triton")
    # task_runner keys the compile-cache arm on the spec name, not the label.
    if spec_name == "geak":
        env["FASTWAM_KVSLAB"] = "0"
    proc = subprocess.run(
        [sys.executable, str(TASK_RUNNER), "performance"],
        env=env, capture_output=True, text=True, timeout=7200,
    )
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0 or "PERF_DONE" not in proc.stdout:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-80:])
        sys.exit(f"[{label}] performance failed (rc={proc.returncode})\n{tail}")
    value = perf_line(proc.stdout)
    if value is None:
        sys.exit(f"[{label}] no Perf line")
    return value


def write_csv(path: Path, rows: list[tuple[str, float]]) -> None:
    lines = ["model,performance,metric"]
    for name, value in rows:
        lines.append(f"{name},{value:.4f},ms")
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def run_interleaved(rounds: int, csv_path: Path) -> None:
    samples: dict[str, list[float]] = {label: [] for label in INTERLEAVED}
    with tempfile.TemporaryDirectory(prefix="fam_cache_") as tmp:
        cache_root = Path(tmp)
        for round_index in range(rounds):
            for label in INTERLEAVED:
                value = measure_arm(label, cache_root)
                samples[label].append(value)
                print(f"  round {round_index + 1} {label:<14} {value:.3f} ms", flush=True)
    rows = []
    print(f"\n{'arm':<14}{'steps':>6}{'median_ms':>12}{'spread':>9}")
    for label in INTERLEAVED:
        values = samples[label]
        median = statistics.median(values)
        spread = (max(values) - min(values)) / min(values) * 100 if len(values) > 1 else 0.0
        _, steps, _ = ARM_SPECS[label]
        print(f"{label:<14}{steps:>6}{median:>12.3f}{spread:>8.2f}%")
        rows.append((label, median))
    write_csv(csv_path, rows)


def run_one(label: str) -> None:
    if label not in ARM_SPECS:
        sys.exit(f"unknown arm {label}")
    with tempfile.TemporaryDirectory(prefix="fam_cache_") as tmp:
        value = measure_arm(label, Path(tmp))
    print(f"performance: {value:.4f} ms")


def run_cae(steps: int) -> None:
    env = base_env()
    env["PYTHONPATH"] = str(CAE_SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["ACTION_VAE_CKPT"] = require("ACTION_VAE_CKPT")
    env["MODEL_CONFIGS"] = str(CONFIGS_CAE)
    env["TASK_CONFIG"] = CAE_TASK
    proc = subprocess.run(
        [sys.executable, str(CAE_WORKER), "--steps", str(steps),
         "--compile-infer", "1", "--expect-src", str(CAE_SRC)],
        env=env, capture_output=True, text=True, timeout=7200,
    )
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-80:])
        sys.exit(f"cae latency failed (rc={proc.returncode})\n{tail}")
    value = perf_line(proc.stdout)
    if value is None:
        sys.exit("cae latency produced no Perf line")
    print(f"performance: {value:.4f} ms")


def instinct_env(src: Path, extra: dict) -> dict:
    env = base_env()
    env.update(extra)
    sim = Path(os.environ.get("INSTINCT_SIM_BUILD", str(FAM / "simulate" / "build-libero")))
    hp = FAM / "simulate" / "python"
    libero = os.environ.get("LIBERO_ROOT", "/opt/LIBERO")
    parts = [
        str(hp / "instinct_sim_libero" / "sitecustomize"),
        str(src), str(hp), str(sim / "python"), libero,
    ]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env["INSTINCT_SIM_BUILD"] = str(sim)
    env["HP_LIBERO_BACKEND"] = "hip"
    env["HSA_ENABLE_SDMA"] = "0"
    env["GPU_MAX_HW_QUEUES"] = "2"
    env["HP_GPU_LABEL"] = "0"
    env["HP_SOFT_FIRST_RESET"] = "1"
    env["MUJOCO_GL"] = "osmesa"
    env["PYOPENGL_PLATFORM"] = "osmesa"
    env["LIBERO_CONFIG_PATH"] = os.environ.get("LIBERO_CONFIG_PATH", "/opt/libero_config")
    rocm_lib = ""
    try:
        import _rocm_sdk_core
        rocm_lib = str(Path(_rocm_sdk_core.__file__).parent / "lib")
    except Exception:
        rocm_lib = "/opt/rocm/lib"
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        p for p in (str(sim), rocm_lib, env.get("LD_LIBRARY_PATH", "")) if p
    )
    return env


def pooled_success(out: Path) -> tuple[int, int]:
    successes = episodes = 0
    for path in out.glob("*/*_results.json"):
        payload = json.loads(path.read_text())
        successes += int(payload.get("successes", 0))
        episodes += int(payload.get("total_episodes", 0))
    return successes, episodes


def run_smoke(label: str) -> None:
    if label not in ("eager", "geak@2"):
        sys.exit(f"smoke arm must be eager or geak@2, got {label}")
    src, steps, extra = ARM_SPECS[label]
    env = instinct_env(src, extra)
    tasks = ",".join(f"{suite}:{task}" for suite in SUITES for task in range(10))
    out = Path(tempfile.mkdtemp(prefix="fam_smoke_"))
    proc = subprocess.run(
        [sys.executable, str(BATCHED),
         "--config-dir", str(CONFIGS_BASELINE),
         "--task", "libero_uncond_2cam224_1e-4",
         "--ckpt", env["MODEL_CKPT"],
         "--dataset-stats", require("DATASET_STATS"),
         "--steps", str(steps),
         "--tasks", tasks,
         "--output-dir", str(out),
         "--trials", "1",
         "--num-envs", "1",
         "--pad-batch", "1",
         "--override", "model.redirect_common_files=false"],
        env=env, timeout=14400,
    )
    successes, episodes = pooled_success(out)
    print(f"smoke {label}: {successes}/{episodes} episodes under {out}")
    if proc.returncode != 0 or episodes == 0:
        sys.exit(f"smoke failed (rc={proc.returncode}, episodes={episodes})")
    rate = successes / episodes
    print(f"performance: {rate:.4f} success_rate")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=("interleaved", "arm", "smoke", "cae"))
    parser.add_argument("--arm", default="")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--csv", default="perf_fam_latency.csv")
    parser.add_argument("--cae-steps", type=int, default=10)
    args = parser.parse_args()
    if args.mode == "interleaved":
        run_interleaved(args.rounds, Path(args.csv).resolve())
    elif args.mode == "arm":
        run_one(args.arm)
    elif args.mode == "smoke":
        run_smoke(args.arm)
    else:
        run_cae(args.cae_steps)


if __name__ == "__main__":
    main()
