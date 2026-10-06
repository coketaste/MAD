#!/usr/bin/env python3
"""Validate every profiling option used by the blog on a small model before the real runs.

Runs madengine's `dummy_profiling` fixture (bf16 GEMM MLP + DDP all-reduce on all GPUs,
~30 steps) once per profiling option, on the same image runtime as the Primus models,
then checks what each collector actually recorded:

  - torch.profiler (framework lens): one Kineto trace per rank, with GPU kernels and RCCL
  - rocm-trace-lite: kernel ops on real GPU ids (not just roctx markers), RCCL in default mode
  - rocprofv3: output inside run_directory, kernel dispatches, RCCL activity
  - TraceLens: non-empty reports, in-container (post-script) and on the host

The fixture launches from a cwd outside run_directory, as Primus run.sh does, so tools that
write relative paths are caught.

Usage:
    python3 docs/blog/runkit/validate_profilers.py                 # all cases
    python3 docs/blog/runkit/validate_profilers.py --cases rtl_lite,rocprof_light
    python3 docs/blog/runkit/validate_profilers.py --check-only <run-dir>   # re-check a run

Environment:
    MADENGINE_SRC     madengine checkout providing the fixture (default ~/codebase/madengine)
    VALIDATE_IMAGE    image to run the fixture in (default: the built Primus FLUX image,
                      i.e. rocm/primus:v26.7 runtime)
    TRACELENS_PYTHON  host interpreter with TraceLens (default ~/.venvs/tracelens/bin/python)
"""

import argparse
import csv
import glob
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

MAD_ROOT = Path(__file__).resolve().parents[3]
MADENGINE_SRC = Path(os.environ.get("MADENGINE_SRC", Path.home() / "codebase" / "madengine"))
FIXTURE = MADENGINE_SRC / "tests" / "fixtures" / "dummy"
BASE = Path(os.environ.get("BLOG_RESULTS_ROOT", "/data/ysha/primus_blog_results")) / "profiler-validate"
OUT = BASE  # per-invocation directory, set in main()
WORKSPACE = BASE
IMAGE = os.environ.get(
    "VALIDATE_IMAGE", "ci-primus_train_megatron_mi300x_flux_535m_pretrain_primus.ubuntu.amd:latest"
)
TAG = "dummy_profiling"
ARCHIVE_IMAGE = "python:3.12-slim"
# Host interpreter with TraceLens (pinned ref from madengine's `tracelens` extra).
TRACELENS_PYTHON = os.environ.get("TRACELENS_PYTHON", str(Path.home() / ".venvs" / "tracelens" / "bin" / "python"))
RCCL_MARKERS = (b"ncclDevKernel", b"ncclKernel", b"rccl", b"nccl:all_reduce", b"AllReduce")

TORCH_ENV = {"DUMMY_TORCH_PROFILE": "1"}

# name: (tools, extra docker_env_vars, checks)
CASES = {
    "baseline": ([], {}, ["perf"]),
    "torch_profiler": ([], TORCH_ENV, ["perf", "kineto"]),
    "torch_profiler_tracelens": ([{"name": "tracelens_pytorch"}], TORCH_ENV, ["perf", "kineto", "tracelens"]),
    "torch_profiler_collective": ([{"name": "tracelens_collective"}], TORCH_ENV, ["perf", "kineto", "tracelens"]),
    "rtl_lite": ([{"name": "rocm_trace_lite"}], {}, ["perf", "rtl"]),
    "rtl_default": ([{"name": "rocm_trace_lite_default"}], {}, ["perf", "rtl", "rtl_rccl"]),
    "rocprof_light": ([{"name": "rocprofv3_lightweight"}], {}, ["perf", "rocprof", "rocprof_kernels"]),
    "rocprof_light_tracelens": (
        [{"name": "rocprofv3_lightweight"}, {"name": "tracelens_rocprof"}], {},
        ["perf", "rocprof", "rocprof_kernels", "tracelens"]),
    "rocprof_comm": ([{"name": "rocprofv3_communication"}], {}, ["perf", "rocprof", "rocprof_rccl"]),
    "rocprof_comm_tracelens": (
        [{"name": "rocprofv3_communication"}, {"name": "tracelens_pftrace"}], {},
        ["perf", "rocprof", "rocprof_rccl", "tracelens"]),
    "rocprof_perfetto": ([{"name": "rocprofv3_perfetto"}], {}, ["perf", "rocprof", "rocprof_kernels"]),
    # v3 kernel lens: the exact command run_matrix.py uses for KT (no --hip-trace).
    "kernel_trace": (
        [{"name": "rocprofv3_lightweight", "cmd": "bash ../scripts/common/tools/rocprof_wrapper.sh "
          "--kernel-trace --memory-copy-trace --output-format json -d ./rocprof_output --"}],
        {}, ["perf", "rocprof", "rocprof_kernels", "rocprof_rccl", "rocprof_no_host_api"]),
    # Overhead probes for the kernel lens (compare perf with `baseline`): output format,
    # and rocprofv3 loaded with only roctx marker tracing (no kernel interception).
    "kernel_trace_csv": (
        [{"name": "rocprofv3_lightweight", "cmd": "bash ../scripts/common/tools/rocprof_wrapper.sh "
          "--kernel-trace --output-format csv -d ./rocprof_output --"}],
        {}, ["perf", "rocprof"]),
    "rocprof_marker_only": (
        [{"name": "rocprofv3_lightweight", "cmd": "bash ../scripts/common/tools/rocprof_wrapper.sh "
          "--marker-trace --output-format csv -d ./rocprof_output --"}],
        {}, ["perf"]),
    # Control: same collector launched from run_directory, isolating cwd effects.
    "rocprof_light_in_run_dir": (
        [{"name": "rocprofv3_lightweight"}], {"DUMMY_PROF_STAY_IN_RUN_DIR": "1"},
        ["perf", "rocprof", "rocprof_kernels"]),
    # A run that fails after training (as Primus FLUX does in its perf extractor) must
    # still get its traces collected and analyzed.
    "failed_run_tracelens": (
        [{"name": "rocprofv3_lightweight"}, {"name": "tracelens_rocprof"}], {"DUMMY_PROF_EXIT_CODE": "1"},
        ["rocprof", "rocprof_kernels", "tracelens"]),
}
EXPECT_FAILURE = {"failed_run_tracelens"}


def setup_workspace() -> set:
    """Copy the fixture into a fresh scratch MAD root; return its top-level entries."""
    shutil.copytree(FIXTURE, WORKSPACE)
    return {p.name for p in WORKSPACE.iterdir()}


def archive(dest: Path, fixture_entries: set) -> None:
    """Move everything the run created in the workspace into dest, owned by the user."""
    names = [p.name for p in WORKSPACE.iterdir() if p.name not in fixture_entries]
    rel = dest.relative_to(OUT)
    script = "cd /o/workspace && " + " && ".join(f"mv -- '{n}' /o/{rel}/" for n in names) if names else "true"
    script += f" && chown -R {os.getuid()}:{os.getgid()} /o/{rel}"
    subprocess.run(["docker", "run", "--rm", "-v", f"{OUT}:/o", ARCHIVE_IMAGE, "sh", "-c", script], check=True)


def run_case(name: str, fixture_entries: set) -> dict:
    tools, env, _ = CASES[name]
    dest = OUT / name
    dest.mkdir(parents=True)
    ctx = {
        "gpu_vendor": "AMD",
        "guest_os": "UBUNTU",
        # Run on the existing image (no build): madengine builds with --pull, which
        # cannot resolve a locally built base image.
        "MAD_CONTAINER_IMAGE": IMAGE,
        "docker_env_vars": dict(env),
    }
    if tools:
        ctx["tools"] = tools
    (dest / "additional_context.json").write_text(json.dumps(ctx, indent=2) + "\n")
    cmd = ["madengine", "run", "--tags", TAG, "--additional-context", json.dumps(ctx),
           "--keep-model-dir", "--timeout", "1800", "-o", "perf.csv"]
    print(f"\n=== {name}\n{' '.join(cmd)}", flush=True)
    t0 = time.time()
    with open(dest / "madengine.log", "w") as log:
        rc = subprocess.run(cmd, cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT).returncode
    archive(dest, fixture_entries)
    status = {"case": name, "rc": rc, "wall_s": round(time.time() - t0, 1)}
    (dest / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


# ---------------------------------------------------------------- checks

def files(d: Path, *patterns) -> list:
    out = []
    for pat in patterns:
        out += [Path(p) for p in glob.glob(str(d / "**" / pat), recursive=True) if Path(p).is_file()]
    return sorted(set(out))


def contains(paths, needles) -> bool:
    for p in paths:
        data = p.read_bytes()
        if p.suffix == ".gz":
            import gzip
            data = gzip.decompress(data)
        if any(n in data for n in needles):
            return True
    return False


def check_perf(d):
    rows = list(csv.DictReader(open(d / "perf.csv"))) if (d / "perf.csv").exists() else []
    ok = [r for r in rows if r.get("model", "").endswith(TAG) and r.get("performance")]
    if not ok:
        return False, "no performance row in perf.csv"
    return True, f"{ok[-1]['performance']} {ok[-1].get('metric', '')} status={ok[-1].get('status', '')}"


def check_kineto(d):
    traces = files(d, "*.pt.trace.json", "*.pt.trace.json.gz")
    if len(traces) < 2:
        return False, f"{len(traces)} Kineto traces"
    kernels = contains(traces[:1], (b'"cat": "kernel"', b'"cat":"kernel"'))
    rccl = contains(traces[:1], RCCL_MARKERS)
    return kernels and rccl, f"{len(traces)} traces, GPU kernels={kernels}, RCCL={rccl}"


def rtl_stats(d):
    dbs = files(d, "trace.db")
    if not dbs:
        return None, "no trace.db"
    c = sqlite3.connect(dbs[0])
    types = dict(c.execute("select s.string, count(*) from rocpd_op o join rocpd_string s "
                           "on o.opType_id=s.id group by 1").fetchall())
    gpus = [g for (g,) in c.execute("select distinct gpuId from rocpd_op where gpuId >= 0")]
    rccl = c.execute("select count(*) from rocpd_op o join rocpd_string s on o.description_id=s.id "
                     "where s.string like '%ccl%'").fetchone()[0]
    return {"types": types, "gpus": gpus, "rccl": rccl}, ""


def check_rtl(d):
    st, err = rtl_stats(d)
    if st is None:
        return False, err
    kernel_ops = sum(n for t, n in st["types"].items() if t != "UserMarker")
    return kernel_ops > 0 and len(st["gpus"]) > 0, \
        f"op types={st['types']}, GPUs with kernels={len(st['gpus'])}"


def check_rtl_rccl(d):
    st, err = rtl_stats(d)
    if st is None:
        return False, err
    return st["rccl"] > 0, f"RCCL ops={st['rccl']}"


def under(d, p, part) -> bool:
    """Whether a path component below the case dir contains `part` (not the case name)."""
    return any(part in seg for seg in p.relative_to(d).parts[:-1])


def rocprof_files(d):
    return [p for p in files(d, "*.json", "*.pftrace", "*.csv", "*.db")
            if under(d, p, "rocprof") and not under(d, p, "tracelens")]


def check_rocprof(d):
    fs = rocprof_files(d)
    if not fs:
        return False, "no rocprofv3 output under the run (written outside run_directory?)"
    return True, f"{len(fs)} files, {sum(p.stat().st_size for p in fs) / 2**20:.1f} MB"


def check_rocprof_kernels(d):
    fs = rocprof_files(d)
    ok = contains(fs, (b"kernel_dispatch", b"KERNEL_DISPATCH", b"Kernel_Name", b"kernel_name"))
    return ok, f"kernel dispatch records={ok}"


def check_rocprof_rccl(d):
    ok = contains(rocprof_files(d), RCCL_MARKERS)
    return ok, f"RCCL activity={ok}"


def check_rocprof_no_host_api(d):
    # The kernel lens must record GPU dispatches on real agents and no host API calls.
    counts = {}
    for p in rocprof_files(d):
        if p.suffix != ".json":
            continue
        tool = json.loads(p.read_bytes().decode("utf-8", "replace"))["rocprofiler-sdk-tool"][0]
        for k, v in tool["buffer_records"].items():
            counts[k] = counts.get(k, 0) + len(v)
        agents = {str(r["dispatch_info"]["agent_id"]) for r in tool["buffer_records"]["kernel_dispatch"]}
        counts["gpu_agents"] = counts.get("gpu_agents", 0) + len(agents)
    host = sum(v for k, v in counts.items() if k.endswith("_api"))
    ok = counts.get("kernel_dispatch", 0) > 0 and host == 0
    return ok, f"{ {k: v for k, v in counts.items() if v} }"


def check_tracelens(d):
    # tracelens_summary.* is written even when no trace was found; it is not a report.
    reports = [p for p in files(d, "*.xlsx", "*.csv") if under(d, p, "tracelens")
               and not p.name.startswith("tracelens_summary") and p.stat().st_size > 0]
    return bool(reports), f"{len(reports)} TraceLens report files"


CHECKS = {
    "perf": check_perf, "kineto": check_kineto, "rtl": check_rtl, "rtl_rccl": check_rtl_rccl,
    "rocprof": check_rocprof, "rocprof_kernels": check_rocprof_kernels,
    "rocprof_rccl": check_rocprof_rccl, "tracelens": check_tracelens,
    "rocprof_no_host_api": check_rocprof_no_host_api,
}


def evaluate(name: str) -> dict:
    d = OUT / name
    status = json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {}
    result = {"case": name, "rc": status.get("rc"), "wall_s": status.get("wall_s"), "checks": {}}
    for chk in CASES[name][2]:
        try:
            ok, detail = CHECKS[chk](d)
        except Exception as e:  # a broken artifact is a failed check, not a crash
            ok, detail = False, f"error: {e}"
        result["checks"][chk] = {"ok": ok, "detail": detail}
    rc_ok = result["rc"] not in (0, None) if name in EXPECT_FAILURE else result["rc"] == 0
    result["pass"] = rc_ok and all(c["ok"] for c in result["checks"].values())
    return result


def host_tracelens() -> dict:
    """Run `madengine report tracelens` on the host against the framework-lens traces."""
    src = OUT / "torch_profiler"
    out = OUT / "host_tracelens"
    res = {}
    for mode in ("pytorch", "collective"):
        cmd = ["madengine", "report", "tracelens", "--root", str(src), "--mode", mode,
               "--output-dir", str(out / mode), "--gpu-arch", "MI300X", "--python", TRACELENS_PYTHON]
        p = subprocess.run(cmd, capture_output=True, text=True)
        (OUT / f"host_tracelens_{mode}.log").write_text(p.stdout + p.stderr)
        reports = [f for f in files(out / mode, "*.xlsx", "*.csv")
                   if f.stat().st_size > 0] if (out / mode).exists() else []
        res[mode] = {"ok": p.returncode == 0 and bool(reports), "detail": f"rc={p.returncode}, {len(reports)} reports"}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="all")
    ap.add_argument("--check-only", metavar="RUN_DIR", help="re-check an earlier run directory")
    ap.add_argument("--skip-host", action="store_true")
    a = ap.parse_args()
    names = list(CASES) if a.cases == "all" else a.cases.split(",")
    global OUT, WORKSPACE
    if a.check_only:
        OUT = Path(a.check_only).resolve()
    else:
        OUT = BASE / time.strftime("run-%Y%m%d-%H%M%S")
        WORKSPACE = OUT / "workspace"
        OUT.mkdir(parents=True)
        print(f"results: {OUT}", flush=True)
        entries = setup_workspace()
        for n in names:
            run_case(n, entries)
    results = [evaluate(n) for n in names]
    if not a.skip_host and "torch_profiler" in names:
        results.append({"case": "host_report_tracelens", "rc": 0, "checks": host_tracelens()})
        results[-1]["pass"] = all(c["ok"] for c in results[-1]["checks"].values())
    (OUT / "validation.json").write_text(json.dumps(results, indent=2) + "\n")
    print()
    for r in results:
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {r['case']:<28} rc={r['rc']}")
        for k, c in r["checks"].items():
            print(f"        {'ok ' if c['ok'] else 'BAD'} {k:<16} {c['detail']}")
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
