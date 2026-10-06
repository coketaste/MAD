#!/usr/bin/env python3
"""Summarize per-step performance and trace footprint for every archived blog run.

Reads blog_results/<gpu>/<model>/<run>/ (written by run_matrix.py) and emits
blog_results/<gpu>/summary.csv with one row per run:

  - steady-state step time (median over steps >= WARMUP, excluding the Primus
    profiler capture window for P1 runs) and the capture-window step time
  - throughput relative to the mean of the two baselines (B0a, B0b), and the
    baseline noise band |B0a - B0b| / mean
  - end-of-run KPIs from primus_perf_output.csv (tokens/s, TFLOP/s, MFU)
  - on-disk size of each lens's artifacts

Step time is parsed from the Primus training log, so every lens is compared on
the same metric the framework itself reports.
"""

import csv
import json
import os
import re
import statistics
import sys
from pathlib import Path

MAD_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = Path(os.environ.get("BLOG_RESULTS_ROOT", "/data/ysha/primus_blog_results"))
RESULTS = RESULTS_ROOT / (sys.argv[1] if len(sys.argv) > 1 else "mi300x")
WARMUP = 5               # steps excluded from steady state (compile/autotune/allocator warmup)
CAPTURE = range(9, 14)   # P1 capture window (steps 10-12) plus one step on each side
RUNS = {"B0a", "B0b", "P1", "K2", "K2s", "R3", "KT", "PT", "PS", "PC"}
WINDOWED = {"P1", "PT", "PS", "PC"}   # PyTorch profiler active on steps 10-11

ANSI = re.compile(r"\x1b\[[0-9;]*m")
# Megatron-LM / Megatron-Bridge / FLUX: "iteration   12/   30 | ... elapsed time per iteration (ms): 1234.5/..."
MEGATRON = re.compile(r"iteration\s+(\d+)/\s*\d+\s*\|.*?elapsed time per iteration \(ms\):\s*([\d.]+)")
# TorchTitan: "step: 12  loss: ... tps: 3,953 ..."; step time derived from tps
TITAN = re.compile(r"step:\s*(\d+)\s.*?tps:\s*([\d,]+)")
# MaxDiffusion: "completed step: 12, seconds: 1.234, ..."
MAXDIFF = re.compile(r"completed step:\s*(\d+),\s*seconds:\s*([\d.]+)")

TRACE_DIRS = {
    "primus_traces": ["run_directory/output/**/tensorboard", "run_directory/outputs/profile_traces",
                      "run_directory/output/**/profile_traces", "run_directory/output/**/plugins/profile"],
    "rtl": ["rocm_trace_lite_output", "run_directory/rocm_trace_lite_output"],
    "rocprofv3": ["rocprof_output", "run_directory/rocprof_output"],
}


def step_series(run_dir: Path) -> dict:
    """Return {step: seconds} from the first training log that matches a known format."""
    logs = sorted((run_dir / "run_directory").glob("output/log_mp_*.txt"))
    logs += sorted(run_dir.glob("*.run.live.log"))
    # Megatron logs training_log on the last rank only; the LLM path forwards it to rank 0,
    # but the FLUX trainer does not, so fall back to the highest rank's debug log.
    rank_logs = list((run_dir / "run_directory").glob("output/**/logs/*/rank-*/debug.log"))
    logs += sorted(rank_logs, key=lambda p: int(p.parent.name.split("-")[1]), reverse=True)[:1]
    for log in logs:
        text = ANSI.sub("", log.read_text(errors="replace"))
        out = {int(i): float(ms) / 1000.0 for i, ms in MEGATRON.findall(text)}
        if out:
            return out
        out = {int(i): float(s) for i, s in MAXDIFF.findall(text)}
        if out:
            return out
        tps = {int(i): float(t.replace(",", "")) for i, t in TITAN.findall(text)}
        if tps:
            # Constant tokens per step, so 1/tps is proportional to step time.
            return {i: 1.0 / t for i, t in tps.items() if t > 0}
    return {}


def dir_size(run_dir: Path, patterns) -> int:
    # madengine copies collector output out of run_directory, so the same trace can exist
    # under several patterns; report the largest location rather than the sum.
    sizes = [0]
    for pat in patterns:
        for d in run_dir.glob(pat):
            if d.is_dir():
                sizes.append(sum(f.stat().st_size for f in d.rglob("*") if f.is_file()))
    return max(sizes)


def kpis(run_dir: Path) -> dict:
    f = run_dir / "primus_perf_output.csv"
    if not f.exists():
        return {}
    out = {}
    for row in csv.DictReader(f.open()):
        try:
            out[row["metric"]] = float(row["performance"])
        except (KeyError, ValueError):
            pass
    return out


def main() -> None:
    rows = []
    for model_dir in sorted(p for p in RESULTS.iterdir() if p.is_dir()):
        # Superseded attempts are kept on disk under a suffixed name (e.g. P1_stopped_disk).
        runs = {p.name: p for p in model_dir.iterdir() if p.is_dir() and p.name in RUNS}
        series = {name: step_series(p) for name, p in runs.items()}

        def steady(name):
            s = series.get(name) or {}
            vals = [v for k, v in s.items() if k >= WARMUP and not (name in WINDOWED and k in CAPTURE)]
            return statistics.median(vals) if vals else None

        base = [steady(b) for b in ("B0a", "B0b") if steady(b)]
        base_mean = statistics.mean(base) if base else None
        noise = (abs(base[0] - base[1]) / base_mean) if len(base) == 2 else None
        for name, p in sorted(runs.items()):
            status = json.loads((p / "status.json").read_text()) if (p / "status.json").exists() else {}
            med = steady(name)
            s = series.get(name) or {}
            window = [v for k, v in s.items() if k in range(10, 13)]
            k = kpis(p)
            rows.append({
                "model": model_dir.name, "run": name, "rc": status.get("rc"),
                "wall_min": round(status.get("wall_s", 0) / 60, 1),
                "steps_parsed": len(s),
                # Significant figures: TorchTitan's 1/tps proxy is ~1e-4.
                "steady_step_s": f"{med:.4g}" if med else "",
                "rel_throughput": round(base_mean / med, 4) if med and base_mean else "",
                "baseline_noise": round(noise, 4) if noise is not None else "",
                "capture_window_step_s": f"{statistics.median(window):.4g}" if name in WINDOWED and window else "",
                "tokens_per_second": k.get("tokens_per_second", ""),
                "tflops": k.get("tflops", ""),
                "mfu": k.get("model_flops_utilization", ""),
                **{f"{lens}_MB": round(dir_size(p, pats) / 2**20, 1) for lens, pats in TRACE_DIRS.items()},
            })
    out = RESULTS / "summary.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
