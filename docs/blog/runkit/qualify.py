#!/usr/bin/env python3
"""Write <model_dir>/qualification.md: gates G1-G7 for one model's v3 runs
(B0a/B0b baselines; PT, PS on rank 0; PC on all ranks; KT optional).

Usage: qualify.py /data/ysha/primus_blog_results/mi300x-v3/D_flux_535m-megatron_diffusion
Run crop_rocprof.py on KT first (its *_steps10-11_results.json files are read here).
"""

import collections
import glob
import gzip
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze_runs import step_series  # noqa: E402

NOISE_MAX, DRIFT_MIN, KT_PUB, KT_MAX, PT_MAX = 0.02, 0.02, 1.10, 1.30, 1.30
FIDELITY, COMPUTE_TOL = 0.05, 0.03   # G6: per roofline kernel, and total GPU compute per step
WARMUP, CAPTURE, WINDOW = 5, range(9, 14), (10, 11)
COMM = re.compile(r"nccl|rccl", re.I)
# Roofline inputs: GEMM (hipBLASLt/Tensile/CK) and attention kernels.
ROOFLINE = re.compile(r"Cijk_|gemm|matmul|fmha|flash|attn|attention|mha_", re.I)


def norm(name):
    return re.sub(r"\(.*", "", name).strip()


def steady(series, windowed):
    vals = [v for k, v in series.items() if k >= WARMUP and not (windowed and k in CAPTURE)]
    return statistics.median(vals) if vals else None


def kernel_medians_kineto(ev):
    d = collections.defaultdict(list)
    for e in ev:
        if e.get("cat") == "kernel":
            d[norm(e["name"])].append(e["dur"])          # us
    return {k: statistics.median(v) for k, v in d.items()}, {k: sum(v) for k, v in d.items()}


def compute_us_per_step(ev):
    """Union of non-collective kernel intervals, per active profiler step (TraceLens's
    computation_time). Collectives are excluded: their duration includes waiting on peers."""
    iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev
                if e.get("cat") == "kernel" and not COMM.search(e["name"]))
    total, end = 0.0, float("-inf")
    for s, e in iv:
        if e > end:
            total += e - max(s, end)
            end = e
    steps = {e["name"] for e in ev if e.get("cat") == "user_annotation" and str(e.get("name", "")).startswith("ProfilerStep#")}
    return total / max(len(steps), 1)


def kernel_medians_rocprof(path):
    doc = json.loads(open(path, "rb").read().decode("utf-8", "replace"))["rocprofiler-sdk-tool"][0]
    names = {s["kernel_id"]: s.get("truncated_kernel_name") or s["kernel_name"] for s in doc["kernel_symbols"]}
    d = collections.defaultdict(list)
    for r in doc["buffer_records"]["kernel_dispatch"]:
        d[norm(names.get(r["dispatch_info"]["kernel_id"], ""))].append((r["end_timestamp"] - r["start_timestamp"]) / 1e3)
    return {k: statistics.median(v) for k, v in d.items()}


def rank0_events(paths):
    """Events of the rank-0 trace: Megatron names it rank[0]; Megatron-Bridge uses
    <host>_<pid>, so fall back to Kineto's distributedInfo."""
    named = [p for p in paths if "rank[0]" in p]
    for p in named or paths:
        with (gzip.open(p, "rt") if p.endswith(".gz") else open(p)) as f:
            doc = json.load(f)
        if named or doc.get("distributedInfo", {}).get("rank", 0) == 0:
            return doc["traceEvents"]
    return []


def traces(run_dir):
    pats = ["run_directory/output/**/tensorboard/*.pt.trace.json*"]
    return sorted(p for pat in pats for p in glob.glob(str(run_dir / pat), recursive=True))


def main():
    mdir = Path(sys.argv[1])
    runs = {p.name: p for p in mdir.iterdir() if p.is_dir()}
    series = {n: step_series(p) for n, p in runs.items()}
    rows, notes = [], []

    base = [steady(series[b], False) for b in ("B0a", "B0b") if series.get(b)]
    base_mean = statistics.mean(base) if base else None
    noise = abs(base[0] - base[1]) / base_mean if len(base) == 2 else None
    rows.append(("G1 baseline noise", f"{noise:.2%}" if noise is not None else "n/a",
                 "PASS" if noise is not None and noise <= NOISE_MAX else "FAIL"))
    drift_tol = max(noise or 0, DRIFT_MIN)

    for name in ("KT", "PT", "PS", "PC"):
        if name not in runs:
            continue
        p, s = runs[name], series.get(name) or {}
        status = json.loads((p / "status.json").read_text()) if (p / "status.json").exists() else {}
        rows.append((f"G2 {name} health", f"rc={status.get('rc')} steps={len(s)}",
                     "PASS" if s and status.get("rc") in (0, 3) else "FAIL"))
        windowed = name != "KT"
        med = steady(s, windowed)
        if med and base_mean:
            if windowed:
                drift = med / base_mean - 1
                rows.append((f"G3 {name} drift outside window", f"{drift:+.2%}", "PASS" if abs(drift) <= drift_tol else "FAIL"))
                win = [s[k] for k in WINDOW if k in s]
                ratio = statistics.median(win) / base_mean if win else None
                if name in ("PT", "PS") and ratio:
                    rows.append((f"G5 {name} window overhead", f"{ratio:.2f}x", "PASS" if ratio <= PT_MAX else "SHARES-ONLY"))
                if name == "PC" and ratio:
                    rows.append(("PC window overhead (info)", f"{ratio:.2f}x", "-"))
            else:
                ratio = med / base_mean
                verdict = "PASS" if ratio <= KT_PUB else ("KERNELS-ONLY" if ratio <= KT_MAX else "FAIL")
                rows.append(("G4 KT overhead", f"{ratio:.2f}x", verdict))

    kt_crops = sorted(glob.glob(str(runs["KT"] / "run_directory/rocprof_output/*/*_steps10-11_results.json"))) if "KT" in runs else []
    if "KT" in runs:
        rows.append(("G7 KT cropped rank files", str(len(kt_crops)), "PASS" if len(kt_crops) == 8 else "FAIL"))
    # PT and PS profile rank 0 only; PC profiles every rank for the collective report.
    rank0 = {}
    for name, want in (("PT", 1), ("PS", 1), ("PC", 8)):
        if name not in runs:
            continue
        tr = traces(runs[name])
        rank0[name] = ev0 = rank0_events(tr)
        steps = {e["name"] for e in ev0 if str(e.get("name", "")).startswith("ProfilerStep#")}
        coll = sum("Collective name" in e.get("args", {}) for e in ev0)
        ok = len(tr) == want and {"ProfilerStep#10", "ProfilerStep#11"} <= steps and coll > 0
        rows.append((f"G7 {name} traces / steps / collectives", f"{len(tr)} / {sorted(steps)} / {coll}",
                     "PASS" if ok else "FAIL"))
    # The op-view run: PS since the PT/PS merge, PT for models profiled before it.
    view = "PS" if "PS" in rank0 else ("PT" if "PT" in rank0 else None)
    if view and base_mean:
        # Idle % from a profiled window is inflated by the profiler; publish GPU compute per
        # step (stable under overhead) against the unprofiled baseline step instead.
        c = compute_us_per_step(rank0[view]) / 1e6
        rows.append((f"GPU compute busy (derived: {view} compute / baseline step)",
                     f"{c * 1e3:.2f} ms / {base_mean * 1e3:.2f} ms = {c / base_mean:.1%}", "-"))
    if "PS" in runs:
        ev = rank0["PS"]
        cpu = [e for e in ev if e.get("cat") == "cpu_op"]
        frac = sum("Input Dims" in e.get("args", {}) for e in cpu) / max(len(cpu), 1)
        rows.append(("G7 PS shapes recorded", f"{frac:.0%} of {len(cpu)} cpu ops", "PASS" if frac >= 0.5 else "FAIL"))
        # Reference for PS kernel durations: an independent run of the same kernels. KT when
        # it exists (rank file with the most kernel time; rocprofv3 names files by pid), else
        # PC's rank 0 (8-rank window, ~7x), else PT's rank 0.
        ref_name, ref_med = None, None
        if kt_crops:
            ref_name, ref_med = "KT", kernel_medians_rocprof(max(kt_crops, key=lambda f: Path(f).stat().st_size))
        else:
            ref_name = next((n for n in ("PC", "PT") if rank0.get(n)), None)
            ref_med = kernel_medians_kineto(rank0[ref_name])[0] if ref_name else None
        if ref_med and ev:
            ps_med, ps_tot = kernel_medians_kineto(ev)
            # Only the roofline inputs (GEMM, attention) need per-kernel agreement; tiny
            # elementwise kernels swing by tenths of a microsecond and are covered by G6b.
            top = [k for k, _ in sorted(ps_tot.items(), key=lambda kv: -kv[1])
                   if k in ref_med and ROOFLINE.search(k)][:10]
            dev = {k: ps_med[k] / ref_med[k] - 1 for k in top}
            worst = max(dev.values(), key=abs) if dev else None
            rows.append((f"G6a PS vs {ref_name} GEMM/attention kernel medians",
                         f"worst {worst:+.1%} over {len(dev)} kernels" if dev else "no overlap",
                         "PASS" if dev and abs(worst) <= FIDELITY else "FAIL"))
            notes += [f"- `{k[:90]}`: PS {ps_med[k]:.1f} us vs {ref_name} {ref_med[k]:.1f} us ({v:+.1%})"
                      for k, v in dev.items()]
            if ref_name in ("PC", "PT"):
                ps_c, ref_c = compute_us_per_step(ev), compute_us_per_step(rank0[ref_name])
                d = ps_c / ref_c - 1
                rows.append((f"G6b PS vs {ref_name} GPU compute per step",
                             f"{ps_c / 1e3:.2f} vs {ref_c / 1e3:.2f} ms ({d:+.1%})",
                             "PASS" if abs(d) <= COMPUTE_TOL else "FAIL"))

    out = [f"# Qualification: {mdir.name}", "", f"Baseline step: {base_mean:.4g} s" if base_mean else "Baseline: missing", "",
           "| Gate | Value | Verdict |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in rows]
    if notes:
        out += ["", "## Kernel agreement (G6a)", ""] + notes
    (mdir / "qualification.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
