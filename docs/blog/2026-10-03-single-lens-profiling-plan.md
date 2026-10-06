# Single-Lens Profiling Round (v3) Implementation Plan

> **For agentic workers:** Execute task by task. Every model ends at a **review gate**: stop, post the qualification report, and wait for the user's approval before starting the next model. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce blog-quality profiling results for the Primus models: every published number comes from a run with exactly one collector, a minimal event set, and a measured, accepted overhead.

**Architecture:** Each run uses one collector and answers one question. TraceLens runs only on the host, after the run. A new host-side cropper cuts whole-run kernel traces to steady-state steps 10–11. A new `qualify.py` writes a per-model qualification report (noise, overhead, drift, trace content, cross-lens kernel agreement), which the user reviews before the next model starts.

**Tech Stack:** madengine (branch `fix/profiling-validation-therock`), rocprofv3 1.3.5 and torch 2.12 in `rocm/primus:v26.7`, TraceLens 0.1.0.dev20261002 (`/home/ysha/.venvs/tracelens`), run kit in `docs/blog/runkit/`.

---

## Why round 2 is not publishable, and what changes

Round 2 already ran one collector per run: P1 used the PyTorch profiler only, R3 used rocprofv3 only, and TraceLens was offline post-processing. The overhead came from **what each collector was asked to record**:

| Run | Configuration | Measured effect |
|---|---|---|
| P1, model B | `record_shapes` (default on), all 8 ranks; ~500k CPU ops/step with concrete inputs | profiled step 47–50 s vs 12.99 s baseline (≈3.8×) |
| P1, model C | Megatron-Bridge hard-codes `with_stack=True` (`profiling.py:125`): 739k python_function events | profiled step 4.86 s vs 1.115 s (≈4.4×) |
| R3, all models | `--hip-trace` on every run; B also `--hsa-trace --rccl-trace --scratch-memory-trace`; JSON output; whole run | throughput 0.23–0.30× baseline; D trace had 1.25M HIP API records vs 63k kernels |
| TraceLens gpu_timeline on R3 | whole run, including init | C: 467 s window, 82% idle (start-up dominates) |

**Principles for v3**

1. **One collector per run, one question per run.** No stacking in the container. TraceLens runs on the host only.
2. **Minimal event set per question.** Kernel timeline → `--kernel-trace --memory-copy-trace` only. Op attribution → PyTorch profiler without shapes or stacks. Roofline → shapes on rank 0 only.
3. **Steady state only.** Analyze steps 10–11 after a 5-step warm-up. For rocprofv3, crop on the host by a per-iteration marker kernel instead of instrumenting the app. (Megatron's `--profile` nsys path would add `emit_nvtx(record_shapes=True)`, i.e. a roctx range per op, which is its own overhead.)
4. **Measure the cost in every run.** Every profiled run is compared with two baselines (B0a/B0b) on the framework's own step time.
5. **Publish each metric from the lens that perturbs it least.** Throughput and step time come only from baselines. GPU busy/idle and kernel mix come from the kernel trace. Op categories and collectives come from the PyTorch profiler. TFLOPS come from the shapes run, and only after its kernel durations agree with the kernel trace.
6. **Validate on the dummy fixture first** (`validate_profilers.py`), then **one model at a time, cheapest first**, with a user review gate after each.

**Run set per model (v3)**

| Run | Collector and settings | Answers | Steps |
|---|---|---|---|
| B0a, B0b | none | step time, noise band | 30 (D: 100) |
| KT | rocprofv3 `--kernel-trace --memory-copy-trace -f json`; whole run; cropped on host to steps 10–11 | GPU busy/idle, kernel-category mix, top kernels, RCCL kernel time | 20 (D: 100) |
| PT | PyTorch profiler; steps 10–11; all 8 ranks; `record_shapes False`, `with_stack False` (C: stack is forced) | op categories, op tree, multi-rank collective report | 30 (D: 100) |
| PS | PyTorch profiler; steps 10–11; **rank 0 only**; `record_shapes True`, `with_stack False` | ops by input shape, GEMM/SDPA TFLOPS (roofline) | 30 (D: 100) |

If PT and PS show the same overhead on the first model, the user may merge them into one run for later models (decided at the gate).

**Qualification gates** (proposed defaults; the user confirms or adjusts them at the first gate)

| Gate | Pass condition | If it fails |
|---|---|---|
| G1 baseline noise | abs(B0a − B0b) / mean ≤ 2% | add steps or rerun baselines |
| G2 run health | rc = 0 (D: rc = 3 with all steps logged) and every step parsed | rerun |
| G3 no lasting perturbation | steady median outside steps 9–13 within max(noise, 2%) of baseline | investigate before using the run |
| G4 KT overhead | steady step / baseline ≤ 1.10 → timeline and idle% publishable; ≤ 1.30 → kernel mix and durations only | > 1.30: stop, redesign |
| G5 PT overhead | window step / baseline ≤ 1.30 → op times publishable; else publish op shares (%) only | — |
| G6 PS fidelity | median duration of the top-10 kernels within ±5% of KT | no roofline from this run |
| G7 trace content | KT: 8 rank files with kernels, RCCL kernels present. PT: 8 traces with ProfilerStep#10/#11 and collective records. PS: ≥ 50% of CPU ops carry `Input Dims` | rerun |
| G8 disk | `/data` free ≥ 3× expected trace size before the run | clean first |

---

## File structure

- Modify: `docs/blog/runkit/run_matrix.py`: add KT/PT/PS runs and per-model PT/PS args; D baseline 100 steps.
- Modify: `docs/blog/runkit/analyze_runs.py`: treat PT/PS like P1 (capture window); add KT/PT/PS to RUNS.
- Modify: `docs/blog/runkit/validate_profilers.py`: add a `kernel_trace` case for the new KT command.
- Create: `docs/blog/runkit/crop_rocprof.py`: crop a rocprofv3 JSON to steps N..M by a per-iteration marker kernel.
- Create: `docs/blog/runkit/qualify.py`: per-model qualification report (gates G1–G7).
- Results: `/data/ysha/primus_blog_results/mi300x-v3/` (select with `BLOG_GPU=mi300x-v3`; no code change needed).

---

### Task 0: Cleanup (user-run; deletion was blocked by the permission classifier)

- [ ] **Step 1: Delete the verified duplicate R3 copies (≈34 GB)**

All top-level copies were byte-compared with `diff -rq` against `R3/run_directory/`. C and D are identical; B's top-level copy is a strict subset.

```bash
cd /data/ysha/primus_blog_results/mi300x-r2 && for r in B_qwen3_30B_A3B-megatron C_qwen3_32b_sft-megatron_bridge D_flux_535m-megatron_diffusion; do rm -rf $r/R3/rocprof_output $r/R3/tracelens_output; done; df -h /data
```

- [ ] **Step 2: After Task 2 passes, delete the remaining round-2 raw rocprofv3 traces (≈54 GB)**

Keep D's R3 JSON until Task 2 has used it as cropper test input. Keep all P1 traces (2.6 GB); they are the "before" data for the overhead comparison.

```bash
cd /data/ysha/primus_blog_results/mi300x-r2 && rm -rf */R3/run_directory/rocprof_output && df -h /data
```

### Task 1: Run kit: KT / PT / PS runs

**Files:** Modify `docs/blog/runkit/run_matrix.py`, `docs/blog/runkit/analyze_runs.py`

- [ ] **Step 1: Add the kernel-trace command and PT/PS args** (after `MEGATRON_P1` in `run_matrix.py`)

```python
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
MEGATRON_PT = _MEGATRON_TORCH + "--torch_profiler_record_shapes False --profile_ranks [0,1,2,3,4,5,6,7]"
MEGATRON_PS = _MEGATRON_TORCH + "--torch_profiler_record_shapes True --profile_ranks [0]"
# Megatron-Bridge hard-codes with_stack=True (profiling.py:125); only record_shapes is ours.
_BRIDGE_TORCH = (
    "--profiling.use_pytorch_profiler True --profiling.profile_step_start 10 "
    "--profiling.profile_step_end 12 "
    "--logger.tensorboard_dir /myworkspace/run_directory/output/tensorboard "
)
BRIDGE_PT = _BRIDGE_TORCH + "--profiling.record_shapes False --profiling.profile_ranks [0,1,2,3,4,5,6,7]"
BRIDGE_PS = _BRIDGE_TORCH + "--profiling.record_shapes True --profiling.profile_ranks [0]"
```

- [ ] **Step 2: Add `pt_args` / `ps_args` to models B, C, D** (A is added in Task 7 after its flags are confirmed)

```python
# B and D
"pt_args": MEGATRON_PT,
"ps_args": MEGATRON_PS,
# C
"pt_args": BRIDGE_PT,
"ps_args": BRIDGE_PS,
```

For D, also change `"base_args"` and `"short_args"` to `"--train_iters 100 --log_interval 1 --attention_backend fused"`. Its 43 ms steps showed 7.9% noise over 30 iterations.

- [ ] **Step 3: Add the run branches** in `build_context` (before `else: raise ValueError(run)`), and set `RUNS`

```python
    elif run == "KT":
        # Whole-run kernel trace; crop_rocprof.py cuts steps 10-11 on the host.
        args += model["short_args"]
        tools = [{"name": "rocprofv3_lightweight", "cmd": KERNEL_TRACE}]
    elif run in ("PT", "PS"):
        args += f"{model['base_args']} {model[run.lower() + '_args']}"
```

```python
RUNS = ["B0a", "B0b", "KT", "PT", "PS"]
EXTRA_RUNS = set()
```

In `main()`, the existing `runs = RUNS if a.runs == "all" else a.runs.split(",")` keeps working. Round-2 runs (P1, K2, K2s, R3, R3c) remain runnable by name through `build_context`.

- [ ] **Step 4: Teach `analyze_runs.py` the new runs**

```python
RUNS = {"B0a", "B0b", "P1", "K2", "K2s", "R3", "KT", "PT", "PS"}
WINDOWED = {"P1", "PT", "PS"}   # PyTorch profiler active on steps 10-11
```

Replace `name == "P1"` with `name in WINDOWED` in `steady()` and in the `capture_window_step_s` field.

- [ ] **Step 5: Dry-run check**

Run: `cd /home/ysha/codebase/MAD && python3 docs/blog/runkit/run_matrix.py --models B,C,D --runs KT,PT,PS --dry-run`
Expected: nine `madengine run` commands. KT contexts carry `"cmd": "bash ../scripts/common/tools/rocprof_wrapper.sh --kernel-trace --memory-copy-trace ..."` with no `--hip-trace`. B/D PT carries `--torch_profiler_record_shapes False`. C PS carries `--profiling.record_shapes True --profiling.profile_ranks [0]`.

### Task 2: Host-side cropper for rocprofv3 JSON

**Files:** Create `docs/blog/runkit/crop_rocprof.py`

- [ ] **Step 1: Write the cropper**

```python
#!/usr/bin/env python3
"""Crop a rocprofv3 *_results.json to training steps FIRST..LAST (inclusive, 1-based).

rocprofv3 traces the whole process, so TraceLens's gpu_timeline would include start-up.
Instead of instrumenting the app (Megatron's nsys window adds a roctx range per op), find a
kernel that runs a fixed number of times per iteration (e.g. the grad-norm
custom_multi_tensor_l2norm_kernel: 20 dispatches in 20 iterations on model D), take its
last dispatch in each iteration as the step boundary, and keep only kernel and copy
records inside the window. Host API records are dropped.

Usage:
    crop_rocprof.py IN.json OUT.json --iters 20 [--first 10 --last 11] [--marker REGEX]
"""

import argparse
import collections
import json
import re
import sys

PREFERRED = re.compile(r"l2norm|adam", re.I)


def load(path):
    # Some kernel names carry bytes that are not valid UTF-8.
    with open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8", "replace"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--iters", type=int, required=True)
    ap.add_argument("--first", type=int, default=10)
    ap.add_argument("--last", type=int, default=11)
    ap.add_argument("--marker", help="regex for the per-iteration kernel (default: auto)")
    a = ap.parse_args()

    doc = load(a.src)
    tool = doc["rocprofiler-sdk-tool"][0]
    names = {s["kernel_id"]: s.get("truncated_kernel_name") or s["kernel_name"] for s in tool["kernel_symbols"]}
    kd = tool["buffer_records"]["kernel_dispatch"]
    counts = collections.Counter(names.get(r["dispatch_info"]["kernel_id"], "") for r in kd)

    if a.marker:
        cands = [n for n in counts if re.search(a.marker, n)]
    else:
        cands = [n for n, c in counts.items() if c and c % a.iters == 0 and PREFERRED.search(n)]
        cands = cands or [n for n, c in counts.items() if c == a.iters]
    if not cands:
        sys.exit(f"no per-iteration marker kernel among {len(counts)} kernels; pass --marker")
    marker = min(cands, key=lambda n: (counts[n] // a.iters, n))
    per_iter = counts[marker] // a.iters
    ends = sorted(r["end_timestamp"] for r in kd if names.get(r["dispatch_info"]["kernel_id"]) == marker)
    ends = ends[per_iter - 1::per_iter]            # last marker dispatch of each iteration
    if len(ends) != a.iters or not 2 <= a.first <= a.last <= a.iters:
        sys.exit(f"marker {marker!r}: {len(ends)} step boundaries for {a.iters} iterations")
    t0, t1 = ends[a.first - 2], ends[a.last - 1]   # end of step FIRST-1 .. end of step LAST

    br = tool["buffer_records"]
    for key in list(br):
        if key in ("kernel_dispatch", "memory_copy"):
            br[key] = [r for r in br[key] if t0 <= r["start_timestamp"] and r["end_timestamp"] <= t1]
        else:
            br[key] = []
    tool["callback_records"] = {k: [] for k in tool.get("callback_records", {})}
    with open(a.dst, "w") as f:
        json.dump(doc, f)
    busy = sum(r["end_timestamp"] - r["start_timestamp"] for r in br["kernel_dispatch"])
    print(json.dumps({
        "marker": marker, "per_iter": per_iter, "steps": [a.first, a.last],
        "window_s": round((t1 - t0) / 1e9, 4),
        "step_s": round((t1 - t0) / 1e9 / (a.last - a.first + 1), 4),
        "kernels": len(br["kernel_dispatch"]), "kernel_time_s": round(busy / 1e9, 4),
    }))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test on round 2's model D trace** (20 iterations, known marker)

Run:
```bash
cd /data/ysha/primus_blog_results/mi300x-r2/D_flux_535m-megatron_diffusion/R3/run_directory/rocprof_output/banff-cyxtera-s83-5
f=$(ls -S *_results.json | head -8 | tail -1)
/home/ysha/.venvs/tracelens/bin/python /home/ysha/codebase/MAD/docs/blog/runkit/crop_rocprof.py $f /tmp/d_crop_results.json --iters 20
```
Expected: `"marker": "custom_multi_tensor_l2norm_kernel"`, `"per_iter": 1`, and `step_s` ≈ 0.31 s. The round-2 R3 log's steady step was 0.3057 s for the same run, so a match confirms the crop boundaries line up with the training log.

- [ ] **Step 3: TraceLens on the cropped file**

Run: `/home/ysha/.venvs/tracelens/bin/TraceLens_generate_perf_report_rocprof --profile_json_path /tmp/d_crop_results.json --output_csvs_dir /tmp/d_crop_tl && cat /tmp/d_crop_tl/gpu_timeline.csv`
Expected: `total_time` ≈ `window_s` × 1000 ms (not 467 s), non-zero `kernel` time.

- [ ] **Step 4: Remove temp files, then run Task 0 Step 2**

Run: `rm -rf /tmp/d_crop_results.json /tmp/d_crop_tl`

### Task 3: Qualification report

**Files:** Create `docs/blog/runkit/qualify.py`

- [ ] **Step 1: Write `qualify.py`**

```python
#!/usr/bin/env python3
"""Write <model_dir>/qualification.md: gates G1-G7 for one model's v3 runs.

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

NOISE_MAX, DRIFT_MIN, KT_PUB, KT_MAX, PT_MAX, FIDELITY = 0.02, 0.02, 1.10, 1.30, 1.30, 0.05
WARMUP, CAPTURE, WINDOW = 5, range(9, 14), (10, 11)


def norm(name):
    return re.sub(r"\(.*", "", name).strip()


def steady(series, windowed):
    vals = [v for k, v in series.items() if k >= WARMUP and not (windowed and k in CAPTURE)]
    return statistics.median(vals) if vals else None


def kineto(path):
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        return json.load(f)["traceEvents"]


def kernel_medians_kineto(ev):
    d = collections.defaultdict(list)
    for e in ev:
        if e.get("cat") == "kernel":
            d[norm(e["name"])].append(e["dur"])          # us
    return {k: statistics.median(v) for k, v in d.items()}, {k: sum(v) for k, v in d.items()}


def kernel_medians_rocprof(path):
    doc = json.loads(open(path, "rb").read().decode("utf-8", "replace"))["rocprofiler-sdk-tool"][0]
    names = {s["kernel_id"]: s.get("truncated_kernel_name") or s["kernel_name"] for s in doc["kernel_symbols"]}
    d = collections.defaultdict(list)
    for r in doc["buffer_records"]["kernel_dispatch"]:
        d[norm(names.get(r["dispatch_info"]["kernel_id"], ""))].append((r["end_timestamp"] - r["start_timestamp"]) / 1e3)
    return {k: statistics.median(v) for k, v in d.items()}


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

    for name in ("KT", "PT", "PS"):
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
                if name == "PT" and ratio:
                    rows.append(("G5 PT window overhead", f"{ratio:.2f}x", "PASS" if ratio <= PT_MAX else "SHARES-ONLY"))
                if name == "PS" and ratio:
                    rows.append(("PS window overhead (info)", f"{ratio:.2f}x", "-"))
            else:
                ratio = med / base_mean
                verdict = "PASS" if ratio <= KT_PUB else ("KERNELS-ONLY" if ratio <= KT_MAX else "FAIL")
                rows.append(("G4 KT overhead", f"{ratio:.2f}x", verdict))

    kt_crops = sorted(glob.glob(str(runs["KT"] / "run_directory/rocprof_output/*/*_steps10-11_results.json"))) if "KT" in runs else []
    if "KT" in runs:
        rows.append(("G7 KT cropped rank files", str(len(kt_crops)), "PASS" if len(kt_crops) == 8 else "FAIL"))
    if "PT" in runs:
        tr = traces(runs["PT"])
        ev = kineto(tr[0]) if tr else []
        steps = {e["name"] for e in ev if str(e.get("name", "")).startswith("ProfilerStep#")}
        coll = sum("Collective name" in e.get("args", {}) for e in ev)
        ok = len(tr) == 8 and {"ProfilerStep#10", "ProfilerStep#11"} <= steps and coll > 0
        rows.append(("G7 PT traces / steps / collectives", f"{len(tr)} / {sorted(steps)} / {coll}", "PASS" if ok else "FAIL"))
    if "PS" in runs:
        tr = traces(runs["PS"])
        ev = kineto(tr[0]) if tr else []
        cpu = [e for e in ev if e.get("cat") == "cpu_op"]
        frac = sum("Input Dims" in e.get("args", {}) for e in cpu) / max(len(cpu), 1)
        rows.append(("G7 PS shapes recorded", f"{frac:.0%} of {len(cpu)} cpu ops", "PASS" if frac >= 0.5 else "FAIL"))
        if kt_crops and ev:
            # Rank 0 of PS vs the KT rank file with the most kernel time (rank files are named by pid).
            ps_med, ps_tot = kernel_medians_kineto(ev)
            kt_med = kernel_medians_rocprof(max(kt_crops, key=lambda f: Path(f).stat().st_size))
            top = [k for k, _ in sorted(ps_tot.items(), key=lambda kv: -kv[1]) if k in kt_med][:10]
            dev = {k: ps_med[k] / kt_med[k] - 1 for k in top}
            worst = max(dev.values(), key=abs) if dev else None
            rows.append(("G6 PS vs KT top-10 kernel medians", f"worst {worst:+.1%} over {len(dev)} kernels" if dev else "no overlap",
                         "PASS" if dev and abs(worst) <= FIDELITY else "FAIL"))
            notes += [f"- `{k[:90]}`: PS {ps_med[k]:.1f} us vs KT {kt_med[k]:.1f} us ({v:+.1%})" for k, v in dev.items()]

    out = [f"# Qualification: {mdir.name}", "", f"Baseline step: {base_mean:.4g} s" if base_mean else "Baseline: missing", "",
           "| Gate | Value | Verdict |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in rows]
    if notes:
        out += ["", "## Kernel agreement (G6)", ""] + notes
    (mdir / "qualification.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test against round 2** (expected to fail the gates; this only checks that it runs)

Run: `python3 docs/blog/runkit/qualify.py /data/ysha/primus_blog_results/mi300x-r2/D_flux_535m-megatron_diffusion`
Expected: G1 shows 7.88% → FAIL. No KT/PT/PS rows, because round 2 has no runs under those names. No traceback.

### Task 4: Step 0: validate the KT command on the dummy fixture

**Files:** Modify `docs/blog/runkit/validate_profilers.py`

- [ ] **Step 1: Add the case** (in `CASES`)

```python
    # v3 kernel lens: the exact command run_matrix.py uses for KT (no --hip-trace).
    "kernel_trace": (
        [{"name": "rocprofv3_lightweight", "cmd": "bash ../scripts/common/tools/rocprof_wrapper.sh "
          "--kernel-trace --memory-copy-trace --output-format json -d ./rocprof_output --"}],
        {}, ["perf", "rocprof", "rocprof_kernels"],
    ),
```

- [ ] **Step 2: Run it**

Run: `cd /home/ysha/codebase/MAD && python3 docs/blog/runkit/validate_profilers.py --cases kernel_trace,torch_profiler_tracelens,torch_profiler_collective`
Expected: all three `PASS`. The `kernel_trace` rocprof JSON has kernel dispatch records on real GPU ids and **zero** `hip_api` records. To check the latter:
`python3 -c "import json,glob;f=sorted(glob.glob('/data/ysha/primus_blog_results/profiler-validate/run-*/kernel_trace/**/*_results.json',recursive=True))[0];t=json.loads(open(f,'rb').read().decode('utf-8','replace'))['rocprofiler-sdk-tool'][0]['buffer_records'];print({k:len(v) for k,v in t.items() if v})"`
→ only `kernel_dispatch` (and possibly `memory_copy`).

- [ ] **Step 3: Review gate 0.** Post the validation output to the user. Confirm the gate thresholds and the model order (D → C → B → A). **Wait for approval.**

### Task 5: Model D (FLUX 535M): cheapest; shakes out the pipeline

- [ ] **Step 1: Disk check (G8).** `df -h /data`: need ≥ 20 GB free.
- [ ] **Step 2: Baselines.** `cd /home/ysha/codebase/MAD && BLOG_GPU=mi300x-v3 python3 docs/blog/runkit/run_matrix.py --models D --runs B0a,B0b` (≈ 2 × 5 min). Run `qualify.py` and check G1 ≤ 2% **before** spending time on profiled runs. If it fails, raise the iteration count and rerun the baselines.
- [ ] **Step 3: KT.** `BLOG_GPU=mi300x-v3 python3 docs/blog/runkit/run_matrix.py --models D --runs KT`, then crop every rank:
```bash
cd /data/ysha/primus_blog_results/mi300x-v3/D_flux_535m-megatron_diffusion/KT/run_directory/rocprof_output/*/
for f in *_results.json; do case $f in *_steps10-11_*) continue;; esac
  /home/ysha/.venvs/tracelens/bin/python /home/ysha/codebase/MAD/docs/blog/runkit/crop_rocprof.py $f ${f%_results.json}_steps10-11_results.json --iters 100; done
```
Some rank files (helper processes) may have no marker kernel. Only the 8 training ranks need to crop successfully.
- [ ] **Step 4: PT, then PS.** `BLOG_GPU=mi300x-v3 python3 docs/blog/runkit/run_matrix.py --models D --runs PT,PS`
- [ ] **Step 5: Host TraceLens** (outputs next to each run)
```bash
R=/data/ysha/primus_blog_results/mi300x-v3/D_flux_535m-megatron_diffusion
for f in $R/KT/run_directory/rocprof_output/*/*_steps10-11_results.json; do /home/ysha/.venvs/tracelens/bin/TraceLens_generate_perf_report_rocprof --profile_json_path $f --output_csvs_dir $R/KT/tracelens/$(basename $f .json) --short_kernel_study; done
madengine report tracelens --root $R/PT/run_directory --mode pytorch --gpu-arch MI300X --output-dir $R/PT/tracelens/pytorch --python /home/ysha/.venvs/tracelens/bin/python
madengine report tracelens --root $R/PT/run_directory --mode collective --gpu-arch MI300X --output-dir $R/PT/tracelens/collective --python /home/ysha/.venvs/tracelens/bin/python
madengine report tracelens --root $R/PS/run_directory --mode pytorch --gpu-arch MI300X --output-dir $R/PS/tracelens/pytorch --python /home/ysha/.venvs/tracelens/bin/python
```
- [ ] **Step 6: Qualify and summarize.** `python3 docs/blog/runkit/qualify.py $R && BLOG_RESULTS_ROOT=/data/ysha/primus_blog_results python3 docs/blog/runkit/analyze_runs.py mi300x-v3`
- [ ] **Step 7: Review gate D.** Post `qualification.md`, KT `gpu_timeline.csv` and kernel categories, PT `gpu_timeline` and `ops_summary_by_category`, PS GEMM sheet, and trace sizes. Ask whether PT and PS can be merged for later models. **Wait for approval.** After approval, delete D's uncropped KT JSONs; keep the cropped ones and the reports.

### Task 6: Model C (Qwen3-32B SFT, Megatron-Bridge)

Same steps as Task 5 with `--models C`, `--iters 20` for the cropper, and ≥ 40 GB free.

C-specific checks for the gate:
- PT/PS carry a forced Python stack (Megatron-Bridge hard-codes it), so G5 may land in "shares only". The blog states this as a backend limitation. Patching third-party code is out of scope unless the user asks.
- TransformerEngine ops (`_LayerNormLinear`, `_Linear`) are categorized as "other" by TraceLens. Report the GEMM share from KT kernel categories instead.
- Do not publish C's TFLOP/s or MFU from `primus_perf_output.csv` (known wrong for the bridge, risk 8). Use tokens/s and step time.
- **Review gate C. Wait for approval.**

### Task 7: Model B (Qwen3-30B-A3B MoE, Megatron-LM)

Same steps as Task 5 with `--models B`, `--iters 20`, and ≥ 60 GB free (KT estimate: ~95k kernels/step).

B-specific checks:
- Identify the MoE dispatch/combine kernels in KT (TraceLens put them in "MoE_comm" in round 2). Check whether they are RCCL all-to-all kernels or Primus-Turbo/DeepEP kernels. That decides whether the "exposed all-to-all" story uses the collective report (PT) or kernel overlap (KT).
- Only if the RCCL API view is still needed after that: add one optional run with `--rccl-trace --kernel-trace` and nothing else, as its own collector run.
- **Review gate B. Wait for approval.**

### Task 8: Model A (Llama 3.1 8B, TorchTitan): blocked on the Hugging Face token

- [ ] **Step 1:** Find TorchTitan's record_shapes/with_stack switches: `grep -rn "record_shapes\|with_stack\|profile_freq" /home/ysha/codebase/MAD/scripts/Primus/third_party/torchtitan/torchtitan/tools/profiling.py /home/ysha/codebase/MAD/scripts/Primus/third_party/torchtitan/torchtitan/config/job_config.py`. Add `pt_args` / `ps_args` to model A to match what that shows. TorchTitan profiles every `profile_freq` steps, so set the run length so that exactly one window falls in steady state.
- [ ] **Step 2:** Same steps as Task 5 with `--models A`.
- [ ] **Review gate A.** A's PS roofline is the blog's main roofline figure (proposal Figure 7).

---

## Execution notes (2026-10-03, Tasks 1–4)

- **Cropper:** TraceLens's rocprof `gpu_timeline` takes `total_time` from `metadata.init_time/fini_time`, not from the records, so `crop_rocprof.py` also rewrites both. Tested on round-2 D (8 ranks): cropped step 0.306–0.307 s vs the log's 306.0/305.5 ms, and TraceLens total 614 ms for 2 steps. The auto marker is `custom_multi_tensor_l2norm_kernel` or `custom_adam_kernel`, both once per iteration.
- **TraceLens bug:** in the rocprof report, `kernel_summary_by_category.csv` column `total_direct_kernel_time_ms` is really **µs**. It divides ns durations by 1000. Percentages are correct; divide the values by 1000 before plotting. Worth reporting upstream.
- **G6:** collective kernels (`nccl|rccl`) are excluded, because their duration includes waiting for peers. On round-2 D data (KT stand-in under `--hip-trace`), 8 of 10 GEMMs agreed within ±1.6% and 2 differed by 10–13%.
- **Archive dedupe:** `run_matrix.archive()` now drops the top-level `rocprof_output` copy when `diff -rq` shows it is a subset of `run_directory/rocprof_output`. That avoids round 2's 34 GB of duplicates.
- **Task 3 smoke test:** run on a scratch copy built from round-2 D data (B0a/B0b, P1 as PT/PS, cropped R3 as KT) so every gate path executes. Round-2 D's numbers under the v3 gates: noise 7.88% (FAIL), R3-as-KT overhead 6.78× (FAIL), P1 window 6.70× (SHARES-ONLY).
- **Validator:** `validate_profilers.py` has a `kernel_trace` case plus a `rocprof_no_host_api` check (kernel dispatches > 0, zero `*_api` records), and two overhead probes (`kernel_trace_csv`, `rocprof_marker_only`).
- **Gate 0 finding — rocprofv3 kernel tracing costs ~280 µs per dispatch on `rocm/primus:v26.7`** (rocprofv3 1.3.5, ROCm 10.0.0 TheRock wheels). Dummy model (runs `run-20261003-100238` and the probe run that followed):

  | case | samples/s | vs baseline |
  |---|---|---|
  | baseline | 106,148 | 1.00× |
  | rocprofv3 loaded, `--marker-trace` only | 105,876 | 1.00× |
  | `--kernel-trace` (JSON) | 13,238 | 0.12× |
  | `--kernel-trace` (CSV) | 13,190 | 0.12× |
  | old `rocprofv3_lightweight` (with `--hip-trace`) | ~13,200 | 0.12× |
  | torch profiler (Kineto also records kernels) | ~103,500 | 0.97× |

  Loading the tool is free; output format and host API tracing don't matter. The cost is in per-dispatch kernel interception: median 169 µs gap between kernels, about 120 kernels per 4.8 ms step. Round 2 fits the same ~300–400 µs per kernel: B +30 s/step over ~95k kernels, C +3.7 s over ~9.5k, D +0.26 s over ~680. **KT as designed cannot pass G4 on any model.**
- **Gate 0 decision (user, 2026-10-03): option A.** PT (Kineto, no shapes, no stack) becomes the source for the GPU timeline and kernel mix. KT is dropped from `--runs all` but stays runnable by name. G6 compares PS against PT's rank 0 (KT is used if it exists). rocprofv3 appears in the blog as a measured cost on this stack (~280 µs per dispatch) and goes upstream as an issue. **The per-model run set is now B0a, B0b, PT, PS.** Task 5 steps 3 (KT and the crop loop) and the KT TraceLens line in step 5 are skipped.

## Execution notes: model D (2026-10-03)

Results are in `/data/ysha/primus_blog_results/mi300x-v3/D_flux_535m-megatron_diffusion/` (`qualification.md`; TraceLens under `PT/tracelens`, `PS/tracelens`). Every run exits rc=3 (known FLUX extractor issue) with all 100 steps logged.

| | B0a/B0b | PT (no shapes, 8 ranks) | PS (shapes, rank 0) |
|---|---|---|---|
| steady step | 42.8 / 43.05 ms (noise 0.58%) | drift −0.76% | drift +0.17% |
| steps 9–11 (warm-up + window) | ~43 ms | 298–302 ms (**7.04×**) | 89–95 ms (**2.18×**) |
| TraceLens computation_time (2 steps) | — | 30.80 ms | 31.15 ms |
| GPU idle in window | — | 84.9% | 73.3% |
| trace size | — | 5 MB | 0.5 MB |

Findings:
- **Overhead scales with the number of profiled ranks, not with shapes.** PT with 8 ranks costs 7×; PS with rank 0 costs 2.2×. Step 9 (profiler warm-up) is already slow, so the cost is in Kineto's collection. The node has 224 CPUs, so core contention is unlikely; the cause is not yet known. Per kernel that is ≈380 µs (8 ranks) vs ≈72 µs (1 rank), with ~680 kernels per step.
- **Correction to gate 0:** the dummy model's "~3% torch profiler cost" was a median over 30 steps, which hides a 2-step window. It does not show the window overhead.
- **GPU-side numbers hold up under overhead:** computation time is 30.8 vs 31.2 ms (1.1%) and the category shares match within 0.8 pt between 7× and 2.2× windows. GEMM medians agree within ±1.5%. What inflates is idle time and exposed communication, which includes waiting on slow ranks.
- **D is host/launch-bound:** ~15.5 ms GPU compute per 42.9 ms unprofiled step (≈36%).
- **G6 as written fails on tiny kernels:** FillFunctor is 2.5 vs 2.9 µs (−12.5%) and an elementwise kernel 19.7 vs 18.0 µs (+9%). GEMMs are within ±1.5%.

### Changes a–c and run d (user-approved, 2026-10-03)

- **a.** PT and PS profile rank 0 only. The new run **PC** (no shapes, all 8 ranks) is only for the collective report. D's earlier 8-rank PT matched PC exactly and was renamed. Run set: B0a, B0b, PT, PS, PC.
- **b.** `qualify.py` reports "GPU compute busy (derived)" = union of non-collective kernel time per step on PT rank 0 ÷ baseline step. Idle % from profiled windows is not published.
- **c.** G6 is now G6a (GEMM/attention kernel medians, PS vs PT, ±5%) plus G6b (GPU compute per step, ±3%).
- **d.** D, PT on rank 0 without shapes, run twice:

  | run | steps 5–8 | steps 14–100 | window (10–11) | G3 drift | G6a / G6b |
  |---|---|---|---|---|---|
  | PT run 1 (`PT_run1_slow_host`) | 48.8 ms | 48.5 ms | 2.15× | +12.75% FAIL | −1.2% / −0.5% PASS |
  | PT run 2 (`PT`) | 45.0 ms | 44.7 ms | 2.60× | +4.14% FAIL | +2.4% / +0.1% PASS |
  | PS (shapes, rank 0) | 43.0 ms | 43.0 ms | 2.18× | +0.17% | — |
  | PC (8 ranks) | 42.6 ms | 42.7 ms | 7.04× | −0.76% | — |

  Rank count drives the window cost (1 rank ≈ 2.2×, 8 ranks ≈ 7×); shapes add nothing measurable. Both PT runs were slower through the **whole** run, including steps 5–8 before collection starts, while PS and PC (same profiler path) were not. Cause unknown: host jitter on a host-bound model, or something about the PT configuration. Kernel-level numbers are unaffected. GPU compute is 15.72–15.81 ms per step in every run, which gives a derived GPU compute busy of ≈36.7%.

## Self-review notes

- Every round-2 overhead source is addressed: P1 shapes on 8 ranks → PT (no shapes) plus PS (rank 0); C's forced stack → documented, publish shares only; R3 `--hip-trace` → KT without host API tracing; whole-run timeline → cropper.
- Open risks: (1) KT per-dispatch overhead on B (~95k kernels/step) is unmeasured; gate G4 decides. (2) If a model has no per-iteration marker kernel, pass `--marker` with a kernel regex chosen from the KT kernel counts. (3) G6 compares PS rank 0 against the largest KT rank file, not necessarily rank 0, since rocprofv3 names files by PID. For data-parallel ranks the kernel mix is identical, so this is acceptable.
