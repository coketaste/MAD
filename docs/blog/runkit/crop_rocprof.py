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
    # TraceLens's gpu_timeline takes total_time from these, not from the records.
    tool["metadata"]["init_time"], tool["metadata"]["fini_time"] = t0, t1
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
