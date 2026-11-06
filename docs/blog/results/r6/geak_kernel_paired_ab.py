"""Paired A/B of the fastwam::skmm configs GEAK tried, in one process.

Configs: the shipped defaults, the 2026-11-03 budget-1 patch, and the first r6 budget-1 candidate.
Each repetition times every config back to back, in a rotating order, with the GEAK harness's method:
median of graph-replayed per-call time, 50 calls per graph. "hot" reuses one weight tensor, as the
harness does. "rotating" cycles 64 weight tensors, more than the 256 MB last-level cache.
"""
import math
import os
import statistics as st
import sys

import torch

sys.path.insert(0, "/ws")
from fastwam.models.wan22 import mot  # noqa: E402

CFGS = {
    "baseline": {"skred": (256, 4), 3072: None, 4096: None},
    "nov3_patch": {"skred": (128, 1), 3072: None, 4096: "32,32,256,8,2,2"},
    "r6_run1": {"skred": (1024, 4), 3072: "32,16,256,4,2,2", 4096: "32,16,256,4,2,2"},
}
CASES = (3072, 4096)
REPS = int(os.environ.get("AB_REPS", "12"))


def apply(cfg):
    mot._SKRED = cfg["skred"]
    for k in CASES:
        key = f"FASTWAM_SKCFG_1024_{k}"
        if cfg[k]:
            os.environ[key] = cfg[k]
        else:
            os.environ.pop(key, None)


def inputs(k, n_w):
    g = torch.Generator(device="cuda").manual_seed(0)
    x = (torch.randn(1, 32, k, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    ws = [(torch.randn(1024, k, device="cuda", generator=g) * 0.05).to(torch.bfloat16) for _ in range(n_w)]
    return x, ws


def per_call_us(x, ws, reps=50, iters=200, warmup=10):
    for _ in range(warmup):
        for i in range(reps):
            mot._fw_skmm(x, ws[i % len(ws)])
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(reps):
            mot._fw_skmm(x, ws[i % len(ws)])
    graph.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        graph.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / reps * 1000)
    return st.median(ts)


def main():
    data = {mode: {k: inputs(k, 1 if mode == "hot" else 64) for k in CASES} for mode in ("hot", "rotating")}
    ref = {}
    apply(CFGS["baseline"])
    for k in CASES:
        x, ws = data["hot"][k]
        ref[k] = mot._fw_skmm(x, ws[0]).float()
    for name, cfg in CFGS.items():
        apply(cfg)
        for k in CASES:
            x, ws = data["hot"][k]
            err = (mot._fw_skmm(x, ws[0]).float() - ref[k]).abs().max().item()
            print(f"CHECK {name} K={k} max_abs_diff_vs_baseline={err:.5f}", flush=True)
    names = list(CFGS)
    res = {(m, n, k): [] for m in ("hot", "rotating") for n in names for k in CASES}
    for rep in range(REPS):
        order = names[rep % len(names):] + names[:rep % len(names)]
        for mode in ("hot", "rotating"):
            for name in order:
                apply(CFGS[name])
                for k in CASES:
                    x, ws = data[mode][k]
                    res[(mode, name, k)].append(per_call_us(x, ws))
        print(f"REP {rep + 1}/{REPS} done", flush=True)
    for mode in ("hot", "rotating"):
        for name in names:
            per_case = []
            for k in CASES:
                base, cand = res[(mode, "baseline", k)], res[(mode, name, k)]
                ratios = sorted(b / c for b, c in zip(base, cand))
                per_case.append(st.median(ratios))
                print(f"RESULT mode={mode} cfg={name} K={k} median_us={st.median(cand):.3f} "
                      f"paired_speedup_median={st.median(ratios):.4f} min={ratios[0]:.4f} max={ratios[-1]:.4f}")
            print(f"GEOMEAN mode={mode} cfg={name} paired_speedup={math.exp(sum(map(math.log, per_case)) / 2):.4f}")


if __name__ == "__main__":
    main()
