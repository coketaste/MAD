#!/usr/bin/env python3
"""Charts for the FastWAM hands-on post. Reads the rerun artifacts and writes SVG."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
LOG = ROOT / ".mad-runs" / "fam-latency-5" / "log"

ARMS = ("eager", "compile", "geak", "geak@2")
LABELS = {"eager": "eager", "compile": "compile", "geak": "GEAK", "geak@2": "GEAK, 2 steps"}


def median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        raise ValueError("empty")
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def spread(xs: list[float]) -> float:
    med = median(xs)
    return (max(xs) - min(xs)) / abs(med) * 100 if len(xs) > 1 and med else 0.0


def rounds_from_log(path: Path) -> dict[str, list[float]]:
    found: dict[str, list[float]] = {a: [] for a in ARMS}
    if not path.exists():
        return found
    for line in path.read_text(errors="replace").splitlines():
        m = re.search(r"round\s+(\d+)\s+(eager|compile|geak@2|geak)\s+([0-9.]+)\s+ms", line)
        if m:
            found[m.group(2)].append(float(m.group(3)))
    return found


def perf_value(path: Path) -> float | None:
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open()))
    vals = []
    for row in rows:
        if row.get("status") not in (None, "", "SUCCESS"):
            continue
        raw = row.get("performance") or ""
        if raw:
            vals.append(float(raw))
    return vals[-1] if vals else None


def multiple_results(path: Path) -> dict[str, float]:
    out = {}
    if not path.exists():
        return out
    for row in csv.DictReader(path.open()):
        name = row["model"]
        arm = name.split("latency_")[-1] if "latency_" in name else name
        out[arm] = float(row["performance"])
    return out


def svg_header(w: int, h: int, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" role="img" aria-label="{title}">\n'
        f'<title>{title}</title>\n'
        '<rect width="100%" height="100%" fill="#ffffff"/>\n'
        '<style>text{font-family:sans-serif;fill:#1a1a1a} .muted{fill:#555}</style>\n'
    )


def hbar(title: str, rows: list[tuple[str, float]], outfile: Path, unit: str) -> None:
    w, h = 720, 36 * len(rows) + 56
    parts = [svg_header(w, h, title), f'<text x="16" y="24" font-size="16" font-weight="600">{title}</text>']
    ymax = max(v for _, v in rows) or 1
    for i, (name, val) in enumerate(rows):
        y = 40 + i * 32
        bw = (w - 280) * val / ymax
        parts.append(f'<text x="16" y="{y+14}" font-size="13">{name}</text>')
        parts.append(f'<rect x="200" y="{y}" width="{bw:.1f}" height="20" fill="#1f4e79"/>')
        parts.append(f'<text x="{210+bw:.1f}" y="{y+14}" font-size="12">{val:.2f}{unit}</text>')
    parts.append("</svg>\n")
    outfile.write_text("".join(parts))


def whisker(title: str, rows: list[tuple[str, float, float, float]], outfile: Path, ylabel: str) -> None:
    """rows are (label, low, median, high)."""
    w, h = 720, 420
    left, top, right, bottom = 64, 48, 24, 64
    plot_w, plot_h = w - left - right, h - top - bottom
    ymax = max(hi for _, _, _, hi in rows) * 1.12
    slot = plot_w / len(rows)
    parts = [svg_header(w, h, title), f'<text x="{left}" y="28" font-size="16" font-weight="600">{title}</text>']
    for i in range(5):
        y = top + plot_h - plot_h * i / 4
        val = ymax * i / 4
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e6e6e6"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" font-size="11" text-anchor="end" class="muted">{val:.0f}</text>')
    parts.append(f'<text x="16" y="{top+plot_h/2:.0f}" font-size="12" transform="rotate(-90 16 {top+plot_h/2:.0f})" class="muted">{ylabel}</text>')
    for i, (label, lo, med, hi) in enumerate(rows):
        cx = left + i * slot + slot / 2
        def yy(v: float) -> float:
            return top + plot_h - plot_h * v / ymax
        parts.append(f'<line x1="{cx:.1f}" y1="{yy(lo):.1f}" x2="{cx:.1f}" y2="{yy(hi):.1f}" stroke="#1f4e79" stroke-width="2"/>')
        parts.append(f'<line x1="{cx-10:.1f}" y1="{yy(lo):.1f}" x2="{cx+10:.1f}" y2="{yy(lo):.1f}" stroke="#1f4e79"/>')
        parts.append(f'<line x1="{cx-10:.1f}" y1="{yy(hi):.1f}" x2="{cx+10:.1f}" y2="{yy(hi):.1f}" stroke="#1f4e79"/>')
        parts.append(f'<circle cx="{cx:.1f}" cy="{yy(med):.1f}" r="5" fill="#c45911"/>')
        parts.append(f'<text x="{cx:.1f}" y="{yy(med)-8:.1f}" font-size="11" text-anchor="middle">{med:.1f}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{h-bottom+18}" font-size="12" text-anchor="middle">{label}</text>')
    parts.append("</svg>\n")
    outfile.write_text("".join(parts))


def flow_rows(title: str, note: str, rows: list[tuple[str, list[str]]], outfile: Path) -> None:
    """Our own left-to-right diagrams. Each row is (name, boxes)."""
    box_w, box_h, gap_x, row_gap = 196, 46, 36, 28
    width = max(len(boxes) for _, boxes in rows)
    w = 150 + width * box_w + (width - 1) * gap_x + 24
    h = 72 + len(rows) * (box_h + row_gap) + 8
    parts = [
        svg_header(w, h, title),
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">'
        '<path d="M0,0 L6,3 L0,6" fill="#c45911"/></marker></defs>\n',
        f'<text x="16" y="26" font-size="16" font-weight="600">{title}</text>',
        f'<text x="16" y="46" font-size="12" class="muted">{note}</text>',
    ]
    for ri, (name, boxes) in enumerate(rows):
        y = 64 + ri * (box_h + row_gap)
        parts.append(f'<text x="16" y="{y+28}" font-size="13" font-weight="600">{name}</text>')
        x0 = 140
        for bi, label in enumerate(boxes):
            x = x0 + bi * (box_w + gap_x)
            parts.append(f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="6" fill="#f4f7fb" stroke="#1f4e79"/>')
            parts.append(f'<text x="{x+box_w/2}" y="{y+28}" font-size="12" text-anchor="middle">{label}</text>')
            if bi:
                parts.append(
                    f'<line x1="{x-gap_x+4}" y1="{y+box_h/2}" x2="{x-4}" y2="{y+box_h/2}" '
                    'stroke="#c45911" stroke-width="2" marker-end="url(#arrow)"/>'
                )
    parts.append("</svg>\n")
    outfile.write_text("".join(parts))


def main() -> None:
    rounds = rounds_from_log(LOG)
    cited = multiple_results(ROOT / "perf_fam_latency.csv")
    sweep = {
        "eager": perf_value(ROOT / "perf_rerun_eager.csv"),
        "compile": perf_value(ROOT / "perf_rerun_compile.csv"),
        "geak": perf_value(ROOT / "perf_rerun_geak.csv"),
        "geak@2": perf_value(ROOT / "perf_rerun_geak2.csv"),
    }
    numbers = {
        "rounds": rounds,
        "cited": cited,
        "cited_spread": {a: spread(rounds[a]) for a in ARMS if len(rounds[a]) >= 2},
        "sweep": sweep,
        "profile_ms": perf_value(ROOT / "perf_rerun_profile_eager.csv"),
        "dispatch_ms": perf_value(ROOT / "perf_rerun_dispatch.csv"),
        "tune_base": [perf_value(ROOT / f"perf_rerun_clean_base_r{i}.csv") for i in (1, 2)],
        "tune_top": [perf_value(ROOT / f"perf_rerun_clean_score_r{i}.csv") for i in (1, 2)],
        "smoke_eager": perf_value(ROOT / "perf_rerun_smoke_eager.csv"),
        "smoke_geak2": perf_value(ROOT / "perf_rerun_smoke_geak2.csv"),
        "cae": perf_value(ROOT / "perf_rerun_cae.csv"),
    }
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=2))

    if all(a in cited for a in ARMS) and all(len(rounds[a]) >= 2 for a in ARMS):
        whisker(
            "One session. Dot is the median; the bar is min to max.",
            [(LABELS[a], min(rounds[a]), median(rounds[a]), max(rounds[a])) for a in ARMS],
            OUT / "latency-ladder.svg",
            "ms",
        )
    cat = ROOT / "tracelens_output/torch_profiler_output_libkineto_trace_1240_csv/ops_summary_by_category.csv"
    if cat.exists():
        rows = []
        for row in csv.DictReader(cat.open()):
            rows.append((row["op category"], float(row["Percentage (%)"])))
        hbar("Eager kernel time by category, 2 s window", rows, OUT / "operator-mix.svg", "%")
    trace = ROOT / "rocm_trace_lite_output/trace_summary.txt"
    if trace.exists():
        kernels = []
        for line in trace.read_text(errors="replace").splitlines():
            if not line.strip() or line.startswith("Kernel") or line.startswith("=") or line.startswith("Trace") or line.startswith("GPU"):
                if line.startswith("GPU"):
                    break
                continue
            parts = line.split()
            if len(parts) >= 5 and parts[-1].replace(".", "", 1).isdigit():
                kernels.append((" ".join(parts[:-4])[:22], float(parts[-1])))
        if kernels:
            hbar("GEAK dispatch, share of kernel time", kernels[:8], OUT / "kernel-share.svg", "%")
    flow_rows(
        "Two ways to turn an observation into an action",
        "FastWAM is the lower path.",
        [
            ("VLA", ["observations", "action"]),
            ("WAM", ["observations", "predicted scene", "action"]),
        ],
        OUT / "vla-vs-wam.svg",
    )
    flow_rows(
        "FastWAM at inference",
        "The video prediction is not displayed.",
        [("FastWAM", ["language, cameras, state", "video DiT", "action DiT", "32 by 7 chunk"])],
        OUT / "dual-dit.svg",
    )
    base = [v for v in numbers["tune_base"] if v is not None]
    top = [v for v in numbers["tune_top"] if v is not None]
    if len(base) == 2 and len(top) == 2:
        whisker(
            "TunableOp against an unmodified GEAK baseline. Dot is the median.",
            [
                ("baseline", min(base), median(base), max(base)),
                ("TunableOp", min(top), median(top), max(top)),
            ],
            OUT / "tunableop.svg",
            "ms",
        )
    print(json.dumps({k: numbers[k] for k in ("cited", "sweep", "tune_base", "tune_top", "smoke_eager", "smoke_geak2", "cae", "profile_ms", "dispatch_ms")}, indent=2))


if __name__ == "__main__":
    main()
