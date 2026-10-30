#!/usr/bin/env python3
"""
Build the MAD recipes site and status report (GitHub Pages), or freeze a release snapshot.

Site layout written to --out:
  index.html                    browse and search every recipe
  models/<framework>/<model>/   one recipe page per model: madengine and plain Docker commands per GPU
  status/index.html             live status report, diffed against the latest frozen release
  status/<version>/index.html   one report per dashboard/releases/<version>.json
  status/releases.html          list of frozen releases
  search.js                     search index for the top bar

Usage (from anywhere; needs madengine, pyyaml, jinja2):
    python3 tools/dashboard/build.py --out _site
    python3 tools/dashboard/build.py --out _site_internal --internal   # adds dashboard/roadmap.internal.yaml
    python3 tools/dashboard/build.py --freeze v26.10                   # writes dashboard/releases/v26.10.json

Options:
    --strict    fail when the snapshot has warnings (unknown ROCm version, missing Primus checkout, ...)
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys

import jinja2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect  # noqa: E402
import recipes  # noqa: E402

DASHBOARD_DIR = os.path.join(collect.REPO_ROOT, "dashboard")
RELEASES_DIR = os.path.join(DASHBOARD_DIR, "releases")
TEMPLATES_DIR = os.path.join(DASHBOARD_DIR, "templates")


def load_releases():
    """Frozen snapshots, oldest first."""
    releases = []
    for path in glob.glob(os.path.join(RELEASES_DIR, "v*.json")):
        with open(path) as f:
            releases.append(json.load(f))
    return sorted(releases, key=lambda s: collect.version_key(s["version"]))


def view(snapshot):
    """Presentation groupings computed from a snapshot (so frozen JSON stays plain data)."""
    groups = {}
    for row in snapshot["models"]:
        group = groups.setdefault(row["framework"], {
            "name": row["framework"], "workload": row["workload"], "models": [], "rocm": [],
            "sku_counts": {sku: 0 for sku in snapshot["skus"]},
        })
        group["models"].append(row)
        if row["rocm"] not in group["rocm"]:
            group["rocm"].append(row["rocm"])
        for sku in row["skus"]:
            group["sku_counts"][sku] += 1
    ci_skus = {sku: [c for c in snapshot["ci"] if sku in (c.get("skus") or [])] for sku in snapshot["skus"]}
    sku_isa = {sku: isa for isa, skus in snapshot["isa_skus"].items() for sku in skus}
    return {"groups": list(groups.values()), "ci_skus": ci_skus, "sku_isa": sku_isa}


def _variant_label(row, family_rows):
    """Precision plus whatever tags set this row apart from the rest of its page."""
    shared = set.intersection(*(set(r["suites"]) for r in family_rows))
    parts = [row["precision"].upper()] if row["precision"] else []
    parts += [s for s in row["suites"] if s not in shared]
    return " · ".join(parts) or "Default"


def sku_label(skus, all_skus):
    if list(skus) == list(all_skus):
        return f"{all_skus[0]}–{all_skus[-1]}"
    return " · ".join(skus)


def catalog(snapshot):
    """Recipe pages: rows grouped by (framework, model family); variants keyed by GPU product.

    One page per model family (e.g. Llama-3.3-70B-Instruct under vLLM) whose variants are its
    precisions / transports; Primus configs that exist once per GPU product merge into one
    variant whose GPU picker selects the product-specific card.
    """
    families = {}
    for row in snapshot["models"]:
        # 'gpt-oss-120b' and AMD's 'gpt-oss120b-w-mxfp4-a-fp8' are one model: ignore punctuation.
        framework_slug, family_slug = row["page"].split("/", 1)
        key = (framework_slug, re.sub(r"[^a-z0-9]", "", family_slug))
        families.setdefault(key, []).append(row)

    pages, frameworks = [], {}
    for rows in families.values():
        page = rows[0]["page"]
        for row in rows:
            row["page"] = page
        by_label = {}
        for row in rows:
            by_label.setdefault(_variant_label(row, rows), []).append(row)
        variants = []
        for label, vrows in by_label.items():
            by_sku = {}
            for row in vrows:
                for sku in row["skus"]:
                    by_sku.setdefault(sku, row)
            # Two cards on the same product under one label: give each its own variant.
            if sum(len(r["skus"]) for r in vrows) > len(by_sku):
                for row in vrows:
                    variants.append({"label": f"{label} · {row['name'].rsplit('/', 1)[-1]}", "rows": [row],
                                     "by_sku": {sku: row for sku in row["skus"]}})
            else:
                variants.append({"label": label, "rows": vrows, "by_sku": by_sku})
        for i, v in enumerate(variants):
            v["index"] = str(i)
            v["hf_repo"] = v["rows"][0]["hf_repo"]
            v["suites"] = sorted({s for r in v["rows"] for s in r["suites"]})
        first = rows[0]
        repos = [r["hf_repo"] for r in rows if r["hf_repo"]]
        # Prefer the upstream repo (meta-llama/...) over AMD's quantized re-uploads for the heading.
        repo = next((r for r in repos if not r.startswith("amd/")), repos[0] if repos else None)
        skus = [s for s in snapshot["skus"] if any(s in r["skus"] for r in rows)]
        n_gpus = sorted({r["n_gpus"] for r in rows})
        family = {
            "page": page,
            "fw_slug": page.split("/", 1)[0],
            "name": first["family"],
            "org": repo.split("/", 1)[0] if repo else "",
            "hf_repo": repo,
            "framework": first["framework"],
            "workload": first["workload"],
            "readme": first["readme"],
            "variants": variants,
            "precisions": sorted({r["precision"].upper() for r in rows if r["precision"]}),
            "suites": sorted({s for r in rows for s in r["suites"]}),
            "skus": skus,
            "sku_label": sku_label(skus, snapshot["skus"]),
            "gpus": ("8 GPU" if n_gpus == ["all"] else " / ".join(n_gpus) + " GPU") + ("/node" if first["nodes"] > 1 else ""),
            "nodes": max(r["nodes"] for r in rows),
        }
        family["search"] = " ".join([family["name"], repo or "", family["framework"], *family["precisions"],
                                     *family["suites"], *(r["name"] for r in rows)]).lower()
        pages.append(family)
        fw = frameworks.setdefault(family["fw_slug"], {
            "slug": family["fw_slug"], "name": family["framework"], "workload": family["workload"],
            "families": [], "models": 0})
        fw["families"].append(family)
        fw["models"] += len(rows)
    return pages, list(frameworks.values())


def search_index(pages):
    entries = [{"n": p["name"], "f": p["framework"], "p": p["page"], "k": p["search"]} for p in pages]
    return "window.MAD_INDEX = " + json.dumps(entries, separators=(",", ":")) + ";\n"


def default_sku(family):
    """Newest GPU product the first variant runs on."""
    return [s for s in family["skus"] if s in family["variants"][0]["by_sku"]][-1]


def render(env, template, out_path, **context):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(env.get_template(template).render(**context))


def build_site(out_dir, snapshot, releases):
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATES_DIR),
        autoescape=jinja2.select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=jinja2.StrictUndefined,
    )
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    # Pages runs Jekyll otherwise, which is slow and skips underscore paths.
    open(os.path.join(out_dir, ".nojekyll"), "w").close()

    pages, frameworks = catalog(snapshot)
    with open(os.path.join(out_dir, "search.js"), "w") as f:
        f.write(search_index(pages))
    render(env, "index.html.j2", os.path.join(out_dir, "index.html"),
           snap=snapshot, families=pages, frameworks=frameworks, root=".")
    for family in pages:
        cards = [{sku: recipes.recipe(row, sku) for sku, row in v["by_sku"].items()} for v in family["variants"]]
        render(env, "recipe.html.j2", os.path.join(out_dir, "models", family["page"], "index.html"),
               snap=snapshot, f=family, frameworks=frameworks, recipes=cards, isa_of=recipes.ISA_OF,
               default_sku=default_sku(family), root="../../..")

    status_dir = os.path.join(out_dir, "status")
    latest = releases[-1] if releases else None
    render(env, "status.html.j2", os.path.join(status_dir, "index.html"), snap=snapshot, view=view(snapshot),
           diff=collect.diff_snapshots(latest, snapshot), root="..")
    for i, release in enumerate(releases):
        prev = releases[i - 1] if i else None
        render(env, "status.html.j2", os.path.join(status_dir, release["version"], "index.html"), snap=release,
               view=view(release), diff=collect.diff_snapshots(prev, release), root="../..")
    render(env, "releases.html.j2", os.path.join(status_dir, "releases.html"),
           snap=snapshot, root="..", releases=list(reversed(releases)))
    return len(pages)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="_site", help="output directory for the site (default: _site)")
    parser.add_argument("--freeze", metavar="VERSION", help="write dashboard/releases/VERSION.json instead of building")
    parser.add_argument("--internal", action="store_true", help="merge dashboard/roadmap.internal.yaml (never publish)")
    parser.add_argument("--strict", action="store_true", help="fail on snapshot warnings")
    args = parser.parse_args()

    if args.freeze and args.internal:
        parser.error("--freeze writes a committed, public file; it cannot include the internal roadmap")
    if args.internal and not os.path.exists(collect.INTERNAL_ROADMAP_PATH):
        parser.error(f"--internal needs {collect.INTERNAL_ROADMAP_PATH}")

    status = collect.load_status(internal_roadmap_path=collect.INTERNAL_ROADMAP_PATH if args.internal else None)

    if args.freeze:
        collect.version_key(args.freeze)
        path = os.path.join(RELEASES_DIR, f"{args.freeze}.json")
        if os.path.exists(path):
            sys.exit(f"ERROR: {path} already exists; frozen releases are immutable")
        if args.freeze != status["release"]:
            print(f"WARNING: freezing {args.freeze} but status.yaml release is {status['release']}", file=sys.stderr)
        snapshot = collect.build_snapshot(status, version=args.freeze, state="frozen")
    else:
        snapshot = collect.build_snapshot(status)

    if args.strict and snapshot["warnings"]:
        sys.exit(f"ERROR: --strict and {len(snapshot['warnings'])} warning(s); see above")

    if args.freeze:
        os.makedirs(RELEASES_DIR, exist_ok=True)
        with open(path, "w") as f:
            json.dump(snapshot, f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"Froze {len(snapshot['models'])} models into {os.path.relpath(path, collect.REPO_ROOT)}")
        return

    releases = load_releases()
    n_pages = build_site(args.out, snapshot, releases)
    print(f"Wrote {args.out}: {n_pages} recipe pages, {len(snapshot['models'])} models, {len(snapshot['images'])} images, "
          f"{len(releases)} frozen release(s){' [INTERNAL - do not publish]' if snapshot['internal'] else ''}")


if __name__ == "__main__":
    main()
