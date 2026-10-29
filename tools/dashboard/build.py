#!/usr/bin/env python3
"""
Build the MAD status dashboard (GitHub Pages) or freeze a release snapshot.

Site layout written to --out:
  index.html            live view of the current tree, diffed against the latest frozen release
  <version>/index.html  one page per dashboard/releases/<version>.json, diffed against the release before it
  releases.html         list of frozen releases

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
import shutil
import sys

import jinja2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect  # noqa: E402

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

    latest = releases[-1] if releases else None
    render(env, "status.html.j2", os.path.join(out_dir, "index.html"), snap=snapshot, view=view(snapshot),
           diff=collect.diff_snapshots(latest, snapshot), root=".")
    for i, release in enumerate(releases):
        prev = releases[i - 1] if i else None
        render(env, "status.html.j2", os.path.join(out_dir, release["version"], "index.html"), snap=release,
               view=view(release), diff=collect.diff_snapshots(prev, release), root="..")
    render(env, "releases.html.j2", os.path.join(out_dir, "releases.html"),
           snap=snapshot, root=".", releases=list(reversed(releases)))


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
    build_site(args.out, snapshot, releases)
    print(f"Wrote {args.out}: {len(snapshot['models'])} models, {len(snapshot['images'])} images, "
          f"{len(releases)} frozen release(s){' [INTERNAL - do not publish]' if snapshot['internal'] else ''}")


if __name__ == "__main__":
    main()
