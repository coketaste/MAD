"""Unit tests for recipe generation and recipe-page grouping (no madengine, Docker or GPU needed)."""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build  # noqa: E402
import recipes  # noqa: E402


def make_row(name="vllm/pyt_vllm_x", registry="vllm", n_gpus="-1", launcher="", args="--model_repo org/X --config c.yaml",
             env_vars=None, multiple_results="perf_X.csv", url="", scripts="scripts/vllm/run.sh", nodes=1,
             precision="", suites=(), skus=("MI300X", "MI325X", "MI350X", "MI355X"), dockerfile="docker/pyt_vllm",
             page="vllm/x", hf_repo="org/X"):
    return {
        "name": name, "registry": registry, "dockerfile": dockerfile, "nodes": nodes, "precision": precision,
        "suites": list(suites), "skus": list(skus), "page": page, "family": "X", "hf_repo": hf_repo,
        "framework": "vLLM", "workload": "inference", "readme": "", "n_gpus": "all" if n_gpus == "-1" else n_gpus,
        "model": hf_repo.split("/")[-1] if hf_repo else name, "image": "vllm/vllm-openai-rocm:v0.29.0", "rocm": "7.2.3",
        "run": {
            "dockerfile_path": dockerfile + ".ubuntu.amd.Dockerfile",
            "context": "." if "primus" in dockerfile else "./docker",
            "build_args": {}, "scripts": scripts, "args": args, "env_vars": env_vars or {},
            "multiple_results": multiple_results, "url": url, "data": "huggingface", "launcher": launcher,
            "n_gpus": n_gpus,
        },
    }


def code(steps):
    return "\n".join(s["code"] for s in steps)


def test_docker_single_node_mirrors_madengine():
    text = code(recipes.docker_steps(make_row(), "MI355X"))
    assert "-f docker/pyt_vllm.ubuntu.amd.Dockerfile" in text
    assert "--build-arg MAD_SYSTEM_GPU_ARCHITECTURE=gfx950" in text
    assert text.rstrip().count("./docker") == 1
    assert "--device=/dev/kfd --device=/dev/dri" in text
    assert '-v "$PWD":/myworkspace -w /myworkspace' in text
    assert "-e MAD_RUNTIME_NGPUS=8" in text and "HIP_VISIBLE_DEVICES" not in text
    assert "-e MAD_OUTPUT_CSV=perf_X.csv" in text
    assert "cp -vLR scripts/vllm/. run_directory/" in text
    assert "cd run_directory && bash run.sh --model_repo org/X --config c.yaml" in text


def test_docker_restricts_gpus_and_selects_isa():
    text = code(recipes.docker_steps(make_row(n_gpus="2"), "MI300X"))
    assert "-e HIP_VISIBLE_DEVICES=0,1" in text and "-e MAD_RUNTIME_NGPUS=2" in text
    assert "MAD_SYSTEM_GPU_ARCHITECTURE=gfx942" in text
    assert "'AMD Instinct MI300X'" in text


def test_docker_clones_model_url_and_directory_scripts():
    row = make_row(url="https://github.com/org/repo", scripts="scripts/huggingface_bert", args="", multiple_results="")
    script, model_dir = recipes.container_script(row)
    assert model_dir == "repo"
    assert script.splitlines() == ["set -e", "git clone https://github.com/org/repo",
                                   "cp -vLR scripts/huggingface_bert/. repo/", "cd repo && bash run.sh"]
    note = recipes.docker_steps(row, "MI300X")[-1]["note"]
    assert "performance:" in note


def test_primus_recipes_fetch_the_submodule():
    row = make_row(dockerfile="docker/primus", registry="primus_train", scripts="scripts/primus_train/run.sh")
    assert "git submodule update --init scripts/Primus" in code(recipes.docker_steps(row, "MI355X"))
    assert "git submodule update --init scripts/Primus" in code(recipes.madengine_steps(row, "MI355X"))


def test_madengine_single_node():
    text = code(recipes.madengine_steps(make_row(), "MI355X"))
    assert "pip install -r requirements.txt" in text
    assert "madengine run --tags vllm/pyt_vllm_x" in text


def test_multi_node_recipes():
    row = make_row(name="vllm_dissag/pyt_vllm_disagg_y", registry="vllm_dissag", launcher="slurm_multi", nodes=2,
                   args="-N 2 -n 2", scripts="scripts/vllm_dissag/run_xPyD_models.slurm", multiple_results="",
                   env_vars={"DOCKER_IMAGE_NAME": "<supply-your-image>", "MODEL_NAME": "DeepSeek-R1"})
    me = code(recipes.madengine_steps(row, "MI355X"))
    assert "--build-on-compute" in me and "--manifest-file build_manifest.json" in me
    assert '"DOCKER_IMAGE_NAME": "<supply-your-image>"' in me
    plain = code(recipes.docker_steps(row, "MI355X"))
    assert "docker push" in plain
    assert "cd scripts/vllm_dissag" in plain
    assert "export DOCKER_IMAGE_NAME='<registry>/<repo>:vllm-dissag-pyt-vllm-disagg-y'" in plain
    assert plain.count("DOCKER_IMAGE_NAME=") == 1
    assert "export MODEL_NAME=DeepSeek-R1" in plain
    assert plain.rstrip().endswith("sbatch -p <partition> -N 2 -n 2 run_xPyD_models.slurm")


def test_multi_node_script_in_subdirectory_submits_from_registry_dir():
    row = make_row(name="kvcache_transfer_bench/k", registry="kvcache_transfer_bench", launcher="slurm_multi",
                   nodes=2, args="", scripts="scripts/kvcache_transfer_bench/scripts/run.slurm", multiple_results="")
    plain = code(recipes.docker_steps(row, "MI300X"))
    assert "cd scripts/kvcache_transfer_bench\n" in plain
    assert plain.rstrip().endswith("sbatch -p <partition> -N 2 -n 2 scripts/run.slurm")


def test_catalog_groups_variants_and_products():
    skus = ["MI300X", "MI325X", "MI350X", "MI355X"]
    rows = [
        make_row(name="vllm/a", page="vllm/llama", hf_repo="meta-llama/Llama"),
        make_row(name="vllm/a_fp4", page="vllm/llama", hf_repo="amd/Llama-MXFP4", precision="mxfp4",
                 skus=["MI350X", "MI355X"]),
        # Primus-style: one card per product, same label -> one variant whose GPU picker selects the card.
        make_row(name="p/MI300X_x", page="primus/x", precision="bf16", skus=["MI300X"], hf_repo=None),
        make_row(name="p/MI355X_x", page="primus/x", precision="bf16", skus=["MI355X"], hf_repo=None),
        # Same punctuation-insensitive family -> same page.
        make_row(name="vllm/oss", page="vllm/gpt-oss-120b", hf_repo="openai/gpt-oss-120b"),
        make_row(name="vllm/oss_w4a8", page="vllm/gpt-oss120b", hf_repo="amd/gpt-oss120b-w", precision="mxfp4"),
    ]
    pages, frameworks = build.catalog({"models": rows, "skus": skus})
    by_page = {p["page"]: p for p in pages}
    llama = by_page["vllm/llama"]
    assert [v["label"] for v in llama["variants"]] == ["Default", "MXFP4"]
    assert llama["org"] == "meta-llama" and llama["hf_repo"] == "meta-llama/Llama"
    assert sorted(llama["variants"][1]["by_sku"]) == ["MI350X", "MI355X"]
    primus = by_page["primus/x"]
    assert len(primus["variants"]) == 1
    assert {s: r["name"] for s, r in primus["variants"][0]["by_sku"].items()} == {"MI300X": "p/MI300X_x", "MI355X": "p/MI355X_x"}
    assert "vllm/gpt-oss120b" not in by_page and len(by_page["vllm/gpt-oss-120b"]["variants"]) == 2
    assert rows[-1]["page"] == "vllm/gpt-oss-120b"
    assert build.default_sku(llama) == "MI355X"
    assert [fw["slug"] for fw in frameworks] == ["vllm", "primus"]


def test_variant_label_uses_distinguishing_suites():
    rows = [make_row(suites=["vllm_disagg", "nixl"]), make_row(suites=["vllm_disagg", "mori_ep"], precision="fp8")]
    assert build._variant_label(rows[0], rows) == "nixl"
    assert build._variant_label(rows[1], rows) == "FP8 · mori_ep"


def test_frozen_recipes_pin_the_commit():
    row = make_row()
    for steps in recipes.recipe(row, "MI355X", ref="abc1234").values():
        assert "git clone https://github.com/ROCm/MAD.git\ncd MAD\ngit checkout abc1234" in steps[0]["code"]
    assert "git checkout" not in code(recipes.recipe(row, "MI355X")["docker"])


def test_support_states():
    variants = [{"label": "Default", "by_sku": {"MI300X": 1, "MI355X": 1}}, {"label": "MXFP4", "by_sku": {"MI355X": 1}}]
    cells = build.support(variants, ["MI300X", "MI325X", "MI355X"])
    assert cells["MI355X"] == {"state": "all", "labels": ["Default", "MXFP4"]}
    assert cells["MI300X"] == {"state": "some", "labels": ["Default"]}
    assert cells["MI325X"]["state"] == "none"


def test_version_order():
    assert sorted(["10", "TheRock preview", "7.2.3", "6.3.0", "7.2"], key=build._version_order) == \
        ["6.3.0", "7.2", "7.2.3", "10", "TheRock preview"]


def fake_snapshot(state="live", version="v26.10"):
    skus = ["MI300X", "MI325X", "MI350X", "MI355X"]
    rows = [
        make_row(name="vllm/a", page="vllm/llama", hf_repo="meta-llama/Llama"),
        make_row(name="vllm/a_fp4", page="vllm/llama", hf_repo="amd/Llama-MXFP4", precision="mxfp4", skus=skus[2:]),
        make_row(name="vllm_dissag/d", registry="vllm_dissag", page="vllm-disaggregated-p-d/deepseek", launcher="slurm_multi",
                 nodes=2, args="-N 2 -n 2", scripts="scripts/vllm_dissag/run.slurm", hf_repo=None),
    ]
    rows[2]["framework"] = "vLLM disaggregated P/D"
    return {
        "schema": 1, "version": version, "release": "v26.10", "state": state, "internal": False,
        "commit": "0123456789abcdef", "date": "2026-10-29", "skus": skus,
        "isa_skus": {"gfx942": skus[:2], "gfx950": skus[2:]}, "models": rows,
        "images": [{"dockerfile": "docker/pyt_vllm", "path": "docker/pyt_vllm.ubuntu.amd.Dockerfile",
                    "image": "vllm/vllm-openai-rocm:v0.29.0", "rocm": "7.2.3", "rocm_source": "status.yaml",
                    "frameworks": ["vLLM", "vLLM disaggregated P/D"], "skus": skus, "models": 3}],
        "registries": [], "suite_tags": {}, "roadmap": [{"model": "Next-Model", "target": "v26.11"}],
        "qa": {"covered": [{"area": "Perf", "scope": "all"}], "not_covered": []},
        "ci": [{"name": "Nightly", "skus": ["MI355X"], "cadence": "nightly"}], "warnings": [],
    }


def test_site_renders_every_page_with_no_broken_links(tmp_path):
    live = fake_snapshot()
    frozen = copy.deepcopy(fake_snapshot(state="frozen", version="v26.9"))
    frozen["models"] = frozen["models"][:1]
    out = str(tmp_path / "site")
    assert build.build_site(out, live, [frozen]) == 2  # raises on any broken link
    for prefix in ("", "v26.9/"):
        for _, rel, _ in build.SECTION_PAGES:
            assert os.path.exists(os.path.join(out, prefix + rel)), prefix + rel
        assert os.path.exists(os.path.join(out, prefix + "models.csv"))
        assert os.path.exists(os.path.join(out, prefix + "models/vllm/llama/index.html"))
    assert os.path.exists(os.path.join(out, "releases/index.html"))
    # Recipes in the frozen tree check out its commit; develop's do not.
    with open(os.path.join(out, "v26.9/models/vllm/llama/index.html")) as f:
        assert "git checkout 0123456789abcdef" in f.read()
    with open(os.path.join(out, "models/vllm/llama/index.html")) as f:
        page = f.read()
    assert "git checkout" not in page
    # The disaggregated page exists only in develop, so v26.9's switcher falls back to its catalog.
    with open(os.path.join(out, "models/vllm-disaggregated-p-d/deepseek/index.html")) as f:
        assert 'value="../../../v26.9/models/index.html"' in f.read()
    with open(os.path.join(out, "index.html")) as f:
        assert "2 added, 0 removed" in f.read()
