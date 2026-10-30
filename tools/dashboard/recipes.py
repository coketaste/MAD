"""
Run recipes for one registered model: the madengine command, and the plain Docker (or
sbatch) commands that reproduce what madengine does without it.

The Docker path mirrors madengine's ContainerRunner: build with the model's Dockerfile and
build context, run with the same device/security options and MAD_* environment, mount
the MAD checkout at /myworkspace, copy the model's script directory into the run
directory and invoke the script with the card's args. SLURM multi-node cards
(distributed.launcher = slurm_multi) run their own launcher script, so their plain
recipe builds and pushes the image and submits that script with sbatch, as the
framework READMEs document.
"""
import json
import os
import re
import shlex

MAD_URL = "https://github.com/ROCm/MAD.git"
ISA_OF = {"MI300X": "gfx942", "MI325X": "gfx942", "MI350X": "gfx950", "MI355X": "gfx950"}
GPUS_PER_NODE = 8

# madengine's AMD `docker run` options (ContainerRunner.run_container) plus the GPU devices.
DOCKER_RUN_OPTS = [
    "--network host --ipc=host --group-add video",
    "--cap-add=SYS_PTRACE --cap-add SYS_ADMIN --device /dev/fuse",
    "--security-opt seccomp=unconfined --security-opt apparmor=unconfined",
    "--device=/dev/kfd --device=/dev/dri",
]

HF_TOKEN_NOTE = "# Hugging Face token; required for gated models"


def is_multi_node(row):
    return row["run"]["launcher"] == "slurm_multi"


def needs_primus(row):
    return "primus" in row["dockerfile"].lower()


def image_name(row):
    return "mad/" + re.sub(r"[^a-z0-9.]+", "-", row["name"].lower()).strip("-")


def continued(head, parts, indent="  "):
    """'head \\\n  part \\\n  part' — one option group per line, like the vLLM recipes."""
    parts = [p for p in parts if p]
    if not parts:
        return head
    return head + " \\\n" + " \\\n".join(indent + p for p in parts)


def placeholder(value):
    return bool(re.fullmatch(r"<[^<>]+>", value.strip()))


def get_mad(row):
    lines = [f"git clone {MAD_URL}", "cd MAD"]
    if needs_primus(row):
        lines.append("git submodule update --init scripts/Primus   # Primus configs and launcher")
    return "\n".join(lines)


def script_invocation(scripts):
    """(directory to copy, command to run in it) — ContainerRunner's rules for 'scripts'."""
    if scripts.endswith(".sh"):
        return os.path.dirname(scripts), "bash " + os.path.basename(scripts)
    if scripts.endswith(".py"):
        return os.path.dirname(scripts), "python3 " + os.path.basename(scripts)
    return scripts, "bash run.sh"


def gpu_count(run):
    n = run["n_gpus"]
    return GPUS_PER_NODE if n in ("-1", "", "all") else int(n)


# --------------------------------------------------------------------------- madengine


def madengine_steps(row, sku):
    tag = row["name"]
    steps = [
        {"title": "Get MAD and madengine", "code": get_mad(row) + "\npip install -r requirements.txt"},
    ]
    if not is_multi_node(row):
        steps.append({
            "title": "Run",
            "code": f"export MAD_SECRETS_HFTOKEN=<your-hf-token>   {HF_TOKEN_NOTE}\n"
                    + continued(f"madengine run --tags {tag}", ["--live-output"]),
            "note": f"madengine detects the {sku} ({ISA_OF[sku]}), builds the image, runs the benchmark and "
                    "appends the result to perf.csv in the MAD directory.",
        })
        return steps

    env = {k: v for k, v in row["run"]["env_vars"].items() if placeholder(v)}
    run_opts = ["--manifest-file build_manifest.json"]
    if env:
        run_opts.append("--additional-context " + shlex.quote(json.dumps({"env_vars": env})))
    steps.append({
        "title": "Build on a compute node and push",
        "code": f"export MAD_SECRETS_HFTOKEN=<your-hf-token>   {HF_TOKEN_NOTE}\n" + continued(
            f"madengine build --tags {tag}",
            ["--registry <namespace>/<repo> --build-on-compute",
             "--additional-context " + shlex.quote(json.dumps({"slurm": {"partition": "<partition>"}}))]),
    })
    steps.append({
        "title": f"Run on {row['nodes']} nodes",
        "code": continued(f"madengine run --tags {tag}", run_opts),
        "note": "Run from a SLURM login node. Replace the <placeholders>; the node count and launcher "
                "come from the model card.",
    })
    return steps


# --------------------------------------------------------------------------- docker


def build_command(row, sku, image):
    run = row["run"]
    build_args = {"MAD_SYSTEM_GPU_ARCHITECTURE": ISA_OF[sku], **run["build_args"]}
    return continued("docker build --network=host --pull", [
        f"-t {shlex.quote(image)}",
        f"-f {shlex.quote(run['dockerfile_path'])}",
        *(f"--build-arg {k}={shlex.quote(str(v))}" for k, v in build_args.items()),
        shlex.quote(run["context"]),
    ])


def container_script(row):
    """What madengine runs inside the container, as a `bash -c` script (one step per line)."""
    run = row["run"]
    script_dir, invoke = script_invocation(run["scripts"])
    steps = []
    if run["url"]:
        model_dir = run["url"].rstrip("/").split("/")[-1]
        steps.append(f"git clone {run['url']}")
    else:
        model_dir = "run_directory"
        steps.append(f"mkdir -p {model_dir}")
    steps.append(f"cp -vLR {script_dir}/. {model_dir}/")
    args = " ".join(shlex.quote(a) for a in shlex.split(run["args"])) if run["args"] else ""
    steps.append(f"cd {model_dir} && {invoke} {args}".rstrip())
    return "\n".join(["set -e"] + steps), model_dir


def docker_steps(row, sku):
    if is_multi_node(row):
        return slurm_steps(row, sku)
    run = row["run"]
    image = image_name(row)
    n = gpu_count(run)
    env = [
        f"-e MAD_MODEL_NAME={shlex.quote(row['name'])}",
        f"-e MAD_SYSTEM_GPU_ARCHITECTURE={ISA_OF[sku]}",
        f"-e MAD_SYSTEM_GPU_PRODUCT_NAME={shlex.quote('AMD Instinct ' + sku)}",
        f"-e MAD_SYSTEM_NGPUS={GPUS_PER_NODE} -e MAD_RUNTIME_NGPUS={n}",
    ]
    if n < GPUS_PER_NODE:
        env.append("-e HIP_VISIBLE_DEVICES=" + ",".join(str(i) for i in range(n)))
    if run["multiple_results"]:
        env.append(f"-e MAD_OUTPUT_CSV={shlex.quote(run['multiple_results'])}")
    env.append('-e MAD_SECRETS_HFTOKEN="$HF_TOKEN"')
    env += [f"-e {k}={shlex.quote(v)}" for k, v in run["env_vars"].items()]
    script, model_dir = container_script(row)
    run_cmd = continued("docker run --rm -it", [
        *DOCKER_RUN_OPTS,
        '-v "$PWD":/myworkspace -w /myworkspace',
        *env,
        image,
        # One command per line inside the quotes; the indent is whitespace to the inner shell.
        "bash -c " + shlex.quote(script).replace("\n", "\n    "),
    ])
    if run["multiple_results"]:
        result = f"Results are written to {model_dir}/{run['multiple_results']} in the MAD directory."
    else:
        result = "The script prints the result as a `performance: <value> <unit>` line."
    return [
        {"title": "Get MAD", "code": get_mad(row)},
        {"title": "Build the image", "code": build_command(row, sku, image)},
        {"title": "Run the benchmark", "code": f"export HF_TOKEN=<your-hf-token>   {HF_TOKEN_NOTE}\n{run_cmd}",
         "note": f"Uses {'all ' + str(n) if n == GPUS_PER_NODE else n} GPU{'s' if n > 1 else ''}. {result}"},
    ]


def slurm_steps(row, sku):
    run = row["run"]
    image = "<registry>/<repo>:" + image_name(row).split("/", 1)[1]
    # Submit from the registry directory (the launchers find their helpers via
    # SLURM_SUBMIT_DIR), naming the script relative to it.
    submit_dir = os.path.join("scripts", row["registry"])
    script = os.path.relpath(run["scripts"], submit_dir)
    # The launchers start one container per node from DOCKER_IMAGE_NAME.
    exports = [f"export DOCKER_IMAGE_NAME={shlex.quote(image)}"]
    exports += [f"export {k}={shlex.quote(v)}" for k, v in run["env_vars"].items() if k != "DOCKER_IMAGE_NAME"]
    exports.append(f"export MAD_MODEL_NAME={shlex.quote(row['name'])}")
    if run["multiple_results"]:
        exports.append(f"export MAD_OUTPUT_CSV={shlex.quote(run['multiple_results'])}")
    sbatch_opts = run["args"] or f"-N {row['nodes']} -n {row['nodes']}"
    return [
        {"title": "Get MAD", "code": get_mad(row)},
        {"title": "Build and push an image every node can pull",
         "code": build_command(row, sku, image) + f"\ndocker push {shlex.quote(image)}"},
        {"title": f"Submit the {row['nodes']}-node job",
         "code": f"cd {submit_dir}\n" + "\n".join(exports)
                 + f"\nsbatch -p <partition> {sbatch_opts} {script}",
         "note": "Replace the <placeholders>. See the framework guide for node, NIC and shared-storage "
                 "prerequisites."},
    ]


def recipe(row, sku):
    """Both methods for one model on one GPU product."""
    return {
        "madengine": madengine_steps(row, sku),
        "docker": docker_steps(row, sku),
    }
