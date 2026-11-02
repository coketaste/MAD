#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License. See docker/pyt_fam.ubuntu.amd.Dockerfile for the full text.
"""Save agent-camera frames from four LIBERO episodes.

The tasks are the four instructions in the source post's rollout figures.
The frames come from this process's simulator, one trial each, not from
those published images.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

from PIL import Image, ImageDraw

FAM = Path(os.environ.get("FAM_ROOT", "/opt/faster-action-models"))
BATCHED = FAM / "eval" / "native" / "fastwam_libero_batched.py"
CONFIGS = FAM / "FastWAM" / "configs_baseline"
TASKS = (
    ("libero_spatial", 1, "spatial",
     "Pick up the black bowl next to the ramekin and place it on the plate."),
    ("libero_object", 0, "object",
     "Pick up the alphabet soup and place it in the basket."),
    ("libero_goal", 0, "goal",
     "Open the middle drawer of the cabinet."),
    ("libero_10", 1, "long",
     "Put both the cream cheese box and the butter in the basket."),
)


def load_batched():
    spec = importlib.util.spec_from_file_location("fastwam_libero_batched", BATCHED)
    if spec is None or spec.loader is None:
        sys.exit(f"cannot load {BATCHED}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def namespace(steps: int):
    return argparse.Namespace(
        config_dir=str(CONFIGS),
        task="libero_uncond_2cam224_1e-4",
        ckpt=os.environ["MODEL_CKPT"],
        dataset_stats=os.environ["DATASET_STATS"],
        steps=steps,
        trials=1,
        seed=42,
        override=["model.redirect_common_files=false"],
    )


def sample(frames: list, count: int = 5) -> list:
    if len(frames) < 2:
        sys.exit("episode produced no camera frames")
    pick = [round(i * (len(frames) - 1) / (count - 1)) for i in range(count)]
    return [frames[i] for i in pick]


def text_image(text: str, scale: int = 3) -> Image.Image:
    scratch = Image.new("RGB", (len(text) * 6 + 8, 12), "white")
    ImageDraw.Draw(scratch).text((2, 0), text, fill=(26, 26, 26))
    return scratch.resize((scratch.width * scale, scratch.height * scale), Image.Resampling.NEAREST)


def row_image(frames: list, caption: str) -> Image.Image:
    tiles = [Image.fromarray(frame) for frame in sample(frames)]
    label = text_image(caption)
    width = sum(tile.width for tile in tiles)
    height = label.height + 8 + tiles[0].height
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(label, (4, 4))
    x = 0
    for tile in tiles:
        canvas.paste(tile, (x, label.height + 8))
        x += tile.width
    return canvas


def one_episode(mod, suite: str, task_id: int, cfg, policy) -> tuple[list, bool, str]:
    import torch
    from experiments.libero.libero_utils import get_libero_dummy_action, get_libero_image
    from instinct_sim_libero import HipLiberoVectorEnv
    from libero.libero import benchmark, get_libero_path

    task = benchmark.get_benchmark_dict()[suite]().get_task(task_id)
    init_states = torch.load(
        Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
        weights_only=False,
    )
    env_args = {
        "bddl_file_name": str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
        "camera_heights": 256,
        "camera_widths": 256,
    }
    wait = int(cfg.EVALUATION.get("num_steps_wait", 5))
    replan = int(cfg.EVALUATION.get("replan_steps", 5))
    limit = mod.EL._get_max_steps(suite) + wait
    venv = HipLiberoVectorEnv(env_args, num_envs=1, seed=int(cfg.seed))
    frames: list = []
    done = False
    try:
        obs = {0: venv.reset([init_states[0]])[0]}
        pending: list = []
        for t in range(limit):
            frames.append(np.ascontiguousarray(get_libero_image(obs[0])["image"]))
            if t >= wait and not pending:
                pending = policy([obs[0]], task.language)[0][:replan].tolist()
            action = get_libero_dummy_action() if t < wait else pending.pop(0)
            nxt, _, flag, _ = venv.step({0: action})[0]
            obs[0] = nxt
            done = bool(flag)
            if done and t >= wait:
                frames.append(np.ascontiguousarray(get_libero_image(obs[0])["image"]))
                venv.finish(0)
                break
        else:
            venv.finish(0)
    finally:
        venv.close()
    return frames, done, task.language


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("eager", "geak@2"))
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out) / ("geak2" if args.arm == "geak@2" else args.arm)
    out.mkdir(parents=True, exist_ok=True)
    mod = load_batched()
    cfg = mod.compose_cfg(namespace(args.steps))
    model, processor, device, horizon, width, height = mod.load_model(cfg, ())
    policy = mod.BatchedPolicy(cfg, model, processor, device, horizon, width, height, 1)
    successes = 0
    rows = []
    for suite, task_id, slug, _expected in TASKS:
        frames, done, language = one_episode(mod, suite, task_id, cfg, policy)
        successes += int(done)
        caption = f"{args.arm}  {'finished' if done else 'not finished'}"
        row_image(frames, caption).save(out / f"{slug}.png")
        record = {
            "arm": args.arm,
            "suite": suite,
            "task_id": task_id,
            "language": language,
            "success": done,
            "frames": len(frames),
        }
        (out / f"{slug}.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"rollout {args.arm} {suite} t{task_id}: {'1/1' if done else '0/1'} {language}", flush=True)
        rows.append(record)
    summary = {"arm": args.arm, "successes": successes, "episodes": len(TASKS), "tasks": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
