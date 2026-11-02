# Hands-on MAD: World Action Models

This post walks MAD's skills, agents, and workflows on one workload: FastWAM on LIBERO, in a container on one AMD Instinct MI350X (gfx950). Every number is from a container run on 2026-11-02. The transcripts are in `docs/blog/transcripts/`. The two comparison notes were computed from those CSVs. Latency runs call the released FastWAM and CAE checkpoints. The accuracy smokes run the FastWAM checkpoint in LIBERO's simulator: simulated episodes, one GPU each. The host has eight GPUs.

Each section ends with the prompt to reproduce it. Paste it into Claude Code from the MAD repo root.

A skill that builds or runs is explicit-only (`disable-model-invocation: true`). In Claude Code the prompt has to start with the slash command, `/mad-add-model`, `/mad-benchmark`, `/mad-profile`, or `/mad-validate-model`. A plain sentence does not load the skill, so the pre-flight, the subagent, and `mad-run.sh` never start. `mad-report` and `mad-validate-model` do not set that flag, so a sentence can load them. The slash form is still the one used below.

The three workflows are not slash commands. There is no `/mad-benchmark-sweep`. They live in `.claude/workflows/` and start through Claude Code's Workflow tool. `args` is a JSON object, not a string. They plan unless `execute` is `true`. GPU runs go through `.claude/skills/mad-common/mad-run.sh`. On this host a plain `madengine run --tags` tried to pull the base image and got `401` from Docker Hub, so the rerun built a manifest with `--use-image` against the image already present and passed that manifest as `-m`. If the pull is denied, switch to the local image the way `docs/blog/transcripts/05-interleaved-latency.txt` does.

## Four arms, and why that count

The sweep is `mad-benchmark-sweep`. Each tag is one cell. Cells run one at a time. There is no precision axis: a different configuration is a different model entry.

Four FastWAM-on-LIBERO latency arms:

- **eager** — PyTorch eager. The baseline, and the arm the operator profile uses.
- **torch.compile** — the compiler step, so the GEAK step is compared with compile on the same GPU.
- **GEAK** — the kernel tree shipped in `src/fastwam-geak`, at 10 denoising steps.
- **GEAK at 2 denoising steps** — fewer steps. The smoke checks that the success rate still holds.

CAE is a second checkpoint, so it is its own benchmark, not a fifth sweep cell. DiT4DiT and RoboCasa are not registered. They appear once, as a plan-mode tag that does not resolve. Two smoke runs, one trial per task, are the accuracy check: 40 simulated episodes each.

The cited latency table is one madengine model that times all four arms inside one session. The sweep still runs the four arms as separate cells, because that is the workflow this post is teaching. Both tables are below. The one-session table is the citation.

## 1. Skill, agent, workflow

A **skill** is the procedure: `mad-benchmark`, `mad-profile`, `mad-tune`, `mad-report`, `mad-add-model`. The ones that build or run are explicit.

An **agent** carries the procedure out. `mad-benchmark` and `mad-profile` run in `mad-benchmark-runner`. `mad-tune` runs in `mad-tuner`. `mad-report` runs in `mad-perf-analyst`. The analyst is read-only. It never starts a GPU job, and it never compares a millisecond with a success rate.

A **workflow** fans the procedure out. `mad-benchmark-sweep`, `mad-tune-search`, and `mad-optimize` live in `.claude/workflows/` and run in Claude Code. They plan by default. `--execute` is what touches GPUs. `mad-optimize` is the one that may come back with a different kernel tree. `mad-tune-search` still stops if a tracked file changes. `docs/mad-optimize.md` is the contract.

![From mad-add-model, the sweep splits into the one-session latency path and the four cells. Both paths reach mad-report. The dashed arrow is the one-session result reaching the report directly. mad-tune-search also continues to the mad-optimize plan.](figures/mad-flow.svg)

FastWAM is a world action model. A video backbone predicts how the scene may evolve, and an action expert denoises an action chunk conditioned on that prediction. The two share attention. At inference the future-frame prediction is not displayed. The ladder starts at eager because that is the unmodified PyTorch path, and the operator profile is on that same arm.

![A VLA maps observations to an action. A WAM predicts the scene, then selects the action. FastWAM is the lower path.](figures/vla-vs-wam.svg)

![FastWAM's video DiT and action DiT share attention and emit a 32 by 7 action chunk. The video prediction is not displayed.](figures/dual-dit.svg)

**Prompt**

```text
In the MAD repo, explain how a skill, an agent, and a workflow differ, using mad-benchmark, mad-benchmark-runner, and mad-benchmark-sweep. Then name mad-tune-search and mad-optimize, and say which one is allowed to return a different kernel tree. Do not run anything.
```

## 2. One directory, one image, no nested docker

FastWAM's own launchers start a container. madengine has already started one, so `scripts/pyt_fam/run.sh` calls the in-image Python entry points and stops there.

The image is `docker/pyt_fam.ubuntu.amd.Dockerfile`. It clones `AMD-AGI/faster-action-models` at `a5fa73101988870e51f04fe1e7be5da56dda3841`. `url` on every registry entry is empty, because a second clone collides with files the registry directory already contains. Nine entries share that image. `docs/blog/transcripts/02-discover.txt` lists the eight that existed when it was captured. `pyt_fam_fastwam_rollout` is the ninth, added for the frame capture. Latency is `ms` (lower is better). Accuracy is `success_rate` (higher is better).

```text
$ python3 tools/validation/model_config_validator.py --model pyt_fam_fastwam_latency
✅ All validation checks passed!
```

`docs/blog/transcripts/00-validator.txt`.

**Prompt**

```text
/mad-validate-model pyt_fam_fastwam_latency

Quote the pass or fail line. Do not build an image and do not start a GPU job.
```

## 3. Adding one entry to the directory that already exists

`/mad-add-model` is how a model gets into MAD. It is explicit-only, same as the run skills. It asks for the name, the repository URL, training or inference, the GPU count, the precision, and the owner. It checks that the name is unused, then it writes. `madengine discover --tags <name>` is that check. A name that already lists a model must not get a second entry.

FastWAM is not a new directory per arm. The Dockerfile and `scripts/pyt_fam/run.sh` already exist, so a new arm does not repeat those steps. It is one object in `scripts/pyt_fam/models.json`. Paths in that file are relative to `scripts/pyt_fam/`. `dockerfile` points at `../../docker/pyt_fam`, and madengine appends `.ubuntu.amd.Dockerfile`. `url` stays empty. `run.sh` is what prints `performance: <value> <unit>`.

`pyt_fam_fastwam_rollout` is the entry added for the frame capture. Its unit is `success_rate`.

```json
{
  "name": "pyt_fam_fastwam_rollout",
  "dockerfile": "../../docker/pyt_fam",
  "scripts": "run.sh",
  "url": "",
  "n_gpus": "1",
  "owner": "mad.support@amd.com",
  "training_precision": "",
  "timeout": 14400,
  "tags": ["pyt", "pyt_fam", "inference", "fam", "fastwam", "rollout"],
  "args": "--mode frames"
}
```

**Prompt**

```text
/mad-add-model

The model is already registered. Do not write files and do not create a second entry. Walk the skill against pyt_fam_fastwam_rollout and stop when the duplicate check finds it.

Answers, so you do not have to ask: framework pyt, project fam, workload fastwam_rollout, full name pyt_fam_fastwam_rollout. Repository URL empty. Inference. 1 GPU. Precision empty. Owner mad.support@amd.com. The directory is scripts/pyt_fam, dockerfile is ../../docker/pyt_fam, scripts is run.sh, args are --mode frames, timeout is 14400. Tags are pyt, pyt_fam, inference, fam, fastwam, rollout.

Run madengine discover --tags pyt_fam_fastwam_rollout and quote the model it lists. Then run python3 tools/validation/model_config_validator.py --model pyt_fam_fastwam_rollout and quote the pass or fail line.
```

## 4. Discover, then the sweep

Pre-flight and discover: `docs/blog/transcripts/01-preflight.txt`, `docs/blog/transcripts/02-discover.txt`. That discover lists eight models as `pyt_fam/<name>`. The registry now also has `pyt_fam_fastwam_rollout`.

The sweep in plan mode (`docs/blog/transcripts/03-sweep-plan.txt`, workflow `wf_35f3d00e-62b`, `execute: false`) prints one command per cell and runs nothing. Headline from that session: 4 cells planned, 0 ran. Each command carries the fam cache mount. In `docker_mounts` the key is the container path and the value is the host path, so `/myworkspace/fam_cache` maps to `/var/tmp/fam_cache`. The session notes that call this reversed are mistaken. The latency run log mounts it as `-v /var/tmp/fam_cache:/myworkspace/fam_cache`, and `FAM_CACHE` points at the container path.

```text
madengine run --tags 'pyt_fam/pyt_fam_fastwam_eager' --live-output -o perf_pyt_fam_pyt_fam_fastwam_eager.csv --additional-context '{"docker_mounts":{"/myworkspace/fam_cache":"/var/tmp/fam_cache"},"docker_env_vars":{"FAM_CACHE":"/myworkspace/fam_cache"}}'
madengine run --tags 'pyt_fam/pyt_fam_fastwam_compile' --live-output -o perf_pyt_fam_pyt_fam_fastwam_compile.csv --additional-context '{"docker_mounts":{"/myworkspace/fam_cache":"/var/tmp/fam_cache"},"docker_env_vars":{"FAM_CACHE":"/myworkspace/fam_cache"}}'
madengine run --tags 'pyt_fam/pyt_fam_fastwam_geak' --live-output -o perf_pyt_fam_pyt_fam_fastwam_geak.csv --additional-context '{"docker_mounts":{"/myworkspace/fam_cache":"/var/tmp/fam_cache"},"docker_env_vars":{"FAM_CACHE":"/myworkspace/fam_cache"}}'
madengine run --tags 'pyt_fam/pyt_fam_fastwam_geak2' --live-output -o perf_pyt_fam_pyt_fam_fastwam_geak2.csv --additional-context '{"docker_mounts":{"/myworkspace/fam_cache":"/var/tmp/fam_cache"},"docker_env_vars":{"FAM_CACHE":"/myworkspace/fam_cache"}}'
```

The same workflow with `pyt_fam_dit4dit` added stays in plan mode (`docs/blog/transcripts/04-sweep-plan-unregistered.txt`). That tag is not in the registry. The cell status is `error`: `No models found corresponding to the given tag: pyt_fam_dit4dit`. The other four stay planned. That prompt stays `execute: false` on purpose. `execute: true` would start the four registered cells.

**Prompt**

```text
Run the mad-benchmark-sweep workflow. Use the Workflow tool. scriptPath is .claude/workflows/mad-benchmark-sweep.js. args must be a JSON object, not a string:

{"tags":["pyt_fam_fastwam_eager","pyt_fam_fastwam_compile","pyt_fam_fastwam_geak","pyt_fam_fastwam_geak2"],"execute":true,"additionalContext":{"docker_mounts":{"/myworkspace/fam_cache":"/var/tmp/fam_cache"},"docker_env_vars":{"FAM_CACHE":"/myworkspace/fam_cache"}}}

This runs four cells, one at a time, on a GPU. Quote each performance line. Lower milliseconds are better. If docker pull of the base image returns 401, stop and reuse the local image with madengine build --use-image and madengine run -m, as in docs/blog/transcripts/05-interleaved-latency.txt.
```

**Prompt for the unregistered tag**

```text
Run mad-benchmark-sweep. scriptPath is .claude/workflows/mad-benchmark-sweep.js. args is a JSON object:

{"tags":["pyt_fam_fastwam_eager","pyt_fam_fastwam_compile","pyt_fam_fastwam_geak","pyt_fam_fastwam_geak2","pyt_fam_dit4dit"],"execute":false}

Do not run madengine run. Quote the dit4dit cell status and confirm the other four stay planned. Leave execute false. execute true would start the four registered cells.
```

## 5. The cited table, and the sweep's own table

The cited run is one container, four arms, three interleaved rounds (`docs/blog/transcripts/05-interleaved-latency.txt`, run `fam-latency-5`). Device: AMD Instinct MI350X, torch `2.11.0+rocm7.14.0`.

| arm | steps | rounds (ms) | median ms | spread |
|---|---:|---|---:|---:|
| eager | 10 | 216.579, 223.929, 229.188 | 223.9293 | 5.63% |
| compile | 10 | 87.373, 88.973, 89.335 | 88.9732 | 2.21% |
| geak | 10 | 24.845, 24.899, 24.946 | 24.8986 | 0.41% |
| geak@2 | 2 | 9.921, 9.931, 9.959 | 9.9310 | 0.38% |

The median column is `perf_fam_latency.csv`. That file has the four performance lines and no status or host columns; the host and the success of the run are in the transcript. The round column is the three lines the log printed. Spread is `(max − min) / median`. Against eager, compile is 2.52×, the shipped GEAK tree is 8.99×, and GEAK at 2 steps is 22.55×. GEAK is 3.57× past compile.

| arm | median ms | times the eager median |
|---|---:|---:|
| eager | 223.9293 | 1.00 |
| compile | 88.9732 | 2.52 |
| geak | 24.8986 | 8.99 |
| geak@2 | 9.9310 | 22.55 |

![One session. The dot is the median of three rounds and the bar is min to max.](figures/latency-ladder.svg)

`docs/blog/transcripts/10-report-latency.txt` is that comparison, computed from the CSV. It is not a `mad-report` agent session. Lower is better. It does not mention the smoke rate.

The four cells then ran one at a time through `mad-run.sh` on local-image manifests (`docs/blog/transcripts/06-sweep-execute.txt`). That is the same one-at-a-time rule as `mad-benchmark-sweep --execute`, and it was not a Claude `--execute` of the workflow. The plan-mode commands write `perf_pyt_fam_pyt_fam_fastwam_*.csv`. These runs wrote `perf_rerun_*.csv`:

| cell | ms | file |
|---|---:|---|
| eager | 221.4562 | `perf_rerun_eager.csv` |
| compile | 90.6402 | `perf_rerun_compile.csv` |
| geak | 24.6736 | `perf_rerun_geak.csv` |
| geak@2 | 9.8859 | `perf_rerun_geak2.csv` |

Against this session's eager number the ratios are 2.44×, 8.98×, and 22.40×. The order matches the cited table.

**Prompt**

```text
/mad-benchmark pyt_fam_fastwam_latency

Mount the fam cache: additional-context {"docker_mounts":{"/myworkspace/fam_cache":"/var/tmp/fam_cache"},"docker_env_vars":{"FAM_CACHE":"/myworkspace/fam_cache"}}. Write the result to perf_fam_interleaved.csv. The skill starts the run through mad-run.sh. Read perf_fam_latency.csv and report the four medians in ms. Lower is better. Do not compare them with a success rate.

If docker pull of the base image returns 401, stop and reuse the local image with madengine build --tags pyt_fam_fastwam_latency --use-image <local tag> and madengine run -m <manifest>, as in docs/blog/transcripts/05-interleaved-latency.txt.
```

## 6. Operator profile

Operator depth answers which PyTorch categories dominate kernel time. One depth only, on `pyt_fam_fastwam_eager`. Stacks off, 2000 ms window, warmup 25 s (`docs/blog/transcripts/07-profile-eager.txt`).

`performance: 248.8302 ms`. That number was measured under the profiler. It is evidence, not a score. TraceLens status `SUCCESS`. Kernel time in the window:

![Eager kernel-time mix from the 2000 ms operator window.](figures/operator-mix.svg)

| category | share |
|---|---:|
| elementwise | 49.91% |
| GEMM | 24.86% |
| reduce | 8.57% |
| SDPA | 8.17% |
| normalization | 3.53% |
| other | 3.18% |
| convolution | 1.78% |

The GPU timeline in that window is 32.57% computation, 0.04% memcpy, and 67.40% idle. The capture has no `nn.Module` events, so it does not attribute time to the action-denoising module.

The eager operator file is a Chrome trace: `torch_profiler_output/libkineto_trace_1240.json`, device AMD Instinct MI350X. [Perfetto](https://ui.perfetto.dev) opens it with **Open trace file**. It is 554 MB, so it stays in the run output and is not copied into this post. The table above is the summary of that file. An older trace, `libkineto_trace_1558.json`, is from the previous day and is not part of this capture.

**Prompt**

```text
/mad-profile pyt_fam_fastwam_eager operator

One run, through mad-run.sh. Mount the fam cache the same way as the latency run. Set TORCH_PROFILE_ITERATIONS=0, TORCH_PROFILE_DURATION_MS=2000, TORCH_PROFILE_WARMUP_S=25, TORCH_PROFILE_WITH_STACKS=0, and TRACELENS_MODE=pytorch. Tools are torch_profiler_dynolog and tracelens. Do not add rocm_trace_lite or rocprofv3. The performance line from this run is evidence, not a score. Read tracelens_output and report kernel time by category. Say whether an nn.Module share is present.
```

## 7. Dispatch trace, one tuner lever, then the smoke gate

`mad-tune-search` in plan mode (`docs/blog/transcripts/08-tune-search-plan.txt`, workflow `wf_e2cf34b5-82f`) resolved `pyt_fam_fastwam_geak`, target latency, depth dispatch, two repeats, one candidate, `execute: false`. The candidate it named is `PYTORCH_TUNABLEOP_ENABLED=1`, with `PYTORCH_TUNABLEOP_TUNING=1` on the first run and `TUNING=0` after that so later runs reuse the results file. That is an environment lever. It does not edit a tracked file.

The plan read a trace already on disk and called `triton_mm` about 67% of kernel time. That file's provenance is the previous day. The dispatch run that followed (`docs/blog/transcripts/09-tune-execute.txt`) is a new `rocm_trace_lite` capture. `performance: 26.8858 ms` is profiled, not a clean score. `triton_mm` is 66.0% of kernel time in that new trace. GPU utilization in that file is 5.39% over 75.7 s of wall, because the trace includes startup. The chart is the kernel share, not that utilization.

![Top kernels on the shipped GEAK tree. triton_mm is 66% of kernel time. These eight bars are the head of the list and do not add up to 100%.](figures/kernel-share.svg)

That dispatch file is a Chrome trace, [traces/geak-dispatch.json.gz](traces/geak-dispatch.json.gz) (about 3 MB, 285,765 GPU slices, one MI350X). The two figures are that file open in [Perfetto](https://ui.perfetto.dev). The same gzip opens there with **Open trace file**. Nearly all of the kernels land on one hardware queue. The other three queues appear at the end of the capture. Perfetto reports three import notes on this file. The queues still draw.

![Perfetto timeline of the GEAK dispatch capture, about 75 seconds. Queue 3 holds the kernels.](figures/perfetto-dispatch-overview.png)

The next view is 50 µs at 27.983 s on that queue. The wider slices are `triton_mm`. The narrow ones are `at::native::vectorized_elementwise_kernel`.

![Perfetto, 50 microseconds at 27.983 seconds. triton_mm and elementwise kernels on queue 3.](figures/perfetto-dispatch-zoom.png)

TunableOp chooses among rocBLAS GEMM implementations. It does not rewrite `triton_mm`. The plan still named it, so it was measured, and then not kept.

Four earlier files, `perf_rerun_tune_base_r1.csv`, `perf_rerun_tune_base_r2.csv`, `perf_rerun_tune_top_r1.csv`, and `perf_rerun_tune_top_r2.csv`, are not scores. They reused the dispatch manifest, and that manifest still had `rocm_trace_lite` in it. A later `madengine run -m` keeps tools a previous run wrote into the same file. Those four ran under the profiler. They are discarded.

The repeats below used a new manifest from `madengine build --use-image`, with no tools key (`docs/blog/transcripts/09-tune-execute.txt`):

| run | environment | result |
|---|---|---|
| baseline | cache mount only | 24.9009 ms |
| baseline | cache mount only | 24.9296 ms |
| plan's first step | `ENABLED=1`, `TUNING=1` | 25.2935 ms, and no results file |
| `ENABLED=1` only | tuning left unset | no performance line |
| `ENABLED=1` only | tuning left unset | 24.9987 ms |

The two baseline samples have median 24.9153 ms and spread 0.12%, `(max − min) / median`. The `TUNING=1` run was pointed at `/var/tmp/fam_cache/tunableop_results.csv`. The run finished, and that file was not created. The log has no TunableOp activity. There is no tuned file for a later `TUNING=0` run to reuse, so those score runs were not taken. One search run is not a gate sample.

`ENABLED=1` with tuning unset was tried twice. The first raised `InductorError: CantSplit: 640 not divisible by 392` and wrote no performance line. The second finished at 24.9987 ms, slower than both baseline samples. The gate needs two candidate samples. This side does not have them.

The lever is not kept. There is no chart for it. The speedup in the cited table is the tree the image already contained. MAD did not produce it.

The 2-step arm needs a success rate beside it, in a different table. Two smokes, one trial on each of 40 tasks (`docs/blog/transcripts/12-smokes-and-cae.txt`, `docs/blog/transcripts/10-report-smoke.txt`):

| model | spatial | object | goal | long | rate |
|---|---:|---:|---:|---:|---:|
| eager | 10/10 | 10/10 | 10/10 | 10/10 | 1.0000 success_rate |
| GEAK at 2 steps | 10/10 | 10/10 | 10/10 | 10/10 | 1.0000 success_rate |

These are one-trial smokes on MI350X. `docs/blog/transcripts/10-report-smoke.txt` compares the two rates with each other. It was computed from the two CSVs. It is not a `mad-report` agent session, and it does not compare them with milliseconds.

The frames below are a separate capture, `pyt_fam_fastwam_rollout`, run `fam-rollout-2` (`docs/blog/transcripts/13-rollouts.txt`). Four instructions, one trial each, on the agent camera of this host's simulator. The top row is eager. The bottom row is GEAK at 2 steps. Five frames are taken from start to finish. All eight episodes finished. An earlier start of this model failed before any frame was saved; that row in `perf_rerun_rollout.csv` is the failure, and the success row is these eight episodes. This rate is not the 40-task smoke.

![Spatial: pick up the black bowl next to the ramekin and place it on the plate. Eager on top, GEAK at 2 steps below. Both finished.](figures/rollout-spatial.png)

![Object: pick up the alphabet soup and place it in the basket. Eager on top, GEAK at 2 steps below. Both finished.](figures/rollout-object.png)

![Goal: open the middle drawer of the cabinet. Eager on top, GEAK at 2 steps below. Both finished.](figures/rollout-goal.png)

![Long: put both the cream cheese box and the butter in the basket. Eager on top, GEAK at 2 steps below. Both finished.](figures/rollout-long.png)

**Prompt**

```text
Run mad-tune-search. Use the Workflow tool. scriptPath is .claude/workflows/mad-tune-search.js. args must be a JSON object:

{"tag":"pyt_fam_fastwam_geak","target":"latency","depth":"dispatch","repeats":2,"candidates":1,"execute":true,"additionalContext":{"docker_mounts":{"/myworkspace/fam_cache":"/var/tmp/fam_cache"},"docker_env_vars":{"FAM_CACHE":"/myworkspace/fam_cache"}}}

This runs on a GPU, one step at a time, through mad-run.sh. Do not put rocm_trace_lite and rocprofv3 in the same command. A performance line taken under the profiler is evidence, not a score. Do not reuse a manifest the profile wrote tools into. Build a fresh one with madengine build --use-image for the clean repeats.

The lever is an environment change, not a file edit. Point PYTORCH_TUNABLEOP_FILENAME at the cache directory, not the repo. Set PYTORCH_TUNABLEOP_ENABLED=1 and PYTORCH_TUNABLEOP_TUNING=1 for the search run. Run the scoring repeats with TUNING=0 only if that search wrote a results file. Do not edit a tracked file.

Keep the lever only when both sides have two clean samples and the improvement beats max(baseline spread, candidate spread, 1%). Lower milliseconds win.
```

**Prompt for the smokes**

```text
/mad-benchmark pyt_fam_fastwam_eager_smoke,pyt_fam_fastwam_geak2_smoke

Mount the fam cache the same way as the latency run. Report each performance line. The metric is success_rate. Higher is better. Do not place these rates in a table with milliseconds.
```

**Prompt for the rollouts**

```text
/mad-benchmark pyt_fam_fastwam_rollout

Mount the fam cache the same way as the latency run. It is one trial of four tasks for eager and for GEAK at 2 steps. Report the eight episode lines. Do not treat that rate as the 40-task smoke.
```

## 8. CAE is a second model

CAE-FastWAM is a different checkpoint (`amd/CAE-FastWAM`), so it is `pyt_fam_cae_fastwam_latency`, run at 10 denoising steps. It is not a fifth cell of the sweep. The run uses the released weights and does not retrain CAE.

```text
performance: 13.5933 ms
```

File: `perf_rerun_cae.csv`.

**Prompt**

```text
/mad-benchmark pyt_fam_cae_fastwam_latency

Mount the fam cache the same way as the latency run. Report the performance line. It is milliseconds, lower is better. Do not describe this run as a fifth cell of the four-arm sweep, and do not retrain CAE.
```

## 9. mad-optimize, planned and not spent

GEAK's entry point for a kernel tree is `kernel_workflow.js`, pinned at `fa86fd1e9bec56dadf27d6575a0c8f8ab7875c57`. `e2e_workflow` targets SGLang and vLLM. It is the wrong entry point for FastWAM.

`mad-optimize` is the MAD workflow that calls that entry point. Plan mode (`docs/blog/transcripts/11-mad-optimize-plan.txt`, workflow `wf_d09edae8-215`) resolved `pyt_fam/pyt_fam_fastwam_geak` and printed the handoff. It did not clone, did not start a GPU job, and did not spend the budget.

- Container path: `/opt/faster-action-models/src/fastwam-geak`
- Host path: `$HOME/.mad-optimize/pyt_fam_pyt_fam_fastwam_geak/src`
- Clone, only if `$HOME/GEAK` is missing: `git clone https://github.com/AMD-AGI/GEAK.git "$HOME/GEAK" && git -C "$HOME/GEAK" checkout fa86fd1e9bec56dadf27d6575a0c8f8ab7875c57`
- `scriptPath`: `$HOME/GEAK/kernel_workflow/kernel_workflow.js`
- `workflow_dir`: `$HOME/GEAK/kernel_workflow`
- `budget`: 1, `gpu_ids`: `"0"`, `apply_to_original`: `"true"`

The kernel text GEAK would edit is the tree the image already has. On execute, Source copies that tree to the host path, baseline runs leave it unmounted, and patched runs mount it back over the same container path. This rerun did not pass `--execute`. The latencies above are the tree `faster-action-models` already ships.

**Prompt**

```text
Run the mad-optimize workflow in plan mode. Use the Workflow tool. scriptPath is .claude/workflows/mad-optimize.js. args must be a JSON object, not a string:

{"tag":"pyt_fam_fastwam_geak","smokeTag":"pyt_fam_fastwam_geak2_smoke","budget":1,"repeats":2,"execute":false,"additionalContext":{"docker_mounts":{"/myworkspace/fam_cache":"/var/tmp/fam_cache"},"docker_env_vars":{"FAM_CACHE":"/myworkspace/fam_cache"}}}

Do not clone GEAK. Do not run madengine run. Do not invoke e2e_workflow. Do not spend the budget. Quote status, the model name, the clone command, containerPath, hostPath, scriptPath, workflow_dir, budget, gpu_ids, and apply_to_original.
```

To run it for real, use the same object with `"execute": true`. `claude --version` must already be at least 2.1.177. If you launch that with `claude -p`, set `CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0` first. That run is not part of the numbers in this post.

## 10. The machine, once

| | |
|---|---|
| Host | `smci350-rck-g03-b06-40`, 8× AMD Instinct MI350X, gfx950; each cited run requests one GPU |
| Image | `rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.11.0`, already local; the rerun did not pull it |
| Tree | `AMD-AGI/faster-action-models` @ `a5fa731` |
| Torch in the container | `2.11.0+rocm7.14.0` |
| Cited latency | one session, four arms, `perf_fam_latency.csv`, milliseconds, lower is better |
| Sweep | the same four arms as separate cells, `perf_rerun_*.csv` |
| Accuracy | two smokes, 40 episodes each, one trial per task, `success_rate`, higher is better |
| Rollouts | four tasks, eager and GEAK at 2 steps, agent camera, `fam-rollout-2`, 8/8 finished |
| Dispatch trace | [traces/geak-dispatch.json.gz](traces/geak-dispatch.json.gz), open in Perfetto |
| GEAK | `mad-optimize` plan only. Budget 1 was printed and not spent |

Weights stayed on the host at `/var/tmp/fam_cache`, mounted at `/myworkspace/fam_cache`.

Related reading: [Optimizing World Action Models on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/faster-action-models/README.html).
