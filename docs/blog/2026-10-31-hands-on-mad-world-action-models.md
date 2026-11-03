# Hands-on MAD skills, agents, and workflows: World Action Models

This post walks MAD's skills, agents, and workflows on one workload: FastWAM on LIBERO, in a container on one AMD Instinct MI350X (gfx950). The walkthrough was recorded on 2026-11-02, and the same models were rerun on this host on 2026-11-03. Every number is from one of those container runs. Where the two days disagree, the section says which day. Latency runs call the released FastWAM and CAE checkpoints. The accuracy smokes run the FastWAM checkpoint in LIBERO's simulator: simulated episodes, one GPU each. The host has eight GPUs.

Each section ends with the prompt to reproduce it. Paste it into Claude Code from the MAD repo root.

A prompt that only explains something stays a sentence. Section 1 is that prompt. A skill that builds, runs, or writes files is explicit-only (`disable-model-invocation: true`), so its prompt starts with `/mad-add-model`, `/mad-benchmark`, `/mad-profile`, or `/mad-tune`. A plain sentence does not load those four, and the pre-flight, the subagent, and `mad-run.sh` never start. `mad-validate-model` and `mad-report` do not set that flag. This post still starts them with `/mad-validate-model` and `/mad-report`.

The three workflows are not slash commands. There is no `/mad-benchmark-sweep`. They live with the other Claude Code workflows and start through the Workflow tool. `args` is a JSON object, not a string. They plan unless `execute` is `true`. GPU runs go through `mad-run.sh`. On 2026-11-03, `docker build --pull` of the base image was denied by Docker Hub, so the rerun built a manifest with `--use-image` against the image already present and passed that manifest as `-m`. If a pull is denied, do the same: build from the local image, then run that manifest.

## Four arms, and why that count

The sweep is `mad-benchmark-sweep`. Each tag is one cell. Cells run one at a time. There is no precision axis: a different configuration is a different model entry.

Four FastWAM-on-LIBERO latency arms:

- **eager** — PyTorch eager. The baseline, and the arm the operator profile uses.
- **torch.compile** — the compiler step, so the GEAK step is compared with compile on the same GPU.
- **GEAK** — the kernel tree shipped with FastWAM, at 10 denoising steps.
- **GEAK at 2 denoising steps** — fewer steps. The smoke checks that the success rate still holds.

CAE is a second checkpoint, so it is its own benchmark, not a fifth sweep cell. DiT4DiT and RoboCasa are not registered. They appear once, as a plan-mode tag that does not resolve. Two smoke runs, one trial per task, are the accuracy check: 40 simulated episodes each.

The cited latency table is one madengine model that times all four arms inside one session. The sweep still runs the four arms as separate cells, because that is the workflow this post is teaching. Both tables are below. The one-session table is the citation.

## 1. Skill, agent, workflow

A **skill** is the procedure: `mad-add-model`, `mad-benchmark`, `mad-profile`, `mad-tune`, `mad-report`. The first four build, run, or write files, so they are explicit-only. `mad-report` is read-only.

An **agent** carries the procedure out. `mad-benchmark` and `mad-profile` run in `mad-benchmark-runner`. `mad-tune` runs in `mad-tuner`. `mad-report` runs in `mad-perf-analyst`. The analyst is read-only. It never starts a GPU job, and it never compares a millisecond with a success rate.

A **workflow** fans the procedure out. `mad-benchmark-sweep`, `mad-tune-search`, and `mad-optimize` run in Claude Code. They plan by default. `--execute` is what touches GPUs. `mad-optimize` is the one that may come back with a different kernel tree. `mad-tune-search` still stops if a tracked file changes.

![From mad-add-model, the sweep splits into the one-session latency path and the four cells. Both paths reach mad-report. The dashed arrow is the one-session result reaching the report directly. mad-tune-search also continues to the mad-optimize plan.](figures/mad-flow.svg)

FastWAM is a world action model. A video backbone predicts how the scene may evolve, and an action expert denoises an action chunk conditioned on that prediction. The two share attention. At inference the future-frame prediction is not displayed. The ladder starts at eager because that is the unmodified PyTorch path, and the operator profile is on that same arm.

![A VLA maps observations to an action. A WAM predicts the scene, then selects the action. FastWAM is the lower path.](figures/vla-vs-wam.svg)

![FastWAM's video DiT and action DiT share attention. The action chunk is 32 by 7. This host did not remeasure that shape. The video prediction is not displayed.](figures/dual-dit.svg)

**Prompt**

```text
In the MAD repo, explain how a skill, an agent, and a workflow differ, using mad-benchmark, mad-benchmark-runner, and mad-benchmark-sweep. Then name mad-tune-search and mad-optimize, and say which one is allowed to return a different kernel tree. Do not run anything.
```

## 2. One directory, one image, no nested docker

FastWAM's own launchers start a container. madengine has already started one, so the run script calls the in-image Python entry points and stops there.

One image serves the whole directory. It clones `AMD-AGI/faster-action-models` at `a5fa73101988870e51f04fe1e7be5da56dda3841`. `url` on every registry entry is empty, because a second clone collides with files the registry directory already contains. Nine entries share that image. Discover listed eight of them when it was captured. `pyt_fam_fastwam_rollout` is the ninth, added for the frame capture. Latency is `ms` (lower is better). Accuracy is `success_rate` (higher is better).

```text
$ python3 tools/validation/model_config_validator.py --model pyt_fam_fastwam_latency
✅ All validation checks passed!
```

**Prompt**

```text
/mad-validate-model pyt_fam_fastwam_latency

Quote the pass or fail line. Do not build an image and do not start a GPU job.
```

## 3. Adding one entry to the directory that already exists

`/mad-add-model` is how a model gets into MAD. It is explicit-only, same as the run skills. It asks for the name, the repository URL, training or inference, the GPU count, the precision, and the owner. It checks that the name is unused, then it writes. `madengine discover --tags <name>` is that check. A name that already lists a model must not get a second entry.

FastWAM is not a new directory per arm. The Dockerfile and the run script already exist, so a new arm does not repeat those steps. It is one object in the directory registry. Paths in that file are relative to the registry directory. `dockerfile` points at the shared image prefix, and madengine appends `.ubuntu.amd.Dockerfile`. `url` stays empty. The run script is what prints `performance: <value> <unit>`.

`pyt_fam_fastwam_rollout` is the entry added for the frame capture. Its unit is `success_rate`. The prompt below does not add a second model. The name is already registered, so `/mad-add-model` stops at the duplicate check and does not write.

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

Pre-flight passed, and discover listed eight models as `pyt_fam/<name>`. The registry now also has `pyt_fam_fastwam_rollout`.

The sweep in plan mode prints one command per cell and runs nothing. Headline from that session: 4 cells planned, 0 ran. Each command carries the weights mount. In `docker_mounts` the key is the container path and the value is the host directory that holds the weights. The container path is `/myworkspace/fam_cache`, and `FAM_CACHE` is that same container path. A note that swaps the two sides is wrong. The latency run mounts the host weights directory onto that container path.

```text
madengine run --tags 'pyt_fam/pyt_fam_fastwam_eager' --live-output --additional-context '{"docker_mounts":{"/myworkspace/fam_cache":"<host-weights-dir>"},"docker_env_vars":{"FAM_CACHE":"/myworkspace/fam_cache"}}'
madengine run --tags 'pyt_fam/pyt_fam_fastwam_compile' --live-output --additional-context '{"docker_mounts":{"/myworkspace/fam_cache":"<host-weights-dir>"},"docker_env_vars":{"FAM_CACHE":"/myworkspace/fam_cache"}}'
madengine run --tags 'pyt_fam/pyt_fam_fastwam_geak' --live-output --additional-context '{"docker_mounts":{"/myworkspace/fam_cache":"<host-weights-dir>"},"docker_env_vars":{"FAM_CACHE":"/myworkspace/fam_cache"}}'
madengine run --tags 'pyt_fam/pyt_fam_fastwam_geak2' --live-output --additional-context '{"docker_mounts":{"/myworkspace/fam_cache":"<host-weights-dir>"},"docker_env_vars":{"FAM_CACHE":"/myworkspace/fam_cache"}}'
```

The same workflow with `pyt_fam_dit4dit` added stays in plan mode. That tag is not in the registry. The cell status is `error`: `No models found corresponding to the given tag: pyt_fam_dit4dit`. The other four stay planned. That prompt stays `execute: false` on purpose. `execute: true` would start the four registered cells.

**Prompt, the plan session**

```text
Run the mad-benchmark-sweep workflow. Use the Workflow tool. scriptPath is .claude/workflows/mad-benchmark-sweep.js. args must be a JSON object, not a string:

{"tags":["pyt_fam_fastwam_eager","pyt_fam_fastwam_compile","pyt_fam_fastwam_geak","pyt_fam_fastwam_geak2"],"execute":false,"additionalContext":{"docker_mounts":{"/myworkspace/fam_cache":"<host-weights-dir>"},"docker_env_vars":{"FAM_CACHE":"/myworkspace/fam_cache"}}}

Do not run madengine run. Quote the four cell commands and the headline. Expect 4 cells planned and 0 ran. Replace <host-weights-dir> with the directory that holds the weights.
```

**Prompt, run the four cells**

```text
Run the mad-benchmark-sweep workflow. Use the Workflow tool. scriptPath is .claude/workflows/mad-benchmark-sweep.js. args must be a JSON object, not a string:

{"tags":["pyt_fam_fastwam_eager","pyt_fam_fastwam_compile","pyt_fam_fastwam_geak","pyt_fam_fastwam_geak2"],"execute":true,"additionalContext":{"docker_mounts":{"/myworkspace/fam_cache":"<host-weights-dir>"},"docker_env_vars":{"FAM_CACHE":"/myworkspace/fam_cache"}}}

Replace <host-weights-dir> with the directory that holds the weights. This runs four cells, one at a time, on a GPU. Quote each performance line. Lower milliseconds are better. The separate-cell table in the next section was produced the same way after Docker Hub denied the pull: madengine build --use-image, then madengine run -m. If the pull is denied, stop and do that. Do not take the milliseconds from the plan-mode session.
```

**Prompt for the unregistered tag**

```text
Run mad-benchmark-sweep. scriptPath is .claude/workflows/mad-benchmark-sweep.js. args is a JSON object:

{"tags":["pyt_fam_fastwam_eager","pyt_fam_fastwam_compile","pyt_fam_fastwam_geak","pyt_fam_fastwam_geak2","pyt_fam_dit4dit"],"execute":false}

Do not run madengine run. Quote the dit4dit cell status and confirm the other four stay planned. Leave execute false. execute true would start the four registered cells.
```

## 5. The cited table, and the sweep's own table

The cited run is one container, four arms, three interleaved rounds (run `fam-latency-5`). Device: AMD Instinct MI350X, torch `2.11.0+rocm7.14.0`.

| arm | steps | rounds (ms) | median ms | spread |
|---|---:|---|---:|---:|
| eager | 10 | 216.579, 223.929, 229.188 | 223.9293 | 5.63% |
| compile | 10 | 87.373, 88.973, 89.335 | 88.9732 | 2.21% |
| geak | 10 | 24.845, 24.899, 24.946 | 24.8986 | 0.41% |
| geak@2 | 2 | 9.921, 9.931, 9.959 | 9.9310 | 0.38% |

The median column is the model's multiple-results output: four performance lines, with no status or host columns. The round column is the three lines the log printed. Spread is `(max − min) / median`. Against eager, compile is 2.52×, the shipped GEAK tree is 8.99×, and GEAK at 2 steps is 22.55×. GEAK is 3.57× past compile.

| arm | median ms | times the eager median |
|---|---:|---:|
| eager | 223.9293 | 1.00 |
| compile | 88.9732 | 2.52 |
| geak | 24.8986 | 8.99 |
| geak@2 | 9.9310 | 22.55 |

![One session. The dot is the median of three rounds and the bar is min to max.](figures/latency-ladder.svg)

That comparison was computed from the run output. It is not a `mad-report` session. Lower is better. It does not mention the smoke rate.

The four cells ran again on 2026-11-03, one at a time through `mad-run.sh` on local-image manifests. That is the same one-at-a-time rule as `mad-benchmark-sweep --execute`, and it was not a Claude `--execute` of the workflow.

| cell | ms |
|---|---:|
| eager | 220.8598 |
| compile | 90.6426 |
| geak | 24.5373 |
| geak@2 | 9.9039 |

Against this session's eager number the ratios are 2.44×, 9.00×, and 22.30×. The order matches the cited table. An earlier separate-cell session was 221.4562, 90.6402, 24.6736, and 9.8859 ms.

The same day, one container timed all four arms once: eager 220.0150 ms, compile 90.2767 ms, GEAK 25.2517 ms, and GEAK at 2 steps 9.8227 ms. That session has one sample per arm, so the three-round table above stays the citation. The speedup in that table is the tree `faster-action-models` already ships. MAD did not produce it.

**Prompt**

```text
/mad-benchmark pyt_fam_fastwam_latency

Mount the weights: additional-context {"docker_mounts":{"/myworkspace/fam_cache":"<host-weights-dir>"},"docker_env_vars":{"FAM_CACHE":"/myworkspace/fam_cache"}}. Replace <host-weights-dir> with the directory that holds the weights. The skill starts the run through mad-run.sh. The madengine record is one row for the session. The model's multiple-results output has one row per arm, and that is the table to read. Report the four medians in ms. Lower is better. Do not compare them with a success rate.

If docker pull of the base image is denied, stop and reuse the local image with madengine build --tags pyt_fam_fastwam_latency --use-image <local tag> and madengine run -m <manifest>.
```

**Prompt, the comparison**

```text
/mad-report

Use the multiple-results output of the pyt_fam_fastwam_latency run you just finished. Lower milliseconds are better. Report the four medians and the ratio of each to the eager median. Do not compare them with a success rate.
```

## 6. Operator profile

Operator depth answers which PyTorch categories dominate kernel time. One depth only, on `pyt_fam_fastwam_eager`. Stacks off, 2000 ms window, warmup 25 s.

`performance: 276.7330 ms` on the 2026-11-03 capture. That number was measured under the profiler. It is evidence, not a score. TraceLens status `SUCCESS`. Kernel time in the 2000 ms window:

![Eager kernel-time mix from the 2000 ms operator window on 2026-11-03.](figures/operator-mix.svg)

| category | share |
|---|---:|
| elementwise | 50.78% |
| GEMM | 24.58% |
| reduce | 8.23% |
| SDPA_fwd | 7.75% |
| NORM_fwd | 3.63% |
| other | 3.04% |
| CONV_fwd | 1.98% |

The GPU timeline in that window is 29.62% computation, 0.04% memcpy, and 70.33% idle. Modules were enabled. The category summary has no `nn.Module` share, so it does not attribute time to the action-denoising module. An earlier window on 2026-11-02 was 49.91% elementwise, 32.57% computation, and 67.40% idle, with `performance: 248.8302 ms` under the profiler. The table above is the 2026-11-03 window.

A warm rocprofv3 kernel trace on the same eager arm recorded 1,720,495 dispatches. No kernel holds the GPU. The largest share is a hipBLASLt stream-K GEMM, the `Cijk_..._MT16x32x512_..._SK5` kernel, at 8.6% (97,500 dispatches, 7.46 µs average). A reduction is 8.2%, an elementwise multiply is 8.2%, and `attn_fwd` is 8.2% (42,900 dispatches, 16.07 µs). `performance: 500.2499 ms` on that traced run is evidence, not a score.

The kernel-depth counter run on this arm did not finish. The GPU stayed at 0% and the host sat in a wait inside `FastWAM.infer_action`, first on that stream-K GEMM and then on `attn_fwd`. Both runs were stopped. There is no counter card for the eager arm. PC sampling did finish. The same stream-K GEMM had 206,286 samples. Its issued share is 0.328. The stall reasons are other-wait 32.8%, arbiter stall 22.2%, and waitcnt 20.7%. `performance: 808.5095 ms` under sampling is evidence, not a score. PC sampling is stochastic.

`/mad-profile` is the skill for a single depth. `mad-tune` is the measure-change-measure loop. It is not how these captures are started. A search that profiles and then measures candidates is `mad-tune-search`.

**Prompt, operator depth. This is the published capture.**

```text
/mad-profile pyt_fam_fastwam_eager operator

One run, through mad-run.sh. Mount the weights the same way as the latency run. Set TORCH_PROFILE_ITERATIONS=0, TORCH_PROFILE_DURATION_MS=2000, TORCH_PROFILE_WARMUP_S=25, TORCH_PROFILE_WITH_STACKS=0, and TRACELENS_MODE=pytorch. Tools are torch_profiler_dynolog and tracelens. Do not add rocm_trace_lite or any rocprofv3 tool. The performance line from this run is evidence, not a score. Report kernel time by category from the TraceLens summary. Say whether an nn.Module share is present.
```

**Prompt, dispatch depth on the eager arm.**

```text
/mad-profile pyt_fam_fastwam_eager dispatch

One run, through mad-run.sh. Mount the weights the same way as the latency run. Warm the code-object cache first, then take a rocprofv3 kernel trace and rank it with kernel-tools.py rank. Do not use rocm_trace_lite to name the top kernel. The performance line is evidence, not a score. Report the kernel-time share from kernel_rank.json. On this arm the largest share is under 10%.
```

**Prompt, kernel depth on the eager arm.**

```text
/mad-profile pyt_fam_fastwam_eager kernel

One run, through mad-run.sh. Mount the weights the same way as the latency run. Collect rocprofv3 counters for the top kernel from the dispatch rank. If the GPU stays at 0% and the host is waiting inside infer_action, stop that run. Do not treat a killed counter run as a score, and do not invent a kernel card. Fall back to PC sampling and report the stall reasons for the kernel that was sampled. The performance line is evidence, not a score.
```

## 7. Dispatch trace, then the kernel

The dispatch depth that names the kernel is a rocprofv3 kernel trace on a warmed cache, then `kernel-tools.py rank`. `rocm_trace_lite` does not see kernels replayed from HIP graphs, so it is the wrong tool for this ranking.

On 2026-11-03 the dispatch run recorded 340,154 dispatches. `_fw_skmm_kernel` is 14.6% of GPU kernel time (58,680 dispatches, 5.01 µs average), then `_fw_skpost_kernel` at 13.7%. The warm-up runs that finished printed `performance: 24.6547 ms` and `performance: 25.3017 ms`. Those are clean scores. The traced run printed `performance: 30.7142 ms`, and that line is evidence, not a score. A second warm trace, taken while `mad-optimize` was profiling, recorded 340,754 dispatches and the same kernel at 14.5% (58,680 dispatches, 4.86 µs). Its clean warm score was 24.9435 ms, and its traced run was 30.5844 ms. The chart is the dispatch run.

| kernel | share | dispatches | average |
|---|---:|---:|---:|
| `_fw_skmm_kernel` | 14.6% | 58,680 | 5.01 µs |
| `_fw_skpost_kernel` | 13.7% | 57,376 | 4.79 µs |
| `triton_tem_fused_add_addmm_mul_native_layer_norm_split_t_view_1` | 7.4% | 19,560 | 7.62 µs |
| `_fw_qrms_attn_kernel` | 7.2% | 19,560 | 7.42 µs |
| `_fw_attn_kernel` | 7.1% | 23,580 | 6.04 µs |
| `_fw_pfsk_kernel` | 5.8% | 8,040 | 14.39 µs |
| `triton_tem_fused_addmm_gelu_skpost_t_view_3` | 5.6% | 19,560 | 5.78 µs |
| fill (`FillFunctor<int>`) | 5.5% | 33,578 | 3.26 µs |

![Top kernels on the shipped GEAK tree after a warm cache. _fw_skmm_kernel is 14.6% of GPU kernel time. These eight bars are the head of the list and do not add up to 100%.](figures/kernel-share.svg)

The counter card for `_fw_skmm_kernel` is bandwidth-bound: 4.32 µs, 28.3% of the MI350X peak bandwidth (8.0 TB/s) and 2.2% of peak MFMA (2306.9 TFLOP/s). The card profiles the same 58,680 dispatches. A later counter run was killed (exit 137) and did not replace this card. PC sampling of the same kernel collected 82,369 samples. Waitcnt is 67.8% of them, other-wait is 21.1%, and the issued share is 0.211. `performance: 96.7716 ms` under sampling is evidence. Assembly found two `_fw_skmm_kernel` code objects. Both have 92 VGPRs. One has 240 instructions and the other 289, so the assembly summary stays ambiguous.

A cold `rocm_trace_lite` capture on the same model put `triton_mm` at 66.0% of kernel time. `performance: 26.8858 ms` on that run is profiled, not a clean score. GPU utilization over that capture is 5.39% across 75.7 s of wall, because the trace includes startup. That share is not the steady state. On a cold `torch.compile` run, `triton_mm` is Inductor trying autotune candidates. Those candidates do not run once the cache is warm. The chart of that capture is kept as `figures/kernel-share-cold.svg`.

The cold capture is a Chrome trace: about 3 MB compressed, 285,765 GPU slices, one MI350X. The two figures are that capture open in [Perfetto](https://ui.perfetto.dev). The same kind of file opens there with **Open trace file**. The trace itself is not attached. Nearly all of the kernels land on one hardware queue. The other three queues appear at the end of the capture. Perfetto shows an import warning on this Chrome trace. The queues still draw. The wide slices in the zoom are the autotune `triton_mm` candidates from that cold capture.

![Perfetto timeline of the cold GEAK dispatch capture, about 75 seconds. Queue 3 holds the kernels.](figures/perfetto-dispatch-overview.png)

![Perfetto, 50 microseconds at 27.983 seconds of the cold capture. triton_mm autotune candidates and elementwise kernels on queue 3.](figures/perfetto-dispatch-zoom.png)

**Prompt**

```text
/mad-profile pyt_fam_fastwam_geak dispatch

One run, through mad-run.sh. Mount the weights the same way as the latency run. Warm the code-object cache first, then take a rocprofv3 kernel trace and rank it with kernel-tools.py rank. Do not use rocm_trace_lite to name the top kernel: it misses kernels replayed from HIP graphs. The performance line from the traced run is evidence, not a score. Report the kernel-time share from kernel_rank.json. The kernel-share chart is that warm ranking. The Perfetto figures are the earlier cold capture.
```

**Prompt, kernel depth on the GEAK arm**

```text
/mad-profile pyt_fam_fastwam_geak kernel

One run, through mad-run.sh, after the warm dispatch rank above. Collect rocprofv3 counters for _fw_skmm_kernel only, then build the kernel card with kernel-tools.py card. Report the bound, the percent of peak bandwidth, and the percent of peak MFMA. If a later counter run is killed, keep the card that already names this kernel. Then run PC sampling and kernel-tools.py asm. If asm reports two code objects, leave the summary ambiguous. The performance line from a profiled run is evidence, not a score.
```

## 8. One tuner lever

`mad-tune-search` in plan mode resolved `pyt_fam_fastwam_geak`, target latency, depth dispatch, two repeats, one candidate, `execute: false`. The candidate it named is `PYTORCH_TUNABLEOP_ENABLED=1`, with `PYTORCH_TUNABLEOP_TUNING=1` on the first run and `TUNING=0` after that so later runs reuse the results file. That is an environment lever. It does not edit a tracked file.

TunableOp chooses among rocBLAS GEMM implementations. It does not rewrite the steady-state kernel, `_fw_skmm_kernel`. The plan named TunableOp because an earlier cold trace had ranked `triton_mm` first. It was measured, and then not kept. The samples below are that measurement. They are not a workflow `--execute`.

Four earlier repeats are not scores. They reused the dispatch manifest, and that manifest still had `rocm_trace_lite` in it. A later `madengine run -m` keeps tools a previous run wrote into the same manifest. Those four ran under the profiler. They are discarded.

The repeats below used a new manifest from `madengine build --use-image`, with no tools key:

| run | environment | result |
|---|---|---|
| baseline | weights mount only | 24.9009 ms |
| baseline | weights mount only | 24.9296 ms |
| plan's first step | `ENABLED=1`, `TUNING=1` | 25.2935 ms, and no results file |
| `ENABLED=1` only | tuning left unset | no performance line |
| `ENABLED=1` only | tuning left unset | 24.9987 ms |

The two baseline samples have median 24.9153 ms and spread 0.12%, `(max − min) / median`. The `TUNING=1` run was pointed at a results file in the weights cache. The run finished, and that file was not created. The log has no TunableOp activity. There is no tuned file for a later `TUNING=0` run to reuse, so those score runs were not taken. One search run is not a gate sample.

`ENABLED=1` with tuning unset was tried twice. The first raised `InductorError: CantSplit: 640 not divisible by 392` and wrote no performance line. The second finished at 24.9987 ms, slower than both baseline samples. The gate needs two candidate samples. This side does not have them.

The lever is not kept. There is no chart for it. This measurement is from 2026-11-02 and was not repeated on 2026-11-03. The ranking in section 7 does not change it: TunableOp was not a win on the samples above, and the speedup in the cited table is still the tree the image already contained.

**Prompt, the plan session**

```text
Run mad-tune-search. Use the Workflow tool. scriptPath is .claude/workflows/mad-tune-search.js. args must be a JSON object:

{"tag":"pyt_fam_fastwam_geak","target":"latency","depth":"dispatch","repeats":2,"candidates":1,"execute":false,"additionalContext":{"docker_mounts":{"/myworkspace/fam_cache":"<host-weights-dir>"},"docker_env_vars":{"FAM_CACHE":"/myworkspace/fam_cache"}}}

Replace <host-weights-dir> with the directory that holds the weights. Do not run madengine run. Quote the proposed lever and confirm execute is false.
```

**Prompt, the measurement**

```text
/mad-tune pyt_fam_fastwam_geak latency dispatch

This measures the lever the plan named. It is not mad-tune-search --execute. The workflow would choose its own lever.

On pyt_fam_fastwam_geak, build a fresh manifest with madengine build --use-image. Do not reuse a manifest that still has rocm_trace_lite in it. Mount the weights the same way as the latency run. Run two clean baseline repeats with the weights mount only.

The lever is an environment change, not a file edit. Point PYTORCH_TUNABLEOP_FILENAME at the weights cache, not the repo. Set PYTORCH_TUNABLEOP_ENABLED=1 and PYTORCH_TUNABLEOP_TUNING=1 for one search run. Run the scoring repeats with TUNING=0 only if that search wrote a results file. Do not edit a tracked file.

Keep the lever only when both sides have two clean samples and the improvement beats max(baseline spread, candidate spread, 1%). Lower milliseconds win.
```

## 9. Smokes and rollouts

The 2-step arm needs a success rate beside it, in a different table. Two smokes, one trial per task, 40 tasks.

The 2026-11-03 smokes, one trial per task, 40 tasks:

| model | spatial | object | goal | long | rate |
|---|---:|---:|---:|---:|---:|
| eager | 10/10 | 10/10 | 10/10 | 10/10 | 1.0000 success_rate |
| GEAK at 2 steps | 10/10 | 10/10 | 9/10 | 10/10 | 0.9750 success_rate |

The missed episode is `libero_goal` task 3. These are one-trial smokes on MI350X. The 2026-11-02 smokes were 10/10 on every suite for both arms, `1.0000` and `1.0000`. The two rates were compared with each other from the run output. That comparison is not a `mad-report` session, and it does not compare them with milliseconds.

The frames below are a separate capture, `pyt_fam_fastwam_rollout` (run `fam-rollout-2`). Four instructions, one trial each, on the agent camera of this host's simulator. The top row is eager. The bottom row is GEAK at 2 steps. Five frames are taken from start to finish. All eight episodes finished. An earlier start of this model failed before any frame was saved. The successful run is these eight episodes. A rerun on 2026-11-03 also finished 8/8 and printed `performance: 1.0000 success_rate`. The pictures below are the earlier capture. This rate is not the 40-task smoke.

![Spatial: pick up the black bowl next to the ramekin and place it on the plate. Eager on top, GEAK at 2 steps below. Both finished.](figures/rollout-spatial.png)

![Object: pick up the alphabet soup and place it in the basket. Eager on top, GEAK at 2 steps below. Both finished.](figures/rollout-object.png)

![Goal: open the middle drawer of the cabinet. Eager on top, GEAK at 2 steps below. Both finished.](figures/rollout-goal.png)

![Long: put both the cream cheese box and the butter in the basket. Eager on top, GEAK at 2 steps below. Both finished.](figures/rollout-long.png)

**Prompt for the smokes**

```text
/mad-benchmark pyt_fam_fastwam_eager_smoke,pyt_fam_fastwam_geak2_smoke

Mount the weights the same way as the latency run. Report each performance line. The metric is success_rate. Higher is better. Do not place these rates in a table with milliseconds. Each suite in the table is one trial of 10 tasks.
```

**Prompt, the comparison**

```text
/mad-report

Compare the eager smoke with the GEAK-at-2-steps smoke from the runs you just finished. Higher success_rate is better. Compare the two rates with each other. Do not compare them with milliseconds.
```

**Prompt for the rollouts**

```text
/mad-benchmark pyt_fam_fastwam_rollout

Mount the weights the same way as the latency run. It is one trial of four tasks for eager and for GEAK at 2 steps. Report the eight episode lines. Do not treat that rate as the 40-task smoke.
```

## 10. CAE is a second model

CAE-FastWAM is a different checkpoint (`amd/CAE-FastWAM`), so it is `pyt_fam_cae_fastwam_latency`, run at 10 denoising steps. It is not a fifth cell of the sweep. The run uses the released weights and does not retrain CAE.

```text
performance: 13.5933 ms
```

A rerun on 2026-11-03 printed `performance: 13.6415 ms`. Both are one sample at 10 denoising steps.

**Prompt**

```text
/mad-benchmark pyt_fam_cae_fastwam_latency

Mount the weights the same way as the latency run. Report the performance line. It is milliseconds, lower is better. Do not describe this run as a fifth cell of the four-arm sweep, and do not retrain CAE.
```

## 11. mad-optimize, budget 1

GEAK's entry point for a kernel tree is `kernel_workflow`, pinned at `fa86fd1e9bec56dadf27d6575a0c8f8ab7875c57`. `e2e_workflow` targets SGLang and vLLM. It is the wrong entry point for FastWAM.

`mad-optimize` calls that entry point. Plan mode prints the handoff and spends nothing. Execute with budget 1 ran on 2026-11-03. It edited `_fw_skmm_kernel`, the kernel section 7 ranked first. The trace inside that run put the kernel at 14.5% of GPU kernel time. The counter pass in that same run was killed (exit 137) and did not replace the card in section 7.

GEAK's own kernel measurement of the patch is 1.0378× on the kernel microbenchmark: 5.150 µs to 4.959 µs, geomean of K=3072 at 1.0476× and K=4096 at 1.0282×. `workload_aligned` was false. That 1.0378× is not the model score. The report says further kernel gain needs removing the second dispatch, not another round of the same parameters. Budget 1 did not try that.

The patch changes two defaults in `mot.py`: `FASTWAM_SKRED` from `256,4` to `128,1`, and the K=4096 tile from 128 to 256. The end-to-end gate did not keep it. Baseline samples were 24.7965 ms and 24.8726 ms (median 24.83 ms, spread 0.31%). Patched samples were 35.8780 ms and 24.7936 ms (median 30.34 ms, spread 36.5%). Lower is better. The patched median is 22% slower. A keep requires the improvement to beat the larger spread and a 1% floor. The second patched sample is 0.16% under the baseline median, inside that floor, so dropping the slow sample still does not keep the patch. The smoke tag did not run inside this workflow. The smoke in section 9 is a separate run.

GEAK wrote `final_patch.diff` beside `geak_eval/` rather than inside it. `geak-run.sh status` copies that file into `geak_eval/` and then reports the run as succeeded. The copied diff is the two-line change above. The gate still did not keep it.

- The kernel tree in the container is the FastWAM GEAK tree the image already has.
- The host copy is the mad-optimize working tree for this model.
- GEAK is cloned only if it is not already present, then checked out at the pin above.
- The script is `kernel_workflow`, and its directory is that workflow's directory.
- `budget` is 1, `gpu_ids` is `"0"`, and `apply_to_original` is `"true"`.

The kernel text GEAK edits is the tree the image already has. On execute, the Source phase copies that tree to the host working tree, baseline runs leave it unmounted, and patched runs mount it back over the same container path. The latencies in section 5 are the tree `faster-action-models` already ships. The budget-1 result above is a separate measurement, and it did not replace those latencies.

**Prompt**

```text
Run the mad-optimize workflow in plan mode. Use the Workflow tool. scriptPath is .claude/workflows/mad-optimize.js. args must be a JSON object, not a string:

{"tag":"pyt_fam_fastwam_geak","smokeTag":"pyt_fam_fastwam_geak2_smoke","budget":1,"repeats":2,"execute":false,"additionalContext":{"docker_mounts":{"/myworkspace/fam_cache":"<host-weights-dir>"},"docker_env_vars":{"FAM_CACHE":"/myworkspace/fam_cache"}}}

Replace <host-weights-dir> with the directory that holds the weights. Do not clone GEAK. Do not run madengine run. Do not invoke e2e_workflow. Do not spend the budget. Quote status, the model name, the clone command, containerPath, hostPath, scriptPath, workflow_dir, budget, gpu_ids, and apply_to_original.
```

To run it for real, use the same object with `"execute": true`. `claude --version` must already be at least 2.1.177. If you launch that with `claude -p`, set `CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0` first. The execute result in this section is that run. It is not part of the section 5 latencies.

## 12. The machine, once

| | |
|---|---|
| Host | 8× AMD Instinct MI350X, gfx950; each cited run requests one GPU |
| Image | `rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.11.0`, already local; the rerun did not pull it |
| Tree | `AMD-AGI/faster-action-models` @ `a5fa731` |
| Torch in the container | `2.11.0+rocm7.14.0` |
| Cited latency | one session, four arms, milliseconds, lower is better |
| Sweep | 2026-11-03 cells: 220.8598, 90.6426, 24.5373, 9.9039 ms |
| Accuracy | 2026-11-03 smokes, 40 episodes each: eager 1.0000, GEAK at 2 steps 0.9750 |
| Rollouts | four tasks, eager and GEAK at 2 steps, agent camera, 8/8 finished on both days |
| Dispatch trace | warm rocprofv3 kernel trace on 2026-11-03; `_fw_skmm_kernel` is 14.6% of GPU kernel time. The Perfetto figures are the cold capture |
| Kernel card | `_fw_skmm_kernel` is bandwidth-bound, 28.3% of peak bandwidth and 2.2% of peak MFMA. Assembly is ambiguous |
| Eager kernel | no kernel above 8.6%. Counter runs did not finish. PC sampling did |
| GEAK | `mad-optimize` budget 1 ran. Kernel microbenchmark 1.0378×. The end-to-end gate did not keep the patch |

Weights stayed in a host directory and were mounted at `/myworkspace/fam_cache`.

Related reading: [Optimizing World Action Models on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/faster-action-models/README.html).
