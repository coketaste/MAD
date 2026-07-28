# Kimi-K3 + ATOM adoption in MAD — Design

Date: 2026-07-28
Reference: [Day 0 Kimi-K3 Inference Deployment with ATOM on AMD Instinct MI355X GPUs](https://www.amd.com/en/developer/resources/technical-articles/2026/kimi-k3-on-amd-instinct-gpus.html)

## Goal

Add Kimi-K3 as a MAD model that deploys through AMD's ATOM inference engine on
8× MI355X (gfx950) GPUs at TP8, reproducing the article's validated Day-0
configuration. The run performs a GSM8K 5-shot correctness check plus a basic
token-throughput measurement.

## Decisions

- **Server path**: ATOM's native OpenAI-compatible server
  (`python -m atom.entrypoints.openai_server`), not the `vllm serve` plugin path.
  Lives in a new `scripts/atom/` tree so ATOM's divergent CLI stays isolated from
  the existing vLLM path.
- **Metrics**: GSM8K 5-shot accuracy (correctness) AND basic token throughput.
- **Hardware gating**: hardcoded TP8, `n_gpus: 8`, tagged gfx950/MI355X only.
  The ~1.56 TB checkpoint is validated only in this single configuration.

## New files

```
docker/pyt_atom.ubuntu.amd.Dockerfile   # FROM the K3 atom-dev image
scripts/atom/run.sh                      # entry: parse args, run_atom.py, move CSV
scripts/atom/run_atom.py                 # server + GSM8K + throughput → CSV
scripts/atom/configs/default.yaml        # K3 TP8 runtime + ATOM_* env block
```

Plus one entry appended to `models.json`, and a blueprint row in `README.md`.

## models.json entry

```json
{
  "name": "pyt_atom_kimi-k3",
  "url": "",
  "dockerfile": "docker/pyt_atom",
  "scripts": "scripts/atom/run.sh",
  "data": "huggingface",
  "n_gpus": "8",
  "owner": "mad.support@amd.com",
  "training_precision": "",
  "multiple_results": "perf_Kimi-K3.csv",
  "tags": ["pyt", "atom", "kimi", "inference", "gfx950"],
  "timeout": -1,
  "args": "--model_repo moonshotai/Kimi-K3 --config configs/default.yaml"
}
```

## Dockerfile

Follows `docker/pyt_vllm.ubuntu.amd.Dockerfile`: MIT header, CONTEXT line,
`ARG BASE_DOCKER`, `USER root`, `WORKDIR /workspace`, `ENTRYPOINT [""]`.

```dockerfile
# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
ARG BASE_DOCKER=rocm/atom-dev:rocm7.2.4_ubuntu24.04_py3.12_pytorch2.10.0_20260727_kimi_k3
FROM $BASE_DOCKER
```

## configs/default.yaml

Captures the article's runtime verbatim: TP8, fp8 KV cache, 16384 max-model-len,
64 max-num-seqs, 10240 max-num-batched-tokens, 0.93 gpu-mem-util, block-size 128,
prefix caching disabled, plus the 11 ATOM_*/AITER_* env vars from the article.

## run_atom.py flow

1. Load config, export `env` block.
2. Launch `python -m atom.entrypoints.openai_server --model moonshotai/Kimi-K3
   --kv_cache_dtype fp8 -tp 8 --trust-remote-code --max-model-len 16384
   --max-num-seqs 64 --max-num-batched-tokens 10240 --gpu-memory-utilization 0.93
   --block-size 128 --no-enable_prefix_caching` via `subprocess.Popen`.
3. Poll `http://localhost:8000/v1/models` until ready (30-min timeout), reusing
   the wait-loop pattern from `run_vllm.py`.
4. **Accuracy**: `lm_eval --model local-completions --tasks gsm8k --num_fewshot 5`
   over the full 1319 samples (matches the article).
5. **Throughput**: `vllm bench serve` against the OpenAI endpoint if present in the
   image; otherwise fall back to ATOM's native bench or a small OpenAI-client loop.
6. Kill server (SIGINT to children then parent, as in `run_vllm.py`), write
   `perf_Kimi-K3.csv`.

## Result CSV

```csv
model,benchmark,tp,performance,metric,unit
Kimi-K3,gsm8k_5shot,8,<acc>,exact_match,flexible-extract,percent
Kimi-K3,serving,8,<tput>,throughput_tot,tok/sec
```

## Trade-offs / open items

- **Parallel script tree** rather than forking `run_vllm.py`: some duplication
  (arg parsing, server-wait loop) accepted to isolate ATOM's distinct CLI.
- **Throughput tool** (step 5): presence of `vllm bench serve` in the ATOM image
  is the one runtime unknown; fallback path defined above.
- **README**: add a Kimi-K3/ATOM blueprint row; no separate `benchmark/atom/`
  README for this Day-0 adoption unless requested.
