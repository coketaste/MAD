#################################################################################
#
# MIT License
#
# Copyright (c) Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################

import os
import csv
import glob
import json
import yaml
import shutil
import signal
import argparse
import subprocess

import psutil

CSV_HEADER = [
    "model",
    "benchmark",
    "tp",
    "inp",
    "out",
    "max_concurrency",
    "cmd",
    "performance",
    "metric",
    "unit",
]

SERVER_URL = "http://localhost:8000"


def parse_args():
    parser = argparse.ArgumentParser(description="Run Kimi-K3 with the ATOM inference engine")
    parser.add_argument("--config", type=str, required=True, help="config yaml file")
    parser.add_argument("--model", type=str, default=None, help="override model repo from config")
    return parser.parse_args()


def build_env(config):
    """Merge the config env block into the process environment."""
    env = os.environ.copy()
    for key, val in config.get("env", {}).items():
        env[str(key)] = str(val)
    return env


def start_server(model, config, env):
    server_cmd = (
        "python -m atom.entrypoints.openai_server "
        f"--model {model} "
        f"--kv_cache_dtype {config['kv_cache_dtype']} "
        f"-tp {config['tp']} "
        f"--trust-remote-code "
        f"--max-model-len {config['max_model_len']} "
        f"--max-num-seqs {config['max_num_seqs']} "
        f"--max-num-batched-tokens {config['max_num_batched_tokens']} "
        f"--gpu-memory-utilization {config['gpu_memory_utilization']} "
        f"--block-size {config['block_size']} "
        f"--no-enable_prefix_caching"
    )
    print(server_cmd, flush=True)
    server = subprocess.Popen(server_cmd, shell=True, env=env)
    return server, server_cmd


def wait_for_server():
    status = subprocess.run(
        f"timeout 1800 bash -c 'until curl -s {SERVER_URL}/v1/models; do sleep 30; done' || exit 1",
        shell=True,
    )
    if status.returncode != 0:
        raise Exception("Server failed to start")
    print("Server contacted successfully", flush=True)


def run_gsm8k(model, config, env):
    """GSM8K 5-shot correctness check against the running ATOM server."""
    model_args = {
        "model": model,
        "max_gen_toks": 2048,
        "num_concurrent": config["max_concurrency"],
        "max_retries": 10,
        "base_url": f"{SERVER_URL}/v1/completions",
    }
    model_args_str = ",".join(f"{k}={v}" for k, v in model_args.items())
    lmeval_cmd = (
        "lm_eval "
        "--model local-completions "
        f"--model_args {model_args_str} "
        "--tasks gsm8k "
        f"--num_fewshot {config['num_fewshot']} "
        f"--batch_size {config['max_concurrency']} "
        f"--limit {config['gsm8k_limit']} "
        "--output_path ./tmp"
    )
    print(lmeval_cmd, flush=True)
    subprocess.run(lmeval_cmd, shell=True, check=True, env=env)

    output_files = glob.glob("./tmp/*/*.json")
    if not output_files:
        raise Exception("No lmeval output files found")
    output_file = output_files[0]
    with open(output_file, "r", encoding="utf-8") as f:
        output = json.load(f)
    shutil.rmtree("./tmp")

    results = []
    gsm8k = output.get("results", {}).get("gsm8k", {})
    if "exact_match,flexible-extract" in gsm8k:
        results.append({
            "benchmark": "gsm8k_5shot",
            "performance": gsm8k["exact_match,flexible-extract"],
            "metric": "exact_match,flexible-extract",
            "unit": "percent",
        })
    return results, lmeval_cmd


def run_throughput(model, config, env):
    """Basic token throughput against the OpenAI-compatible endpoint."""
    num_prompts = int(config["max_concurrency"]) * 8
    output_json = "atom_throughput.json"
    bench_cmd = (
        "vllm bench serve "
        f"--backend openai "
        f"--base-url {SERVER_URL} "
        f"--endpoint /v1/completions "
        f"--model {model} "
        f"--dataset-name random "
        f"--random-input-len {config['inp']} "
        f"--random-output-len {config['out']} "
        f"--max-concurrency {config['max_concurrency']} "
        f"--num-prompts {num_prompts} "
        f"--ignore-eos "
        f"--percentile-metrics ttft,tpot,itl,e2el "
        f"--save-result "
        f"--result-filename {output_json}"
    )
    print(bench_cmd, flush=True)
    proc = subprocess.run(bench_cmd, shell=True, env=env)
    if proc.returncode != 0 or not os.path.exists(output_json):
        print("Warning: throughput benchmark unavailable or failed; skipping throughput metric", flush=True)
        return [], bench_cmd

    with open(output_json, "r", encoding="utf-8") as f:
        output = json.load(f)

    results = []
    if "total_token_throughput" in output:
        metrics = {
            "throughput_tot": (output["total_token_throughput"], "tok/sec"),
            "throughput_gen": (output["output_throughput"], "tok/sec"),
            "median_ttft": (output["median_ttft_ms"], "ms"),
            "median_tpot": (output["median_tpot_ms"], "ms"),
        }
        for metric, (perf, unit) in metrics.items():
            results.append({
                "benchmark": "serving",
                "performance": str(perf),
                "metric": metric,
                "unit": unit,
            })
    return results, bench_cmd


def stop_server(server):
    parent = psutil.Process(server.pid)
    for child in parent.children(recursive=True):
        child.send_signal(signal.SIGINT)
    server.send_signal(signal.SIGINT)
    _ = server.communicate()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        configs = yaml.safe_load(f)

    if os.environ.get("MAD_SECRETS_HFTOKEN"):
        os.environ["HF_TOKEN"] = os.environ["MAD_SECRETS_HFTOKEN"]
    else:
        print("Warning: MAD_SECRETS_HFTOKEN is not set. Set it if a gated model is used.", flush=True)

    for config in configs:
        model = args.model or config["model"]
        model_name = os.path.basename(model)

        # Prefer pre-provisioned weights when MAD supplies a data home.
        if mad_datahome := os.environ.get("MAD_DATAHOME"):
            model = mad_datahome
        else:
            download_command = (
                f"hf download {model} --exclude \"original/*\" \"*.tf\" \"*.onnx\" \"*.flax\" \"*.rust\""
            )
            subprocess.run(download_command, shell=True, check=True)

        env = build_env(config)
        server, server_cmd = start_server(model, config, env)
        results = []
        try:
            wait_for_server()

            gsm8k_results, gsm8k_cmd = run_gsm8k(model, config, env)
            for r in gsm8k_results:
                r["cmd"] = f"{server_cmd};{gsm8k_cmd}"
                results.append(r)

            tput_results, tput_cmd = run_throughput(model, config, env)
            for r in tput_results:
                r["cmd"] = f"{server_cmd};{tput_cmd}"
                results.append(r)
        finally:
            stop_server(server)

        output_csv = "perf_" + model_name + ".csv"
        with open(output_csv, "w", newline="") as outf:
            writer = csv.DictWriter(outf, delimiter=",", fieldnames=CSV_HEADER)
            writer.writeheader()
            for r in results:
                row = {
                    "model": model_name,
                    "tp": config["tp"],
                    "inp": config["inp"],
                    "out": config["out"],
                    "max_concurrency": config["max_concurrency"],
                    **r,
                }
                writer.writerow(row)
        print(f"Wrote {output_csv}", flush=True)


if __name__ == "__main__":
    main()
