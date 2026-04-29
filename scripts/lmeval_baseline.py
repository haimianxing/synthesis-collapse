"""
Phase 1: Baseline lm-eval on Qwen2.5-1.5B-Instruct
6 small multi-domain benchmarks, fully integrated pipeline.

GPU 1: Baseline evaluation
"""
import os, sys, json, subprocess, re, glob
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
GPU_ID = os.environ.get("GPU_ID", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-1___5B-Instruct"
RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/lmeval")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PYTHON = "/home/zcz/miniconda3/envs/unsloth/bin/python"

BENCHMARKS = {
    "piqa":          {"num_fewshot": 5,  "desc": "Physical reasoning"},
    "arc_challenge": {"num_fewshot": 25, "desc": "Reasoning"},
    "hellaswag":     {"num_fewshot": 10, "desc": "Commonsense"},
    "winogrande":    {"num_fewshot": 5,  "desc": "Coreference"},
    "gsm8k":         {"num_fewshot": 5,  "desc": "Math"},
    "mmlu":          {"num_fewshot": 5,  "desc": "Knowledge"},
}

results_file = RESULTS_DIR / "baseline_results.json"
existing = json.load(open(results_file)) if results_file.exists() else {}

print(f"=== Phase 1: Baseline lm-eval (GPU {GPU_ID}) ===", flush=True)

for bench_name, cfg in BENCHMARKS.items():
    if bench_name in existing and existing[bench_name].get("status") == "completed":
        score = existing[bench_name].get('acc_norm', existing[bench_name].get('acc', '?'))
        print(f"  {bench_name}: done ({score}), skipping", flush=True)
        continue

    print(f"\n--- {bench_name} ({cfg['desc']}) ---", flush=True)

    cmd = [
        PYTHON, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={MODEL_PATH},trust_remote_code=True,dtype=bfloat16",
        "--tasks", bench_name,
        "--num_fewshot", str(cfg["num_fewshot"]),
        "--batch_size", "auto",
        "--output_path", str(RESULTS_DIR / f"baseline_{bench_name}"),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    output = proc.stdout + "\n" + proc.stderr

    result = {"benchmark": bench_name, "desc": cfg["desc"], "status": "unknown"}

    # Parse from lm-eval table output: |task|version|...|acc|↑|0.1234|...
    for metric in ["acc_norm", "acc"]:
        pattern = rf'\|{bench_name}\s*\|[^|]*\|[^|]*\|[^|]*\|{metric}\s*\|↑\s*\|\s*([\d.]+)'
        m = re.search(pattern, output)
        if m:
            result[metric] = float(m.group(1))
            break
        # Alternative: look for any acc line
        pattern2 = rf'\|\s*{bench_name}\s*\|.*\|{metric}\s*\|[↑↓]\s*\|\s*([\d.]+)'
        m2 = re.search(pattern2, output)
        if m2:
            result[metric] = float(m2.group(1))
            break

    # Fallback: try result JSON files
    if "acc" not in result and "acc_norm" not in result:
        result_files = sorted(glob.glob(str(RESULTS_DIR / f"baseline_{bench_name}" / "**" / "*.json"), recursive=True))
        for rf in result_files:
            try:
                with open(rf) as f:
                    lm_result = json.load(f)
                if "results" in lm_result:
                    for task_key, task_vals in lm_result["results"].items():
                        for key in ["acc_norm,none", "acc,none", "acc_norm", "acc"]:
                            if key in task_vals:
                                result[key.split(",")[0]] = task_vals[key]
                                break
                        if "acc" in result or "acc_norm" in result:
                            break
            except:
                continue

    result["status"] = "completed"
    existing[bench_name] = result
    score = result.get('acc_norm', result.get('acc', '?'))
    print(f"  {bench_name}: {score}", flush=True)

    with open(results_file, "w") as f:
        json.dump(existing, f, indent=2)

# Summary
print(f"\n{'='*60}", flush=True)
print("BASELINE RESULTS", flush=True)
print(f"{'='*60}", flush=True)
for b, r in existing.items():
    s = r.get('acc_norm', r.get('acc', '?'))
    print(f"  {b:20s}: {s}", flush=True)
print(f"{'='*60}", flush=True)
