"""
Phase 2.5: Large-scale multi-seed lm-eval comparison
N=2000 samples, 3 seeds, 6 benchmarks → statistical significance

Fixes SAC CW1 (missing math results) + MI1 (small data) + MI2 (no error bars)

CONFIG=random/greedy/qd GPU=x
"""
import os, sys, json, random, re, torch, numpy as np, subprocess, glob
from pathlib import Path
from collections import defaultdict

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

GPU_ID = int(os.environ.get("GPU_ID", "4"))
CONFIG = os.environ.get("CONFIG", "random")

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-1___5B-Instruct"
DEVICE = "cuda:0"
GRID_RES = 10
PYTHON = "/home/zcz/miniconda3/envs/unsloth/bin/python"
N_TRAIN = 2000
N_SEEDS = 3
SEEDS = [42, 123, 271]

RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/lmeval_large")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARKS = ["piqa", "arc_challenge", "hellaswag", "winogrande", "gsm8k", "mmlu"]
FEWSHOT = {"piqa": 5, "arc_challenge": 25, "hellaswag": 10, "winogrande": 5, "gsm8k": 5, "mmlu": 5}

print(f"=== Phase 2.5: {CONFIG.upper()}-{N_TRAIN} multi-seed (GPU {GPU_ID}) ===", flush=True)

# ============ Selection ============
def get_math_cell(answer):
    if not answer or len(answer) < 20: return None
    steps = answer.count('<<') + 1
    difficulty = min(len(answer) / 500.0, 1.0)
    is_multi = 1 if steps >= 3 else 0
    return (int(difficulty * GRID_RES), int(min(steps / 10.0, 1.0) * GRID_RES), int(is_multi * GRID_RES))

def extract_answer(text):
    m = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    return m.group(1).replace(',', '') if m else None

def load_cells(strategy, rnd):
    path = Path(f"/mnt/data2/zcz/neurIps-emnlp/neurips/results/math_iterative_v2/{strategy}_archive_r{rnd}.json")
    if not path.exists(): return set()
    return {c for item in json.load(open(path)) if (c := get_math_cell(item.get('answer', '')))}

def stratified_select(pool_list, target_cells, n_total, seed=42):
    random.seed(seed)
    cell_to_items = defaultdict(list)
    for item in pool_list:
        if item['cell'] in target_cells:
            cell_to_items[item['cell']].append(item)
    if not cell_to_items: return []
    per_cell = max(1, n_total // len(cell_to_items))
    selected = []
    for cell in sorted(cell_to_items.keys()):
        items = sorted(cell_to_items[cell], key=lambda x: x['quality'], reverse=True)
        selected.extend(items[:per_cell])
    if len(selected) > n_total:
        selected = random.sample(selected, n_total)
    elif len(selected) < n_total:
        remaining = []
        for cell in sorted(cell_to_items.keys()):
            items = sorted(cell_to_items[cell], key=lambda x: x['quality'], reverse=True)
            remaining.extend(items[per_cell:])
        remaining.sort(key=lambda x: x['quality'], reverse=True)
        for item in remaining:
            if len(selected) >= n_total: break
            if item not in selected: selected.append(item)
    return selected[:n_total]

def fmt_sample(item):
    return f"<|im_start|>system\nSolve the math problem step by step.<|im_end|>\n<|im_start|>user\n{item['question'][:512]}<|im_end|>\n<|im_start|>assistant\n{item['answer'][:1024]}<|im_end|>"

# ============ Build Pool ============
print("Loading GSM8K pool...", flush=True)
gsm8k_full = load_dataset("gsm8k", "main", split="train")
pool = []
for ex in gsm8k_full:
    q, a = ex['question'], ex['answer']
    ans = extract_answer(a)
    if ans and len(a) > 20:
        c = get_math_cell(a)
        if c: pool.append({'question': q, 'answer': a, 'cell': c, 'quality': min(len(a)/300.0, 1.0)})

# ============ Multi-seed training & eval ============
results_file = RESULTS_DIR / f"{CONFIG}_{N_TRAIN}_results.json"
all_results = json.load(open(results_file)) if results_file.exists() else {}

for seed in SEEDS:
    seed_key = f"{CONFIG}_{N_TRAIN}_s{seed}"
    merged_dir = RESULTS_DIR / f"merged_{seed_key}"

    if seed_key in all_results and all_results[seed_key].get("status") == "completed":
        print(f"  {seed_key}: done, skipping", flush=True)
        continue

    # Select data
    if CONFIG == "random":
        random.seed(seed)
        selected = random.sample(pool, min(N_TRAIN, len(pool)))
    elif CONFIG == "greedy":
        selected = stratified_select(pool, load_cells("greedy", 7), N_TRAIN, seed)
    elif CONFIG == "qd":
        selected = stratified_select(pool, load_cells("qd", 7), N_TRAIN, seed)

    texts = [fmt_sample(s) for s in selected]
    cells = set(s['cell'] for s in selected)
    print(f"\n  {seed_key}: {len(selected)} samples, {len(cells)} cells", flush=True)

    # Train
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj"], lora_dropout=0.05, task_type="CAUSAL_LM"))

    ds = Dataset.from_dict({"text": texts})
    trainer = SFTTrainer(model=model, args=SFTConfig(
        output_dir=str(RESULTS_DIR / f"ckpt_{seed_key}"), num_train_epochs=3,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=2e-4, logging_steps=50, save_strategy="no",
        bf16=True, report_to="none", max_length=768,
        dataset_text_field="text", packing=False),
        train_dataset=ds, processing_class=tokenizer)
    trainer.train()

    model.eval()
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    del model, merged_model; torch.cuda.empty_cache()
    print(f"  Saved merged model to {merged_dir}", flush=True)

    # Eval on all benchmarks
    seed_results = {"config": CONFIG, "n_train": N_TRAIN, "seed": seed, "n_cells": len(cells), "status": "unknown"}

    for bench in BENCHMARKS:
        cmd = [
            PYTHON, "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={merged_dir},trust_remote_code=True,dtype=bfloat16",
            "--tasks", bench,
            "--num_fewshot", str(FEWSHOT[bench]),
            "--batch_size", "auto",
            "--output_path", str(RESULTS_DIR / f"{seed_key}_{bench}"),
        ]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            output = proc.stdout + "\n" + proc.stderr

            for metric in ["acc_norm", "acc"]:
                pattern = rf'\|\s*{re.escape(bench)}\s*\|[^|]*\|[^|]*\|[^|]*\|{metric}\s*\|[↑↓]\s*\|\s*([\d.]+)'
                m = re.search(pattern, output)
                if m:
                    seed_results[f"{bench}_{metric}"] = float(m.group(1))
                    break

            if f"{bench}_acc" not in seed_results and f"{bench}_acc_norm" not in seed_results:
                for rf in sorted(glob.glob(str(RESULTS_DIR / f"{seed_key}_{bench}" / "**" / "*.json"), recursive=True)):
                    try:
                        with open(rf) as f: lm_r = json.load(f)
                        if "results" in lm_r:
                            for _, vals in lm_r["results"].items():
                                for key in ["acc_norm,none", "acc,none", "acc_norm", "acc"]:
                                    if key in vals:
                                        seed_results[f"{bench}_{key.split(',')[0]}"] = vals[key]
                                        break
                                if f"{bench}_acc" in seed_results or f"{bench}_acc_norm" in seed_results: break
                    except: continue

            score = seed_results.get(f"{bench}_acc_norm", seed_results.get(f"{bench}_acc", "?"))
            print(f"    {bench}: {score}", flush=True)
        except Exception as e:
            seed_results[f"{bench}_error"] = str(e)
            print(f"    {bench}: ERROR - {e}", flush=True)

    seed_results["status"] = "completed"
    all_results[seed_key] = seed_results
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

# ============ Aggregate ============
print(f"\n{'='*70}", flush=True)
print(f"{CONFIG.upper()}-{N_TRAIN} AGGREGATE (3 seeds)", flush=True)
print(f"{'='*70}", flush=True)

for bench in BENCHMARKS:
    vals = []
    for seed in SEEDS:
        key = f"{CONFIG}_{N_TRAIN}_s{seed}"
        if key in all_results:
            v = all_results[key].get(f"{bench}_acc_norm", all_results[key].get(f"{bench}_acc"))
            if v and v != '?': vals.append(float(v))
    if vals:
        print(f"  {bench:20s}: {np.mean(vals):.4f} ± {np.std(vals):.4f} (n={len(vals)})", flush=True)
    else:
        print(f"  {bench:20s}: no data", flush=True)
print(f"{'='*70}", flush=True)
