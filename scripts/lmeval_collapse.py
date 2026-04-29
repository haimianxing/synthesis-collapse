"""
Phase 3: Collapse Prevention — Per-round data → lm-eval
Greedy R0/R3/R7 vs QD R0/R3/R7 on 6 benchmarks.

STRATEGY=greedy GPU=4 | STRATEGY=qd GPU=5
"""
import os, sys, json, random, re, torch, numpy as np, subprocess, glob
from pathlib import Path
from collections import defaultdict

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

GPU_ID = int(os.environ.get("GPU_ID", "4"))
STRATEGY = os.environ.get("STRATEGY", "greedy")

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
MATH_TOP_K = 10
PYTHON = "/home/zcz/miniconda3/envs/unsloth/bin/python"

RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/lmeval")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARKS = ["piqa", "arc_challenge", "hellaswag", "winogrande", "gsm8k", "mmlu"]
FEWSHOT = {"piqa": 5, "arc_challenge": 25, "hellaswag": 10, "winogrande": 5, "gsm8k": 5, "mmlu": 5}
ROUNDS = [0, 3, 7]

print(f"=== Phase 3: {STRATEGY.upper()} Collapse (GPU {GPU_ID}) ===", flush=True)

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
print(f"Pool: {len(pool)} samples", flush=True)

# ============ Per-Round ============
all_results_file = RESULTS_DIR / f"{STRATEGY}_collapse_results.json"
all_results = json.load(open(all_results_file)) if all_results_file.exists() else {}

for rnd in ROUNDS:
    config_name = f"{STRATEGY}_r{rnd}"
    merged_dir = RESULTS_DIR / f"merged_{config_name}"
    train_done = RESULTS_DIR / f"{config_name}_train.done"

    if config_name in all_results and all_results[config_name].get("status") == "completed":
        print(f"  {config_name}: done, skipping", flush=True)
        continue

    cells = load_cells(STRATEGY, rnd)
    if not cells:
        print(f"  {config_name}: no cells, skipping", flush=True)
        continue

    # Train if needed
    if not train_done.exists() or not merged_dir.exists():
        cell_to_items = defaultdict(list)
        for item in pool:
            if item['cell'] in cells:
                cell_to_items[item['cell']].append(item)
        selected = []
        for cell, items in cell_to_items.items():
            selected.extend(sorted(items, key=lambda x: x['quality'], reverse=True)[:MATH_TOP_K])

        texts = [fmt_sample(s) for s in selected]
        print(f"\n  {config_name}: {len(cells)} cells → {len(selected)} samples", flush=True)

        if len(texts) < 5:
            print(f"  Too few samples, skipping", flush=True)
            continue

        torch.manual_seed(42); random.seed(42); np.random.seed(42)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True)
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32,
            target_modules=["q_proj","k_proj","v_proj","o_proj"], lora_dropout=0.05, task_type="CAUSAL_LM"))

        ds = Dataset.from_dict({"text": texts})
        trainer = SFTTrainer(model=model, args=SFTConfig(
            output_dir=str(RESULTS_DIR / f"ckpt_{config_name}"), num_train_epochs=3,
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
        train_done.touch()
        print(f"  Saved merged model", flush=True)

    # Eval
    print(f"  Evaluating {config_name}...", flush=True)
    round_results = {"config": config_name, "round": rnd, "n_cells": len(cells), "status": "unknown"}

    for bench in BENCHMARKS:
        cmd = [
            PYTHON, "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={merged_dir},trust_remote_code=True,dtype=bfloat16",
            "--tasks", bench,
            "--num_fewshot", str(FEWSHOT[bench]),
            "--batch_size", "auto",
            "--output_path", str(RESULTS_DIR / f"{config_name}_{bench}"),
        ]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            output = proc.stdout + "\n" + proc.stderr

            for metric in ["acc_norm", "acc"]:
                pattern = rf'\|\s*{re.escape(bench)}\s*\|[^|]*\|[^|]*\|[^|]*\|{metric}\s*\|[↑↓]\s*\|\s*([\d.]+)'
                m = re.search(pattern, output)
                if m:
                    round_results[f"{bench}_{metric}"] = float(m.group(1))
                    break

            if f"{bench}_acc" not in round_results and f"{bench}_acc_norm" not in round_results:
                for rf in sorted(glob.glob(str(RESULTS_DIR / f"{config_name}_{bench}" / "**" / "*.json"), recursive=True)):
                    try:
                        with open(rf) as f: lm_r = json.load(f)
                        if "results" in lm_r:
                            for _, vals in lm_r["results"].items():
                                for key in ["acc_norm,none", "acc,none", "acc_norm", "acc"]:
                                    if key in vals:
                                        round_results[f"{bench}_{key.split(',')[0]}"] = vals[key]
                                        break
                                if f"{bench}_acc" in round_results or f"{bench}_acc_norm" in round_results: break
                    except: continue

            score = round_results.get(f"{bench}_acc_norm", round_results.get(f"{bench}_acc", "?"))
            print(f"    {bench}: {score}", flush=True)
        except Exception as e:
            round_results[f"{bench}_error"] = str(e)
            print(f"    {bench}: ERROR - {e}", flush=True)

    round_results["status"] = "completed"
    all_results[config_name] = round_results
    with open(all_results_file, "w") as f:
        json.dump(all_results, f, indent=2)

# Summary
print(f"\n{'='*60}", flush=True)
print(f"{STRATEGY.upper()} COLLAPSE RESULTS", flush=True)
print(f"{'='*60}", flush=True)
for cn, res in sorted(all_results.items()):
    if res.get("status") != "completed": continue
    print(f"  {cn} (cells={res.get('n_cells','?')}):", flush=True)
    for bench in BENCHMARKS:
        s = res.get(f"{bench}_acc_norm", res.get(f"{bench}_acc", "?"))
        print(f"    {bench:20s}: {s}", flush=True)
print(f"{'='*60}", flush=True)
