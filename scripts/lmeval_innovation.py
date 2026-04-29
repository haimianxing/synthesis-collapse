"""
Phase 2: Innovation Point — QD vs Greedy vs Random → lm-eval
Fine-tune Qwen2.5-1.5B, merge LoRA, evaluate on 6 benchmarks.

CONFIG=random GPU=4 | CONFIG=greedy GPU=5 | CONFIG=qd GPU=6
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

RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/lmeval")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MERGED_DIR = RESULTS_DIR / f"merged_{CONFIG}_500"
MERGED_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARKS = ["piqa", "arc_challenge", "hellaswag", "winogrande", "gsm8k", "mmlu"]
FEWSHOT = {"piqa": 5, "arc_challenge": 25, "hellaswag": 10, "winogrande": 5, "gsm8k": 5, "mmlu": 5}

print(f"=== Phase 2: {CONFIG.upper()}-500 (GPU {GPU_ID}) ===", flush=True)

# ============ Selection Functions ============
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

def stratified_select(pool_list, target_cells, n_total):
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
        random.seed(42); selected = random.sample(selected, n_total)
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

# ============ Build Pool & Select ============
results_file = RESULTS_DIR / f"{CONFIG}_500_results.json"
train_done_file = RESULTS_DIR / f"{CONFIG}_500_train.done"

if train_done_file.exists() and MERGED_DIR.exists():
    print(f"Training already done, loading from {MERGED_DIR}", flush=True)
else:
    print("Loading GSM8K pool...", flush=True)
    gsm8k_full = load_dataset("gsm8k", "main", split="train")
    pool = []
    for ex in gsm8k_full:
        q, a = ex['question'], ex['answer']
        ans = extract_answer(a)
        if ans and len(a) > 20:
            c = get_math_cell(a)
            if c: pool.append({'question': q, 'answer': a, 'cell': c, 'quality': min(len(a)/300.0, 1.0)})

    if CONFIG == "random":
        random.seed(42)
        selected = random.sample(pool, min(500, len(pool)))
    elif CONFIG == "greedy":
        selected = stratified_select(pool, load_cells("greedy", 7), 500)
    elif CONFIG == "qd":
        selected = stratified_select(pool, load_cells("qd", 7), 500)

    texts = [fmt_sample(s) for s in selected]
    print(f"Selected: {len(selected)} samples, {len(set(s['cell'] for s in selected))} cells", flush=True)

    # Fine-tune
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj"], lora_dropout=0.05, task_type="CAUSAL_LM"))

    ds = Dataset.from_dict({"text": texts})
    trainer = SFTTrainer(model=model, args=SFTConfig(
        output_dir=str(RESULTS_DIR / f"ckpt_{CONFIG}_500"), num_train_epochs=3,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=2e-4, logging_steps=50, save_strategy="no",
        bf16=True, report_to="none", max_length=768,
        dataset_text_field="text", packing=False),
        train_dataset=ds, processing_class=tokenizer)
    trainer.train()

    # Merge and save full model
    model.eval()
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(MERGED_DIR))
    tokenizer.save_pretrained(str(MERGED_DIR))
    del model, merged_model; torch.cuda.empty_cache()

    train_done_file.touch()
    print(f"Saved merged model to {MERGED_DIR}", flush=True)

# ============ lm-eval Evaluation ============
all_results = json.load(open(results_file)) if results_file.exists() else {}

print(f"\n--- Evaluating on benchmarks ---", flush=True)
for bench in BENCHMARKS:
    if bench in all_results and all_results[bench].get("status") == "completed":
        score = all_results[bench].get('acc_norm', all_results[bench].get('acc', '?'))
        print(f"  {bench}: done ({score}), skipping", flush=True)
        continue

    print(f"  Running {bench}...", flush=True)
    cmd = [
        PYTHON, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={MERGED_DIR},trust_remote_code=True,dtype=bfloat16",
        "--tasks", bench,
        "--num_fewshot", str(FEWSHOT[bench]),
        "--batch_size", "auto",
        "--output_path", str(RESULTS_DIR / f"{CONFIG}_500_{bench}"),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    output = proc.stdout + "\n" + proc.stderr

    result = {"benchmark": bench, "config": CONFIG, "status": "unknown"}

    # Parse lm-eval table output
    for metric in ["acc_norm", "acc"]:
        pattern = rf'\|\s*{re.escape(bench)}\s*\|[^|]*\|[^|]*\|[^|]*\|{metric}\s*\|[↑↓]\s*\|\s*([\d.]+)'
        m = re.search(pattern, output)
        if m:
            result[metric] = float(m.group(1))
            break

    # Fallback: result JSON files
    if "acc" not in result and "acc_norm" not in result:
        for rf in sorted(glob.glob(str(RESULTS_DIR / f"{CONFIG}_500_{bench}" / "**" / "*.json"), recursive=True)):
            try:
                with open(rf) as f: lm_result = json.load(f)
                if "results" in lm_result:
                    for _, vals in lm_result["results"].items():
                        for key in ["acc_norm,none", "acc,none", "acc_norm", "acc"]:
                            if key in vals:
                                result[key.split(",")[0]] = vals[key]
                                break
                        if "acc" in result or "acc_norm" in result: break
            except: continue

    result["status"] = "completed"
    all_results[bench] = result
    score = result.get('acc_norm', result.get('acc', '?'))
    print(f"  {bench}: {score}", flush=True)

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

# Summary
print(f"\n{'='*60}", flush=True)
print(f"{CONFIG.upper()}-500 RESULTS", flush=True)
print(f"{'='*60}", flush=True)
for b, r in all_results.items():
    s = r.get('acc_norm', r.get('acc', '?'))
    print(f"  {b:20s}: {s}", flush=True)
print(f"{'='*60}", flush=True)
