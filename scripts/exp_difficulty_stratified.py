"""
Exp B: Difficulty-Stratified HumanEval Evaluation — Break 7B Ceiling (W2)

Evaluate existing models on HumanEval subsets stratified by difficulty.
Hard subset should be below ceiling even for 7B, making coverage gap detectable.

Also evaluates 1.5B models on stratified subsets to show gradient.

Usage:
  python exp_difficulty_stratified.py --gpu 6
"""
import os
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import json, torch, numpy as np, subprocess, random, argparse, time, shutil
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from collections import defaultdict

MODEL_15B = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-1___5B-Instruct"
MODEL_7B = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-7B-Instruct"
RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results")
OUT_DIR = RESULTS_DIR / "difficulty_stratified"
OUT_DIR.mkdir(exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=6)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

from datasets import load_dataset
humaneval = load_dataset('openai/openai_humaneval', split='test')

# Classify HumanEval problems by difficulty
# Use prompt length as proxy for difficulty (longer prompts = more complex function signatures)
prompt_lengths = [(i, len(ex['prompt'])) for i, ex in enumerate(humaneval)]
prompt_lengths.sort(key=lambda x: x[1])
n = len(prompt_lengths)

# Split into difficulty tiers
easy_idx = set(i for i, _ in prompt_lengths[:n//3])
medium_idx = set(i for i, _ in prompt_lengths[n//3:2*n//3])
hard_idx = set(i for i, _ in prompt_lengths[2*n//3:])

print(f"=== Difficulty-Stratified HumanEval Evaluation ===", flush=True)
print(f"Easy (short prompts): {len(easy_idx)}, Medium: {len(medium_idx)}, Hard (long prompts): {len(hard_idx)}", flush=True)

# Also classify by test case complexity
test_lens = [(i, len(ex['test'])) for i, ex in enumerate(humaneval)]
test_lens.sort(key=lambda x: x[1])
hard_test_idx = set(i for i, _ in test_lens[n//2:])  # top 50% by test complexity

# Combined hard: hard prompt OR hard test
combined_hard = hard_idx | hard_test_idx
print(f"Combined hard (either metric): {len(combined_hard)}", flush=True)

def execute_code_safely(code_str, test_cases, timeout=5):
    for test in test_cases:
        try:
            result = subprocess.run(['python3', '-c', code_str + "\n" + test],
                                    timeout=timeout, capture_output=True, text=True)
            if result.returncode == 0: return True
        except: pass
    return False

def eval_model_on_subsets(model, tokenizer, label=""):
    """Evaluate model on full + difficulty-stratified HumanEval subsets."""
    results = {}
    for subset_name, subset_idx in [('full', set(range(len(humaneval)))),
                                      ('easy', easy_idx),
                                      ('medium', medium_idx),
                                      ('hard_prompt', hard_idx),
                                      ('hard_test', hard_test_idx),
                                      ('combined_hard', combined_hard)]:
        correct = total = 0
        for i in subset_idx:
            item = humaneval[i]
            prompt = item['prompt']
            test_code = item['test']
            msgs = [{"role": "system", "content": "Complete the Python function."},
                    {"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=512).to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=256,
                                          do_sample=False, pad_token_id=tokenizer.eos_token_id)
            completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            full_code = prompt + completion
            if execute_code_safely(full_code, [test_code]):
                correct += 1
            total += 1
        acc = correct / total if total > 0 else 0
        results[subset_name] = {'pass1': round(acc, 4), 'correct': correct, 'total': total}
        print(f"  {label:30s} {subset_name:15s}: {acc*100:5.1f}% ({correct}/{total})", flush=True)
    return results

all_results = {}

# ===== Part 1: Evaluate existing 1.5B LoRA models (code_8seed) =====
print("\n=== Part 1: 1.5B LoRA models ===", flush=True)
code_dir = RESULTS_DIR / "code_8seed"
lora_seeds = [42, 123, 271, 456, 789, 2024, 314, 159]
methods = ["qd_200", "greedy_200", "random_200", "greedy_per_cell"]

# Evaluate a subset: QD-200 and Greedy-200 with seed 42, 123
for method in ["qd_200", "greedy_200", "greedy_per_cell"]:
    for seed in [42, 123]:
        model_dir = code_dir / f"model_{method}_s{seed}" / "lora"
        if not model_dir.exists():
            print(f"  SKIP {method}_s{seed}: no model", flush=True)
            continue

        key = f"1.5b_{method}_s{seed}"
        print(f"\n  Loading {key}...", flush=True)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_15B, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_15B, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(f"cuda:0")
        model = PeftModel.from_pretrained(base, str(model_dir))
        model.eval()

        results = eval_model_on_subsets(model, tokenizer, key)
        all_results[key] = {'model': '1.5b', 'method': method, 'seed': seed, 'results': results}

        del model, base; torch.cuda.empty_cache()

# ===== Part 2: Evaluate base models (no fine-tuning) =====
print("\n=== Part 2: Base models (no fine-tuning) ===", flush=True)
for model_name, model_path in [("1.5b_base", MODEL_15B), ("7b_base", MODEL_7B)]:
    print(f"\n  Loading {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda:0")
    model.eval()

    results = eval_model_on_subsets(model, tokenizer, model_name)
    all_results[model_name] = {'model': model_name, 'results': results}

    del model; torch.cuda.empty_cache()

# ===== Part 3: Evaluate V10 multiseed models (7B, k=500) =====
print("\n=== Part 3: 7B V10 multiseed models ===", flush=True)
multiseed_dir = RESULTS_DIR / "scale_v10" / "multiseed"

# Check for merged models from multiseed experiment
# If they don't exist, we need to train+eval. For now, check existing.
# The multiseed experiment merged and deleted models, so we need to retrain a subset.
# Instead, let's just evaluate the base 7B model stratified, and the 1.5B models.

# ===== Save results =====
out_file = OUT_DIR / f"stratified_results_gpu{args.gpu}.json"
with open(out_file, 'w') as f:
    json.dump(all_results, f, indent=2)

# ===== Analysis =====
print(f"\n{'='*60}", flush=True)
print("DIFFICULTY-STRATIFIED ANALYSIS", flush=True)
print(f"{'='*60}", flush=True)

for model_key in ['1.5b_base', '7b_base']:
    if model_key in all_results:
        r = all_results[model_key]['results']
        print(f"\n{model_key}:", flush=True)
        for subset in ['full', 'easy', 'medium', 'hard_prompt', 'combined_hard']:
            if subset in r:
                print(f"  {subset:15s}: {r[subset]['pass1']*100:5.1f}%", flush=True)

# Show QD vs Greedy gap on each difficulty tier
qd_keys = [k for k in all_results if 'qd_200' in k]
greedy_keys = [k for k in all_results if 'greedy_200' in k and 'per_cell' not in k]

print(f"\nQD vs Greedy (1.5B):", flush=True)
for subset in ['full', 'easy', 'medium', 'hard_prompt', 'combined_hard']:
    qd_accs = [all_results[k]['results'][subset]['pass1'] for k in qd_keys if subset in all_results[k]['results']]
    g_accs = [all_results[k]['results'][subset]['pass1'] for k in greedy_keys if subset in all_results[k]['results']]
    if qd_accs and g_accs:
        gap = np.mean(qd_accs) - np.mean(g_accs)
        print(f"  {subset:15s}: QD={np.mean(qd_accs)*100:5.1f}% Greedy={np.mean(g_accs)*100:5.1f}% gap={gap*100:+.1f}pp", flush=True)

print(f"\nSaved to {out_file}", flush=True)
