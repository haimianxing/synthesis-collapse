"""
Exp B: Difficulty-Stratified HumanEval Evaluation (v2)
Uses unsloth conda env (Python 3.10, peft 0.18.1, torch 2.6+cu124).
Evaluates 1.5B code models + base models on difficulty-stratified HumanEval.

Usage:
  CUDA_VISIBLE_DEVICES=1 /home/zcz/miniconda3/envs/unsloth/bin/python run_difficulty_stratified.py
"""
import os
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
import json, torch, numpy as np, subprocess, random, time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset

MODEL_15B = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-1___5B-Instruct"
RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results")
OUT_DIR = RESULTS_DIR / "difficulty_stratified"
OUT_DIR.mkdir(exist_ok=True)

print(f"=== Difficulty-Stratified HumanEval Evaluation ===", flush=True)

# Load HumanEval from cache
try:
    humaneval = load_dataset('openai/openai_humaneval', split='test')
except:
    # Try loading from the local cache directly
    cache_path = "/home/zcz/.cache/huggingface/datasets/openai___openai_humaneval/openai_humaneval/0.0.0/7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"
    humaneval = load_dataset('json', data_files=str(Path(cache_path) / "*.json"), split='train')

# Classify HumanEval problems by difficulty (prompt length proxy)
prompt_lengths = [(i, len(ex['prompt'])) for i, ex in enumerate(humaneval)]
prompt_lengths.sort(key=lambda x: x[1])
n = len(prompt_lengths)

easy_idx = set(i for i, _ in prompt_lengths[:n//3])
medium_idx = set(i for i, _ in prompt_lengths[n//3:2*n//3])
hard_idx = set(i for i, _ in prompt_lengths[2*n//3:])

# Also by test case complexity
test_lens = [(i, len(ex['test'])) for i, ex in enumerate(humaneval)]
test_lens.sort(key=lambda x: x[1])
hard_test_idx = set(i for i, _ in test_lens[n//2:])
combined_hard = hard_idx | hard_test_idx

print(f"Easy: {len(easy_idx)}, Medium: {len(medium_idx)}, Hard: {len(hard_idx)}, Combined hard: {len(combined_hard)}", flush=True)


def execute_code_safely(code_str, test_cases, timeout=5):
    for test in test_cases:
        try:
            result = subprocess.run(['python3', '-c', code_str + "\n" + test],
                                    timeout=timeout, capture_output=True, text=True)
            if result.returncode == 0: return True
        except:
            pass
    return False


def eval_model_on_subsets(model, tokenizer, label=""):
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

# ===== Part 1: 1.5B LoRA models =====
print("\n=== Part 1: 1.5B LoRA models ===", flush=True)
code_dir = RESULTS_DIR / "code_8seed"

# Evaluate QD, Greedy, Random with 3 seeds each for stats
for method in ["qd_200", "greedy_200", "random_200"]:
    for seed in [42, 123, 456]:
        model_dir = code_dir / f"model_{method}_s{seed}" / "lora"
        if not model_dir.exists():
            print(f"  SKIP {method}_s{seed}: no model", flush=True)
            continue

        key = f"1.5b_{method}_s{seed}"
        print(f"\n  Loading {key}...", flush=True)
        t0 = time.time()

        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_15B, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_15B, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to("cuda:0")
        model = PeftModel.from_pretrained(base, str(model_dir))
        model.eval()

        results = eval_model_on_subsets(model, tokenizer, key)
        all_results[key] = {'model': '1.5b', 'method': method, 'seed': seed, 'results': results}
        print(f"    ({time.time()-t0:.0f}s)", flush=True)

        del model, base; torch.cuda.empty_cache()

# ===== Part 2: Base models (no fine-tuning) =====
print("\n=== Part 2: Base models (no fine-tuning) ===", flush=True)
for model_name, model_path in [("1.5b_base", MODEL_15B)]:
    print(f"\n  Loading {model_name}...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda:0")
    model.eval()

    results = eval_model_on_subsets(model, tokenizer, model_name)
    all_results[model_name] = {'model': model_name, 'results': results}
    print(f"    ({time.time()-t0:.0f}s)", flush=True)

    del model; torch.cuda.empty_cache()

# ===== Save results =====
out_file = OUT_DIR / "stratified_results.json"
with open(out_file, 'w') as f:
    json.dump(all_results, f, indent=2)

# ===== Analysis =====
print(f"\n{'='*60}", flush=True)
print("DIFFICULTY-STRATIFIED ANALYSIS", flush=True)
print(f"{'='*60}", flush=True)

# Base model reference
if '1.5b_base' in all_results:
    r = all_results['1.5b_base']['results']
    print(f"\n1.5b_base:", flush=True)
    for subset in ['full', 'easy', 'medium', 'hard_prompt', 'combined_hard']:
        if subset in r:
            print(f"  {subset:15s}: {r[subset]['pass1']*100:5.1f}%", flush=True)

# QD vs Greedy per difficulty tier
for subset in ['full', 'easy', 'medium', 'hard_prompt', 'combined_hard']:
    qd_accs = [all_results[k]['results'][subset]['pass1']
               for k in all_results if 'qd_200' in k and subset in all_results[k]['results']]
    g_accs = [all_results[k]['results'][subset]['pass1']
              for k in all_results if 'greedy_200' in k and 'per_cell' not in k and subset in all_results[k]['results']]
    r_accs = [all_results[k]['results'][subset]['pass1']
              for k in all_results if 'random_200' in k and subset in all_results[k]['results']]

    if qd_accs and g_accs:
        qd_m = np.mean(qd_accs)*100
        g_m = np.mean(g_accs)*100
        r_m = np.mean(r_accs)*100 if r_accs else 0
        gap = qd_m - g_m
        print(f"\n{subset:15s}: QD={qd_m:5.1f}% Greedy={g_m:5.1f}% Random={r_m:5.1f}% | QD-Greedy={gap:+.1f}pp", flush=True)

print(f"\nSaved to {out_file}", flush=True)
