"""
Focused Fair Comparison — Phase 1 Only
3 configs × 3 seeds = 9 training runs per domain
Math: N=500, Code: N=200

GPU 2: Math fair comparison (random_500, greedy_r7_500, qd_r7_500)
GPU 4: Code fair comparison (random_200, greedy_r7_200, qd_r7_200)
"""
import os, sys, json, random, re, torch, numpy as np, ast
from pathlib import Path
from collections import defaultdict

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

GPU_ID = int(os.environ.get("GPU_ID", "2"))
DOMAIN = os.environ.get("DOMAIN", "math")

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

RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/fair_comparison")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"=== Fair Comparison: GPU {GPU_ID}, DOMAIN={DOMAIN} ===", flush=True)

def get_code_cell(code, prompt=''):
    if not code or len(code) < 20: return None
    difficulty = min(len(code) / 1000.0, 1.0)
    api_count = 0
    try:
        import ast as ast_mod
        tree = ast_mod.parse(code)
        for node in ast_mod.walk(tree):
            if isinstance(node, ast_mod.Call): api_count += 1
            elif isinstance(node, ast_mod.Import): api_count += len(node.names)
            elif isinstance(node, ast_mod.ImportFrom): api_count += len(node.names)
    except: api_count = len(re.findall(r'\b\w+\.\w+\(', code))
    has_debug = 1 if ('try:' in code or 'except' in code or 'assert' in code) else 0
    return (int(difficulty * GRID_RES), int(min(api_count / 10.0, 1.0) * GRID_RES), int(has_debug * GRID_RES))

def get_math_cell(answer):
    if not answer or len(answer) < 20: return None
    steps = answer.count('<<') + 1
    difficulty = min(len(answer) / 500.0, 1.0)
    is_multi = 1 if steps >= 3 else 0
    return (int(difficulty * GRID_RES), int(min(steps / 10.0, 1.0) * GRID_RES), int(is_multi * GRID_RES))

def extract_answer(text):
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    return match.group(1).replace(',', '') if match else None

if DOMAIN == "code":
    ITER_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/code_iterative_v2")
    FIXED_N = 200
    print("Loading MBPP...", flush=True)
    try: mbpp_train = load_dataset("mbpp", "sanitized", split="train")
    except: mbpp_train = load_dataset("mbpp", split="train")
    try: mbpp_test = load_dataset("mbpp", "sanitized", split="test")
    except: mbpp_test = load_dataset("mbpp", split="test")
    try: mbpp_val = load_dataset("mbpp", "sanitized", split="validation")
    except: mbpp_val = []
    pool = []
    for ex in list(mbpp_train) + list(mbpp_test) + list(mbpp_val):
        code = ex.get('code', ''); prompt = ex.get('prompt', ex.get('text', ''))
        if code and len(code) > 20:
            c = get_code_cell(code, prompt)
            if c: pool.append({'prompt': prompt, 'code': code, 'cell': c, 'quality': min(len(code)/500.0, 1.0)})
    humaneval = load_dataset("openai_humaneval", split="test")
    def eval_fn(model, tokenizer):
        correct = total = 0
        for ex in humaneval:
            msgs = [{"role":"system","content":"Complete the Python function."},{"role":"user","content":ex['prompt']}]
            txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = tokenizer(txt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=256, temperature=0.0, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
            try: exec_globals = {}; exec(ex['prompt'] + resp, exec_globals); exec(ex['test'], exec_globals); correct += 1
            except: pass
            total += 1
            if total % 40 == 0: print(f"    Eval: {total}/164, correct={correct}", flush=True)
        return correct, total
    def fmt(item):
        return f"<|im_start|>system\nComplete the Python function.<|im_end|>\n<|im_start|>user\n{item['prompt'][:512]}<|im_end|>\n<|im_start|>assistant\n{item['code'][:1024]}<|im_end|>"

elif DOMAIN == "math":
    ITER_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/math_iterative_v2")
    FIXED_N = 500
    print("Loading GSM8K...", flush=True)
    gsm8k_full = load_dataset("gsm8k", "main", split="train")
    random.seed(42)
    gsm8k_test = random.sample(list(load_dataset("gsm8k", "main", split="test")), min(500, len(load_dataset("gsm8k", "main", split="test"))))
    pool = []
    for ex in gsm8k_full:
        q, a = ex['question'], ex['answer']
        ans = extract_answer(a)
        if ans and len(a) > 20:
            c = get_math_cell(a)
            if c: pool.append({'question': q, 'answer': a, 'cell': c, 'quality': min(len(a)/300.0, 1.0)})
    def eval_fn(model, tokenizer):
        correct = total = 0
        for ex in gsm8k_test:
            msgs = [{"role":"system","content":"Solve the math problem step by step."},{"role":"user","content":ex['question']}]
            txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = tokenizer(txt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=256, temperature=0.0, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
            pred_match = re.search(r'####\s*(-?[\d,]+\.?\d*)', resp)
            if not pred_match: pred_match = re.search(r'(\d+\.?\d*)\s*$', resp.strip())
            pred = pred_match.group(1).replace(',', '') if pred_match else None
            gold = extract_answer(ex['answer'])
            if pred and gold and pred.strip() == gold.strip(): correct += 1
            total += 1
            if total % 100 == 0: print(f"    Eval: {total}/{len(gsm8k_test)}, correct={correct}", flush=True)
        return correct, total
    def fmt(item):
        return f"<|im_start|>system\nSolve the math problem step by step.<|im_end|>\n<|im_start|>user\n{item['question'][:512]}<|im_end|>\n<|im_start|>assistant\n{item['answer'][:1024]}<|im_end|>"

print(f"{DOMAIN} pool: {len(pool)} samples, N={FIXED_N}", flush=True)

def load_cells(strategy, rnd):
    path = ITER_DIR / f"{strategy}_archive_r{rnd}.json"
    if not path.exists(): return set()
    cells = set()
    for item in json.load(open(path)):
        c = get_code_cell(item.get('code',''), item.get('prompt','')) if DOMAIN=="code" else get_math_cell(item.get('answer',''))
        if c: cells.add(c)
    return cells

def stratified_select(pool, target_cells, n):
    cell_to_items = defaultdict(list)
    for item in pool:
        if item['cell'] in target_cells:
            cell_to_items[item['cell']].append(item)
    if not cell_to_items: return []
    per_cell = max(1, n // len(cell_to_items))
    selected = []
    for cell in sorted(cell_to_items.keys()):
        items = sorted(cell_to_items[cell], key=lambda x: x['quality'], reverse=True)
        selected.extend(items[:per_cell])
    if len(selected) > n:
        random.seed(42); selected = random.sample(selected, n)
    elif len(selected) < n:
        remaining = []
        for cell in sorted(cell_to_items.keys()):
            items = sorted(cell_to_items[cell], key=lambda x: x['quality'], reverse=True)
            remaining.extend(items[per_cell:])
        remaining.sort(key=lambda x: x['quality'], reverse=True)
        for item in remaining:
            if len(selected) >= n: break
            if item not in selected: selected.append(item)
    return selected[:n]

def finetune_and_eval(texts, config_name, seed=42):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    output_dir = RESULTS_DIR / f"model_{DOMAIN}_{config_name}_s{seed}"
    if len(texts) < 5:
        return {"pass_at_1": 0, "accuracy": 0, "correct": 0, "total": 0, "n_train": len(texts)}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"], lora_dropout=0.05, task_type="CAUSAL_LM"))
    ds = Dataset.from_dict({"text": texts})
    trainer = SFTTrainer(model=model, args=SFTConfig(
        output_dir=str(output_dir), num_train_epochs=3, per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=2e-4, logging_steps=50, save_strategy="no", bf16=True, report_to="none", max_length=768,
        dataset_text_field="text", packing=False), train_dataset=ds, processing_class=tokenizer)
    trainer.train()
    model.eval()
    correct, total = eval_fn(model, tokenizer)
    del model; torch.cuda.empty_cache()
    metric = round(correct/total, 4) if total > 0 else 0
    metric_name = "pass_at_1" if DOMAIN == "code" else "accuracy"
    print(f"  [{config_name} s{seed}] {metric_name}={metric} ({correct}/{total}), n={len(texts)}", flush=True)
    return {metric_name: metric, "correct": correct, "total": total, "n_train": len(texts)}

results_file = RESULTS_DIR / f"{DOMAIN}_fair_results.json"
all_results = json.load(open(results_file)) if results_file.exists() else {}

# Run 3 configs × 3 seeds
for config_name, data in [
    (f"random_{FIXED_N}", lambda: (random.seed(42), random.sample(pool, min(FIXED_N, len(pool))))[1]),
    (f"greedy_r7_{FIXED_N}", lambda: stratified_select(pool, load_cells("greedy", 7), FIXED_N)),
    (f"qd_r7_{FIXED_N}", lambda: stratified_select(pool, load_cells("qd", 7), FIXED_N)),
]:
    if config_name in all_results:
        print(f"  {config_name}: done, skipping", flush=True); continue
    items = data()
    texts = [fmt(s) for s in items]
    # Report cell info
    cells = set(s['cell'] for s in items)
    print(f"\n  {config_name}: {len(items)} samples, {len(cells)} unique cells", flush=True)
    seed_results = []
    for seed in [42, 123, 271]:
        r = finetune_and_eval(texts, config_name, seed)
        r['seed'] = seed
        seed_results.append(r)
    metric_name = "pass_at_1" if DOMAIN == "code" else "accuracy"
    values = [r[metric_name] for r in seed_results]
    agg = {"config": config_name, "n_train": len(texts), "n_cells": len(cells),
           metric_name: round(np.mean(values), 4), "std": round(np.std(values), 4), "seeds": seed_results}
    all_results[config_name] = agg
    print(f"  {config_name}: {metric_name}={agg[metric_name]}±{agg['std']}, n={len(texts)}, cells={len(cells)}", flush=True)
    with open(results_file, "w") as f: json.dump(all_results, f, indent=2, default=str)

print(f"\n{'='*50}", flush=True)
metric_name = "pass_at_1" if DOMAIN == "code" else "accuracy"
for k, v in sorted(all_results.items()):
    print(f"  {k}: {v.get(metric_name,'?')}±{v.get('std','?')}, n={v.get('n_train','?')}, cells={v.get('n_cells','?')}", flush=True)
print(f"{'='*50}", flush=True)
with open(results_file, "w") as f: json.dump(all_results, f, indent=2, default=str)
