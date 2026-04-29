"""Math Domain Matched-k Downstream Evaluation

CRITICAL FIX: Previous experiment compared QD (n_train=75, archive size)
vs Greedy/Random (n_train=500). This was an UNFAIR comparison.

This script fixes QD selection to fill to k (like Code V10):
  1. Per-cell dedup: best per cell (23 cells in Math)
  2. Quality fill: remaining slots from pool by quality

Then compares ALL strategies at MATCHED k values.

Usage:
  python math_matched_k.py --gpu 4 --config k200    # GPU 4
  python math_matched_k.py --gpu 5 --config k500    # GPU 5
  python math_matched_k.py --gpu 6 --config control  # GPU 6 (k=75)
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys, json, random, re, torch, numpy as np, time, argparse, shutil
from pathlib import Path
from collections import Counter, defaultdict
from datasets import load_dataset

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-1___5B-Instruct"
GRID_RES = 10
N_TEST = 200
ALL_SEEDS = [42, 123, 456, 789, 1024]

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, required=True)
parser.add_argument('--config', type=str, required=True,
                    choices=['k200', 'k500', 'control'])
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/math_matched_k")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Set k based on config
K_MAP = {'k200': 200, 'k500': 500, 'control': 75}
K = K_MAP[args.config]

print(f"=== Math Matched-k: config={args.config}, k={K}, GPU={args.gpu} ===", flush=True)

print("Loading GSM8K...", flush=True)
train_ds = load_dataset('gsm8k', 'main', split='train')
test_ds = load_dataset('gsm8k', 'main', split='test')
print(f"Train: {len(train_ds)}, Test: {len(test_ds)}", flush=True)

def extract_answer(text):
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '')
    return None

def compute_descriptors(problem, solution):
    steps = solution.count('<<') + 1
    sol_len = len(solution)
    is_multi_step = 1 if steps >= 3 else 0
    difficulty = min(sol_len / 500.0, 1.0)
    return {'difficulty': difficulty, 'num_steps': min(steps / 10.0, 1.0), 'is_multi_step': is_multi_step}

def get_cell(desc):
    return (int(desc['difficulty'] * GRID_RES), int(desc['num_steps'] * GRID_RES), int(desc['is_multi_step'] * GRID_RES))

print("Building pool...", flush=True)
pool = []
for ex in train_ds:
    q, a = ex['question'], ex['answer']
    ans = extract_answer(a)
    if ans is None: continue
    desc = compute_descriptors(q, a)
    quality = min(len(a) / 300.0, 1.0)
    pool.append({'question': q, 'answer': a, 'answer_num': ans,
                 'descriptors': desc, 'quality': quality, 'cell': get_cell(desc)})
print(f"Pool: {len(pool)} samples", flush=True)

# Count cells
cells_in_pool = set(p['cell'] for p in pool)
print(f"Unique cells in pool: {len(cells_in_pool)}", flush=True)

def select_qd_matched(items, k):
    """FIXED: Per-cell dedup + quality fill to k (like Code V10)"""
    # Step 1: Best per cell
    grid = {}
    for item in items:
        cell = item['cell']
        if cell not in grid or item['quality'] > grid[cell]['quality']:
            grid[cell] = item
    deduped = list(grid.values())
    deduped_cells = set(x['cell'] for x in deduped)

    # Step 2: Fill remaining by quality
    used = set(id(x) for x in deduped)
    remaining = sorted([x for x in items if id(x) not in used],
                       key=lambda x: x['quality'], reverse=True)
    selected = deduped + remaining[:max(0, k - len(deduped))]
    return selected[:k]

def select_greedy(items, k):
    return sorted(items, key=lambda x: x['quality'], reverse=True)[:k]

def select_random(items, k, seed=42):
    return random.Random(seed).sample(items, min(k, len(items)))

def finetune_and_eval(train_samples, config_name, seed):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    ckpt_dir = RESULTS_DIR / f"ckpt_{config_name}_s{seed}"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0.05, task_type="CAUSAL_LM"))

    def fmt(s):
        return f"<|im_start|>system\nSolve the math problem step by step.<|im_end|>\n<|im_start|>user\n{s['question']}<|im_end|>\n<|im_start|>assistant\n{s['answer'][:768]}<|im_end|>"

    texts = [fmt(s) for s in train_samples]

    class CodeDataset(torch.utils.data.Dataset):
        def __init__(self, texts, tokenizer, max_len=768):
            self.texts = texts; self.tokenizer = tokenizer; self.max_len = max_len
        def __len__(self): return len(self.texts)
        def __getitem__(self, idx):
            enc = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_len,
                                 padding='max_length', return_tensors='pt')
            return {'input_ids': enc['input_ids'].squeeze(),
                    'attention_mask': enc['attention_mask'].squeeze(),
                    'labels': enc['input_ids'].squeeze()}

    dataset = CodeDataset(texts, tokenizer)
    trainer = Trainer(model=model, args=TrainingArguments(
        output_dir=str(ckpt_dir), num_train_epochs=3, per_device_train_batch_size=4,
        gradient_accumulation_steps=4, learning_rate=2e-4, bf16=True,
        logging_steps=50, save_strategy="no", report_to="none", seed=seed,
        remove_unused_columns=False),
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False))
    trainer.train()

    # Merge and save
    merged = model.merge_and_unload()
    merged_path = str(RESULTS_DIR / f"merged_{config_name}_s{seed}")
    merged.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    del trainer, model, merged; torch.cuda.empty_cache()

    # Evaluate
    print(f"  Evaluating {config_name}_s{seed}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True)
    eval_model = AutoModelForCausalLM.from_pretrained(
        merged_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    eval_model.eval()

    correct = total = 0
    per_diff = defaultdict(lambda: {"c": 0, "t": 0})

    for i, ex in enumerate(test_ds):
        if i >= N_TEST: break
        test_ans = extract_answer(ex['answer'])
        desc = compute_descriptors(ex['question'], ex['answer'])
        db = "easy" if desc['difficulty'] < 0.33 else ("medium" if desc['difficulty'] < 0.66 else "hard")

        msgs = [{"role":"system","content":"Solve the math problem step by step."},
                {"role":"user","content":ex['question']}]
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(txt, return_tensors="pt", truncation=True, max_length=768).to(eval_model.device)
        with torch.no_grad():
            out = eval_model.generate(**inp, max_new_tokens=256, do_sample=False,
                                       pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)

        nums = re.findall(r'-?[\d,]+\.?\d*', resp.replace(',', ''))
        pred = nums[-1] if nums else None
        ok = False
        if pred and test_ans:
            try:
                if abs(float(pred) - float(test_ans)) < 0.01: ok = True; correct += 1
            except: pass
        total += 1; per_diff[db]["t"] += 1
        if ok: per_diff[db]["c"] += 1

    del eval_model; torch.cuda.empty_cache()
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    shutil.rmtree(merged_path, ignore_errors=True)

    acc = correct / total if total > 0 else 0
    n_cells = len(set(s['cell'] for s in train_samples))

    result = {
        'config': args.config, 'k': K, 'seed': seed,
        'n_train': len(train_samples), 'n_cells': n_cells,
        'accuracy': round(acc, 4), 'correct': correct, 'total': total,
        'per_difficulty': {k: round(v["c"]/max(v["t"],1), 4) for k,v in per_diff.items()},
        'model': '1.5B'
    }
    print(f"  Acc: {acc:.4f} ({correct}/{total}), cells={n_cells}", flush=True)
    return result

# Run all seeds
all_results = {}
results_file = RESULTS_DIR / f"results_{args.config}_gpu{args.gpu}.json"

for seed in ALL_SEEDS:
    print(f"\n=== Seed {seed} ===", flush=True)

    # Select data for each strategy
    qd_sel = select_qd_matched(pool, K)
    greedy_sel = select_greedy(pool, K)
    random_sel = select_random(pool, K, seed)

    strategies = {
        'qd': qd_sel,
        'greedy': greedy_sel,
        'random': random_sel,
    }

    for strat_name, sel in strategies.items():
        config_key = f"{strat_name}_{args.config}_s{seed}"

        # Skip if already done
        if results_file.exists():
            existing = json.load(open(results_file))
            if config_key in existing:
                print(f"  SKIP {config_key}: already done", flush=True)
                all_results.update(existing)
                continue

        n_cells = len(set(s['cell'] for s in sel))
        pct_correct = sum(1 for s in sel if s['quality'] >= 0.5) / len(sel) * 100
        print(f"  {strat_name}: n={len(sel)}, cells={n_cells}, quality>0.5: {pct_correct:.0f}%", flush=True)

        t0 = time.time()
        result = finetune_and_eval(sel, f"{strat_name}_{args.config}", seed)
        result['elapsed'] = round(time.time() - t0)
        all_results[config_key] = result

        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

# Final summary
print(f"\n=== SUMMARY: {args.config} (k={K}) ===", flush=True)
for strat in ['qd', 'greedy', 'random']:
    accs = []
    for seed in ALL_SEEDS:
        key = f"{strat}_{args.config}_s{seed}"
        if key in all_results:
            accs.append(all_results[key]['accuracy'] * 100)
    if accs:
        print(f"  {strat}: {np.mean(accs):.1f} ± {np.std(accs):.1f}% (n={len(accs)})", flush=True)

# ANOVA if all complete
try:
    from scipy.stats import f_oneway
    groups = {}
    for strat in ['qd', 'greedy', 'random']:
        accs = [all_results[f"{strat}_{args.config}_s{s}"]['accuracy'] * 100
                for s in ALL_SEEDS if f"{strat}_{args.config}_s{s}" in all_results]
        if len(accs) == len(ALL_SEEDS):
            groups[strat] = accs

    if len(groups) == 3:
        f_stat, p_val = f_oneway(groups['qd'], groups['greedy'], groups['random'])
        ss_total = sum((x - np.mean(list(groups.values())))**2 for accs in groups.values() for x in accs)
        ss_between = sum(len(v) * (np.mean(v) - np.mean([x for accs in groups.values() for x in accs]))**2 for v in groups.values())
        eta_sq = ss_between / ss_total if ss_total > 0 else 0
        print(f"\n  ANOVA: F={f_stat:.2f}, p={p_val:.4f}, η²={eta_sq:.3f}", flush=True)
        print(f"  Compare: Code η²=0.031 (invisible) vs Math η²={eta_sq:.3f}", flush=True)
except Exception as e:
    print(f"  ANOVA failed: {e}", flush=True)

with open(results_file, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\n=== DONE: {args.config} ===", flush=True)
