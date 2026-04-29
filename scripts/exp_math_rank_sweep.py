"""
Exp: Math Domain LoRA Rank Sweep — Cross-domain validation of detectability threshold

Vary LoRA rank [4, 8, 16, 32, 64] at k=500 with 7B model, 3 methods, 5 seeds.
Uses GSM8K training pool and GSM8K test evaluation.

Purpose: Validate that Eq.eta_rank predictions hold across domains:
  (P1) QD-Greedy gap grows monotonically
  (P2) Random-QD gap shrinks at higher ranks

Usage:
  python exp_math_rank_sweep.py --gpu 1 --rank-group 0   # ranks [4,8]
  python exp_math_rank_sweep.py --gpu 2 --rank-group 1   # ranks [16,32]
  python exp_math_rank_sweep.py --gpu 3 --rank-group 2   # rank [64]
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys, json, random, re, torch, numpy as np, shutil, argparse, time
from pathlib import Path
from collections import defaultdict

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-7B-Instruct"
RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/math_rank_sweep")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MAX_SEQ_LENGTH = 768
K = 500
GRID_RES = 10

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, required=True)
parser.add_argument('--rank-group', type=int, required=True)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
DEVICE = "cuda:0"

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset as HFDataset

# Load GSM8K
print("Loading GSM8K...", flush=True)
train_ds = load_dataset("gsm8k", "main", split="train")
test_ds = load_dataset("gsm8k", "main", split="test")
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

# Build pool
print("Building Math pool...", flush=True)
pool = []
for ex in train_ds:
    q, a = ex['question'], ex['answer']
    ans = extract_answer(a)
    if ans is None: continue
    desc = compute_descriptors(q, a)
    quality = min(len(a) / 300.0, 1.0)
    pool.append({'question': q, 'answer': a, 'answer_num': ans, 'descriptors': desc, 'quality': quality})
print(f"Pool: {len(pool)} items", flush=True)

# Count unique cells
cells = set(get_cell(p['descriptors']) for p in pool)
print(f"Unique cells: {len(cells)}", flush=True)

ALL_SEEDS = [42, 123, 456, 789, 2024]
ALL_RANKS = [4, 8, 16, 32, 64]

def qd_select(items, k, seed=42):
    grid = {}
    for item in items:
        cell = get_cell(item['descriptors'])
        if cell not in grid or item['quality'] > grid[cell]['quality']:
            grid[cell] = item
    return sorted(grid.values(), key=lambda x: x['quality'], reverse=True)[:k]

def greedy_select(items, k, seed=42):
    return sorted(items, key=lambda x: x['quality'], reverse=True)[:k]

def random_select(items, k, seed=42):
    return random.Random(seed).sample(items, k)

SELECTORS = {'qd': qd_select, 'greedy': greedy_select, 'random': random_select}

def train_and_eval(selected, strategy, rank, lora_seed):
    config_key = f"{strategy}_r{rank}_s{lora_seed}"

    # Format training data
    train_texts = []
    for s in selected:
        text = f"<|im_start|>system\nSolve the math problem step by step.<|im_end|>\n<|im_start|>user\n{s['question']}<|im_end|>\n<|im_start|>assistant\n{s['answer'][:768]}<|im_end|>"
        train_texts.append(text)

    if len(train_texts) < 10: return None

    random.seed(lora_seed); np.random.seed(lora_seed); torch.manual_seed(lora_seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=rank*2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LoRA params: {n_params/1e6:.2f}M (r={rank})", flush=True)

    class MathDataset(torch.utils.data.Dataset):
        def __init__(self, texts, tokenizer, max_len=768):
            self.texts = texts; self.tokenizer = tokenizer; self.max_len = max_len
        def __len__(self): return len(self.texts)
        def __getitem__(self, idx):
            enc = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_len,
                                 padding='max_length', return_tensors='pt')
            return {'input_ids': enc['input_ids'].squeeze(),
                    'attention_mask': enc['attention_mask'].squeeze(),
                    'labels': enc['input_ids'].squeeze()}

    dataset = MathDataset(train_texts, tokenizer, MAX_SEQ_LENGTH)
    ckpt_dir = RESULTS_DIR / f"ckpt_{config_key}"
    training_args = TrainingArguments(
        output_dir=str(ckpt_dir), num_train_epochs=3,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        learning_rate=2e-4, bf16=True, logging_steps=10,
        save_strategy="no", report_to="none", seed=lora_seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset,
                      data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False))
    trainer.train()

    merged = model.merge_and_unload()
    merged_path = str(RESULTS_DIR / f"merged_{config_key}")
    merged.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    del trainer, model, merged; torch.cuda.empty_cache()

    # Evaluate on GSM8K test (use 200 problems for speed)
    print(f"  Evaluating GSM8K for {config_key}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eval_model = AutoModelForCausalLM.from_pretrained(
        merged_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    eval_model.eval()

    correct = total = 0
    N_TEST = 200
    for i, ex in enumerate(test_ds):
        if i >= N_TEST: break
        test_ans = extract_answer(ex['answer'])
        if test_ans is None: continue

        msgs = [{"role": "system", "content": "Solve the math problem step by step."},
                {"role": "user", "content": ex['question']}]
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(txt, return_tensors="pt").to(eval_model.device)
        with torch.no_grad():
            out = eval_model.generate(**inp, max_new_tokens=256, temperature=0.0, do_sample=False,
                                       pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)

        nums = re.findall(r'-?[\d,]+\.?\d*', resp.replace(',', ''))
        pred = nums[-1] if nums else None

        ok = False
        if pred and test_ans:
            try:
                if abs(float(pred) - float(test_ans)) < 0.01:
                    ok = True
                    correct += 1
            except: pass
        total += 1

    acc = correct / total if total > 0 else 0
    print(f"  GSM8K: {acc:.4f} ({correct}/{total}) r={rank}", flush=True)

    result = {
        'strategy': strategy, 'k': K, 'lora_rank': rank, 'lora_seed': lora_seed,
        'n_train': len(selected), 'lora_params_M': round(n_params/1e6, 2),
        'gsm8k_accuracy': round(acc, 4), 'correct': correct, 'total': total,
    }

    del eval_model; torch.cuda.empty_cache()
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    shutil.rmtree(merged_path, ignore_errors=True)
    return result

# Split configs by rank groups
RANK_GROUPS = [[4, 8], [16, 32], [64]]
my_ranks = RANK_GROUPS[args.rank_group]

CONFIGS = []
for rank in my_ranks:
    for strategy in ['qd', 'greedy', 'random']:
        for seed in ALL_SEEDS:
            CONFIGS.append((strategy, rank, seed))

print(f"\n=== Math Rank Sweep (GPU {args.gpu}, group {args.rank_group}) ===", flush=True)
print(f"Ranks: {my_ranks}, Configs: {len(CONFIGS)}", flush=True)

results_file = RESULTS_DIR / f"results_group{args.rank_group}_gpu{args.gpu}.json"
all_results = {}
if results_file.exists():
    all_results = json.load(open(results_file))

for i, (strategy, rank, lora_seed) in enumerate(CONFIGS):
    config_key = f"{strategy}_r{rank}_s{lora_seed}"
    if config_key in all_results and 'error' not in all_results[config_key]:
        print(f"[{i+1}/{len(CONFIGS)}] SKIP {config_key}", flush=True)
        continue

    print(f"\n[{i+1}/{len(CONFIGS)}] {config_key}", flush=True)
    t0 = time.time()
    try:
        selected = SELECTORS[strategy](pool, K, seed=42)
        cells_n = len(set(get_cell(s['descriptors']) for s in selected))
        print(f"  Selected {len(selected)} items, {cells_n} cells", flush=True)
        result = train_and_eval(selected, strategy, rank, lora_seed)
        if result:
            result['elapsed'] = round(time.time() - t0, 1)
            result['cells'] = cells_n
            all_results[config_key] = result
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        import traceback; traceback.print_exc()
        all_results[config_key] = {'error': str(e)}

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

# Summary
print(f"\n{'='*60}", flush=True)
for rank in my_ranks:
    for strategy in ['qd', 'greedy', 'random']:
        accs = [all_results[k]['gsm8k_accuracy'] for k in all_results
                if all_results[k].get('lora_rank') == rank and all_results[k].get('strategy') == strategy
                and 'error' not in all_results[k]]
        if accs:
            print(f"r={rank:2d} {strategy:8s}: {np.mean(accs)*100:5.1f} +/- {np.std(accs)*100:5.1f}%", flush=True)

print(f"\nSaved to {results_file}", flush=True)
