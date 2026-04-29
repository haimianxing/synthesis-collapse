"""
Exp A: LoRA Rank Sweep — Prove Detectability Transition (W2+W3)

Vary LoRA rank [4, 8, 16, 32, 64] at k=500 with 7B model, 3 methods, 5 seeds.
Predict: at r=64, QD should beat Random (noise decreases → coverage becomes detectable).
This creates a clean transition curve predicted by the detectability threshold.

Usage:
  python exp_rank_sweep.py --gpu 1 --rank-group 0   # ranks [4,8]
  python exp_rank_sweep.py --gpu 2 --rank-group 1   # ranks [16,32]
  python exp_rank_sweep.py --gpu 3 --rank-group 2   # rank [64]
"""
import os
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import sys, json, torch, numpy as np, subprocess, shutil, random, argparse, time
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-7B-Instruct"
RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/scale_v10")
OUT_DIR = RESULTS_DIR / "rank_sweep"
OUT_DIR.mkdir(exist_ok=True)
MAX_SEQ_LENGTH = 512
MAX_CODE_TOKENS = 512

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, required=True)
parser.add_argument('--rank-group', type=int, required=True)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

from datasets import load_dataset
humaneval = load_dataset('openai/openai_humaneval', split='test')

# Load pool
pool = json.load(open(RESULTS_DIR / "merged_pool.json"))
for s in pool:
    if isinstance(s['cell'], list):
        s['cell'] = tuple(s['cell'])

ALL_SEEDS = [42, 123, 456, 789, 2024]
ALL_RANKS = [4, 8, 16, 32, 64]

def qd_select(solutions, k, seed=42):
    cells = {}
    for s in solutions:
        key = str(s['cell'])
        if key not in cells: cells[key] = []
        cells[key].append(s)
    for key in cells:
        cells[key].sort(key=lambda x: x.get('quality', 0), reverse=True)
    selected = [cells[key][0] for key in cells]
    remaining = k - len(selected)
    if remaining > 0:
        fill = []
        for key in cells:
            for s in cells[key][1:]:
                fill.append(s)
        fill.sort(key=lambda x: x.get('quality', 0), reverse=True)
        selected.extend(fill[:remaining])
    return selected[:k]

def greedy_select(solutions, k, seed=42):
    return sorted(solutions, key=lambda x: x.get('quality', 0), reverse=True)[:k]

def random_select(solutions, k, seed=42):
    return random.Random(seed).sample(solutions, min(k, len(solutions)))

SELECTORS = {'qd': qd_select, 'greedy': greedy_select, 'random': random_select}

def execute_code_safely(code_str, test_cases, timeout=5):
    for test in test_cases:
        try:
            result = subprocess.run(['python3', '-c', code_str + "\n" + test],
                                    timeout=timeout, capture_output=True, text=True)
            if result.returncode == 0: return True
        except: pass
    return False

def train_and_eval(selected, strategy, rank, lora_seed):
    config_key = f"{strategy}_r{rank}_s{lora_seed}"
    train_texts = [f"### Problem:\n{s['prompt']}\n\n### Solution:\n{s['code']}" for s in selected]
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

    class CodeDataset(torch.utils.data.Dataset):
        def __init__(self, texts, tokenizer, max_len=512):
            self.texts = texts; self.tokenizer = tokenizer; self.max_len = max_len
        def __len__(self): return len(self.texts)
        def __getitem__(self, idx):
            enc = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_len,
                                 padding='max_length', return_tensors='pt')
            return {'input_ids': enc['input_ids'].squeeze(),
                    'attention_mask': enc['attention_mask'].squeeze(),
                    'labels': enc['input_ids'].squeeze()}

    dataset = CodeDataset(train_texts, tokenizer, MAX_SEQ_LENGTH)
    ckpt_dir = OUT_DIR / f"ckpt_{config_key}"
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
    merged_path = str(OUT_DIR / f"merged_{config_key}")
    merged.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    del trainer, model, merged; torch.cuda.empty_cache()

    # Evaluate HumanEval
    print(f"  Evaluating HumanEval for {config_key}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True)
    eval_model = AutoModelForCausalLM.from_pretrained(
        merged_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    eval_model.eval()

    he_correct = 0; he_total = 0
    for item in humaneval:
        prompt = item['prompt']
        test_code = item['test']
        msgs = [{"role": "system", "content": "Complete the Python function."},
                {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=MAX_SEQ_LENGTH).to(eval_model.device)
        with torch.no_grad():
            outputs = eval_model.generate(**inputs, max_new_tokens=MAX_CODE_TOKENS,
                                           do_sample=False, pad_token_id=tokenizer.eos_token_id)
        completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        full_code = prompt + completion
        if execute_code_safely(full_code, [test_code]):
            he_correct += 1
        he_total += 1

    he_acc = he_correct / he_total if he_total > 0 else 0
    print(f"  HE: {he_acc:.4f} ({he_correct}/{he_total}) r={rank}", flush=True)

    result = {
        'strategy': strategy, 'k': 500, 'lora_rank': rank, 'lora_seed': lora_seed,
        'n_train': len(selected), 'lora_params_M': round(n_params/1e6, 2),
        'humaneval_pass1': round(he_acc, 4), 'he_correct': he_correct, 'he_total': he_total,
    }

    del eval_model; torch.cuda.empty_cache()
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    shutil.rmtree(merged_path, ignore_errors=True)
    return result

# Split configs by rank groups
# Group 0: ranks [4, 8] × 3 methods × 5 seeds = 30 configs
# Group 1: ranks [16, 32] × 3 methods × 5 seeds = 30 configs
# Group 2: rank [64] × 3 methods × 5 seeds = 15 configs
RANK_GROUPS = [[4, 8], [16, 32], [64]]
my_ranks = RANK_GROUPS[args.rank_group]

CONFIGS = []
for rank in my_ranks:
    for strategy in ['qd', 'greedy', 'random']:
        for seed in ALL_SEEDS:
            CONFIGS.append((strategy, rank, seed))

print(f"=== LoRA Rank Sweep (GPU {args.gpu}, group {args.rank_group}) ===", flush=True)
print(f"Ranks: {my_ranks}, Configs: {len(CONFIGS)}", flush=True)

results_file = OUT_DIR / f"results_group{args.rank_group}_gpu{args.gpu}.json"
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
        selected = SELECTORS[strategy](pool, 500, seed=42)
        result = train_and_eval(selected, strategy, rank, lora_seed)
        if result:
            result['elapsed'] = round(time.time() - t0, 1)
            all_results[config_key] = result
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        all_results[config_key] = {'error': str(e)}

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

# Summary
print(f"\n{'='*60}", flush=True)
for rank in my_ranks:
    for strategy in ['qd', 'greedy', 'random']:
        accs = [all_results[k]['humaneval_pass1'] for k in all_results
                if all_results[k].get('lora_rank') == rank and all_results[k].get('strategy') == strategy
                and 'error' not in all_results[k]]
        if accs:
            print(f"r={rank:2d} {strategy:8s}: {np.mean(accs)*100:5.1f} ± {np.std(accs)*100:5.1f}%", flush=True)

print(f"\nSaved to {results_file}", flush=True)
