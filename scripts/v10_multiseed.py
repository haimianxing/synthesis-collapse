"""V10 Multi-Seed Experiment for Table 8 Error Bars.
Runs 5 LoRA training seeds per (strategy, k) config.
Only evaluates HumanEval (fast) to get error bars.

Usage:
  python v10_multiseed.py --gpu 4 --seed-group 0
  python v10_multiseed.py --gpu 5 --seed-group 1
  ...

Seed groups (7 groups for 7 GPUs):
  Group 0: qd_k500 seeds 42,123,456 | qd_k1000 seeds 42,123
  Group 1: qd_k1000 seeds 456,789,1024 | qd_k2000 seeds 42,123
  Group 2: qd_k2000 seeds 456,789,1024 | greedy_k500 seeds 42
  Group 3: greedy_k500 seeds 123,456,789,1024 | greedy_k1000 seed 42
  Group 4: greedy_k1000 seeds 123,456,789,1024
  Group 5: random_k500 seeds 42,123,456,789,1024
  Group 6: random_k1000+random_k2000 (5 seeds each, 10 total)
"""
import os
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import sys, json, re, torch, numpy as np, subprocess, shutil, random, argparse
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-7B-Instruct"
RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/scale_v10")
OUT_DIR = RESULTS_DIR / "multiseed"
OUT_DIR.mkdir(exist_ok=True)
MAX_SEQ_LENGTH = 512
MAX_CODE_TOKENS = 512

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, required=True)
parser.add_argument('--seed-group', type=int, required=True)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

from datasets import load_dataset
humaneval = load_dataset('openai/openai_humaneval', split='test')

# Load pool
pool = json.load(open(RESULTS_DIR / "merged_pool.json"))
for s in pool:
    if isinstance(s['cell'], list):
        s['cell'] = tuple(s['cell'])

ALL_SEEDS = [42, 123, 456, 789, 1024]

def qd_select(solutions, k, seed=42):
    rng = random.Random(seed)
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
    passed = 0
    for test in test_cases:
        try:
            result = subprocess.run(['python3', '-c', code_str + "\n" + test],
                                    timeout=timeout, capture_output=True, text=True)
            if result.returncode == 0: passed += 1
        except: pass
    return passed, len(test_cases)

def train_and_eval_he(selected, strategy, k, lora_seed):
    """Train LoRA with specific seed and evaluate HumanEval only."""
    config_key = f"{strategy}_k{k}_s{lora_seed}"

    train_texts = [f"### Problem:\n{s['prompt']}\n\n### Solution:\n{s['code']}" for s in selected]
    if len(train_texts) < 10:
        return None

    random.seed(lora_seed)
    np.random.seed(lora_seed)
    torch.manual_seed(lora_seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

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

    # Evaluate HumanEval only (fast)
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
        passed, _ = execute_code_safely(full_code, [test_code])
        if passed > 0: he_correct += 1
        he_total += 1

    he_acc = he_correct / he_total if he_total > 0 else 0
    print(f"  HE: {he_acc:.4f} ({he_correct}/{he_total})", flush=True)

    result = {
        'strategy': strategy, 'k': k, 'lora_seed': lora_seed,
        'n_train': len(selected),
        'humaneval_pass1': he_acc, 'he_correct': he_correct, 'he_total': he_total,
    }

    del eval_model; torch.cuda.empty_cache()
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    shutil.rmtree(merged_path, ignore_errors=True)

    return result

# Define configs for each seed group
# 3 strategies × 3 k values × 5 seeds = 45 total
# Split into 7 groups for 7 GPUs
CONFIGS = []
for strategy in ['qd', 'greedy', 'random']:
    for k in [500, 1000, 2000]:
        for seed in ALL_SEEDS:
            CONFIGS.append((strategy, k, seed))

# 45 configs, split into 7 groups
GROUP_SIZE = len(CONFIGS) // 7
REMAINDER = len(CONFIGS) % 7
groups = []
idx = 0
for g in range(7):
    size = GROUP_SIZE + (1 if g < REMAINDER else 0)
    groups.append(CONFIGS[idx:idx+size])
    idx += size

my_configs = groups[args.seed_group]
print(f"GPU {args.gpu}, Group {args.seed_group}: {len(my_configs)} configs", flush=True)
for strategy, k, seed in my_configs:
    print(f"  {strategy}_k{k}_s{seed}", flush=True)

# Run configs
all_results = {}
results_file = OUT_DIR / f"results_group{args.seed_group}_gpu{args.gpu}.json"

for strategy, k, lora_seed in my_configs:
    config_key = f"{strategy}_k{k}_s{lora_seed}"
    print(f"\n=== {config_key} ===", flush=True)

    # Check if already done
    if results_file.exists():
        existing = json.load(open(results_file))
        if config_key in existing:
            print(f"  SKIP: already done", flush=True)
            all_results.update(existing)
            continue

    # Select data (deterministic for qd/greedy, seed-dependent for random)
    selected = SELECTORS[strategy](pool, k, seed=42)  # Selection seed is always 42
    print(f"  Selected {len(selected)} solutions", flush=True)

    result = train_and_eval_he(selected, strategy, k, lora_seed)
    if result:
        all_results[config_key] = result
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

print(f"\n=== DONE: {len(all_results)} configs evaluated ===", flush=True)
with open(results_file, 'w') as f:
    json.dump(all_results, f, indent=2)
