"""V10 Correct-Only Pool Experiment.
Filters to only correct solutions (300 total, 35 cells).
Runs QD/Greedy/Random selection at k=50,100,200,300.
Trains LoRA and evaluates HumanEval+MBPP.

Usage:
  python v10_correct_only.py --gpu 1 --strategy qd
"""
import os
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import sys, json, re, torch, numpy as np, subprocess, time, shutil, random, argparse
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-7B-Instruct"
RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/scale_v10")
OUT_DIR = RESULTS_DIR / "correct_only"
OUT_DIR.mkdir(exist_ok=True)
MAX_SEQ_LENGTH = 512
MAX_CODE_TOKENS = 512

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, required=True)
parser.add_argument('--strategy', type=str, required=True, choices=['qd', 'greedy', 'random'])
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

from datasets import load_dataset
humaneval = load_dataset('openai/openai_humaneval', split='test')
mbpp_test = load_dataset('mbpp', 'sanitized', split='test')

# Load pool and filter to correct only
pool = json.load(open(RESULTS_DIR / "merged_pool.json"))
for s in pool:
    if isinstance(s['cell'], list):
        s['cell'] = tuple(s['cell'])

correct_pool = [s for s in pool if s.get('correct', False)]
print(f"Correct-only pool: {len(correct_pool)} solutions, "
      f"{len(set(str(s['cell']) for s in correct_pool))} cells", flush=True)

def execute_code_safely(code_str, test_cases, timeout=5):
    passed = 0
    for test in test_cases:
        try:
            full_code = code_str + "\n" + test
            result = subprocess.run(['python3', '-c', full_code], timeout=timeout,
                                    capture_output=True, text=True)
            if result.returncode == 0: passed += 1
        except: pass
    return passed, len(test_cases)

def qd_select(solutions, k, seed=42):
    """MAP-Elites: best solution per cell, then fill remaining."""
    rng = random.Random(seed)
    # Group by cell
    cells = {}
    for s in solutions:
        key = str(s['cell'])
        if key not in cells:
            cells[key] = []
        cells[key].append(s)

    # Sort each cell by quality (descending)
    for key in cells:
        cells[key].sort(key=lambda x: x.get('quality', 0), reverse=True)

    # Select best from each cell (archive initialization)
    selected = []
    for key in cells:
        selected.append(cells[key][0])

    # Fill remaining with best from cells with multiple solutions
    remaining = k - len(selected)
    if remaining > 0:
        fill_candidates = []
        for key in cells:
            for s in cells[key][1:]:
                fill_candidates.append(s)
        fill_candidates.sort(key=lambda x: x.get('quality', 0), reverse=True)
        selected.extend(fill_candidates[:remaining])

    return selected[:k]

def greedy_select(solutions, k, seed=42):
    """Greedy: top-k by quality."""
    sorted_sols = sorted(solutions, key=lambda x: x.get('quality', 0), reverse=True)
    return sorted_sols[:k]

def random_select(solutions, k, seed=42):
    """Random: uniformly sample k solutions."""
    rng = random.Random(seed)
    return rng.sample(solutions, min(k, len(solutions)))

SELECTORS = {
    'qd': qd_select,
    'greedy': greedy_select,
    'random': random_select,
}

def train_and_eval(selected, config_key, k_target):
    """Train LoRA and evaluate."""
    k_val = len(selected)
    train_texts = []
    for s in selected:
        code = s['code']
        text = f"### Problem:\n{s['prompt']}\n\n### Solution:\n{code}"
        train_texts.append(text)

    if len(train_texts) < 10:
        print(f"  SKIP {config_key}: only {len(train_texts)} train texts", flush=True)
        return None

    # Train LoRA
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
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
        save_strategy="no", report_to="none",
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
    print(f"  Evaluating HumanEval...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True)
    eval_model = AutoModelForCausalLM.from_pretrained(
        merged_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )
    eval_model.eval()

    he_correct = 0; he_total = 0
    for idx, item in enumerate(humaneval):
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
        passed, n_tests = execute_code_safely(full_code, [test_code])
        if passed > 0: he_correct += 1
        he_total += 1

    he_acc = he_correct / he_total if he_total > 0 else 0
    print(f"  HumanEval: {he_acc:.4f} ({he_correct}/{he_total})", flush=True)

    # Evaluate MBPP
    print(f"  Evaluating MBPP...", flush=True)
    mbpp_correct = 0; mbpp_total = 0
    for idx, item in enumerate(mbpp_test):
        prompt = item['prompt']
        test_list = item['test_list']
        msgs = [{"role": "system", "content": "Write a Python function to solve the problem."},
                {"role": "user", "content": f"Write a Python function:\n\n{prompt}\n\nProvide the implementation:"}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=MAX_SEQ_LENGTH).to(eval_model.device)
        with torch.no_grad():
            outputs = eval_model.generate(**inputs, max_new_tokens=MAX_CODE_TOKENS,
                                           do_sample=False, pad_token_id=tokenizer.eos_token_id)
        completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        code_blocks = re.findall(r'```python\s*(.*?)```', completion, re.DOTALL)
        code = code_blocks[0] if code_blocks else completion
        passed, n_tests = execute_code_safely(code, test_list)
        if passed == n_tests: mbpp_correct += 1
        mbpp_total += 1

    mbpp_acc = mbpp_correct / mbpp_total if mbpp_total > 0 else 0
    print(f"  MBPP: {mbpp_acc:.4f} ({mbpp_correct}/{mbpp_total})", flush=True)

    result = {
        'strategy': args.strategy, 'k': k_target, 'seed': SEED,
        'n_train': len(selected),
        'n_cells': len(set(str(s['cell']) for s in selected)),
        'entropy': compute_entropy(selected),
        'humaneval_pass1': he_acc, 'he_correct': he_correct, 'he_total': he_total,
        'mbpp_acc': mbpp_acc, 'mbpp_correct': mbpp_correct, 'mbpp_total': mbpp_total,
        'pool': 'correct_only', 'pool_size': len(correct_pool),
    }

    # Cleanup
    del eval_model; torch.cuda.empty_cache()
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    shutil.rmtree(merged_path, ignore_errors=True)

    return result

def compute_entropy(selected):
    cells = Counter(str(s['cell']) for s in selected)
    total = len(selected)
    probs = [c/total for c in cells.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)

# Run for all k values
all_results = {}
selector = SELECTORS[args.strategy]

for k_val in [50, 100, 200, 300]:
    config_key = f"{args.strategy}_k{k_val}_s{SEED}"
    print(f"\n=== {config_key} ===", flush=True)

    if k_val > len(correct_pool):
        print(f"  SKIP: k={k_val} > pool size {len(correct_pool)}", flush=True)
        continue

    selected = selector(correct_pool, k_val, seed=SEED)
    cells = len(set(str(s['cell']) for s in selected))
    print(f"  Selected {len(selected)} solutions across {cells} cells", flush=True)

    result = train_and_eval(selected, config_key, k_val)
    if result:
        all_results[config_key] = result
        # Save intermediate
        with open(OUT_DIR / f"results_{args.strategy}_gpu{args.gpu}.json", 'w') as f:
            json.dump(all_results, f, indent=2)

print(f"\n=== DONE: {len(all_results)} configs evaluated ===", flush=True)
with open(OUT_DIR / f"results_{args.strategy}_gpu{args.gpu}.json", 'w') as f:
    json.dump(all_results, f, indent=2)
