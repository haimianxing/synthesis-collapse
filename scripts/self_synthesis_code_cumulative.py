"""
Self-Synthesis v8: Code Domain — CUMULATIVE Training (Non-Base-Reset)
=====================================================================
Critical experiment to validate W2: Does QD's archive-guided generation
discover NEW cells that Greedy's narrower model cannot reach?

Key difference from v7 (base-reset):
  R0: base model → generate → QD/Greedy select → train → merged_r0
  R1: merged_r0 model → generate → select → train → merged_r1
  R2: merged_r1 model → generate → select → train → merged_r2
  R3: merged_r2 model → generate → select → train → merged_r3

Hypothesis:
  - Greedy's model narrows generation pool → fewer cells over rounds
  - QD's model maintains diverse generation → cells stay stable/grow
  - This is the "iterative generation advantage" that base-reset cannot test

Usage:
  CUDA_VISIBLE_DEVICES=1 nohup python -u self_synthesis_code_cumulative.py --strategy qd --seed 42 &
  CUDA_VISIBLE_DEVICES=2 nohup python -u self_synthesis_code_cumulative.py --strategy greedy --seed 42 &
  CUDA_VISIBLE_DEVICES=3 nohup python -u self_synthesis_code_cumulative.py --strategy qd --seed 123 &
  CUDA_VISIBLE_DEVICES=4 nohup python -u self_synthesis_code_cumulative.py --strategy greedy --seed 123 &
"""
import os, sys, json, random, re, torch, numpy as np, subprocess, time, argparse
from pathlib import Path
from collections import Counter, defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

# Force offline mode
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# === Parse Args ===
parser = argparse.ArgumentParser()
parser.add_argument('--strategy', type=str, required=True, choices=['qd', 'greedy'])
parser.add_argument('--seed', type=int, required=True)
args = parser.parse_args()

STRATEGY = args.strategy
SEED = args.seed
MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-7B-Instruct"

# Generation params
N_GENERATE_PER_PROBLEM = 5
N_SELECT = 120
N_ROUNDS = 4
GRID_RES = 5  # 5x5x5=125 cells

# Training params
LORA_R = 16
LORA_ALPHA = 32
TRAIN_EPOCHS = 3
LR = 2e-4
MAX_SEQ_LENGTH = 512
MAX_CODE_TOKENS = 512

# Eval
N_EVAL_HUMANEVAL = 164

RESULTS_DIR = Path(f"/mnt/data2/zcz/neurIps-emnlp/neurips/results/self_synthesis_v8_cumulative/{STRATEGY}_s{SEED}")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"=== Self-Synthesis v8 (Code CUMULATIVE): {STRATEGY.upper()} seed={SEED} ===", flush=True)
print(f"  KEY DIFFERENCE: Each round generates from PREVIOUS round's merged model", flush=True)
print(f"  N_GEN={N_GENERATE_PER_PROBLEM}, N_SEL={N_SELECT}, N_ROUNDS={N_ROUNDS}", flush=True)
print(f"  MODEL={MODEL_PATH}", flush=True)
print(f"  RESULTS_DIR={RESULTS_DIR}", flush=True)

torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

# === Load Datasets ===
from datasets import load_dataset

mbpp_train = load_dataset('mbpp', 'sanitized', split='train')
mbpp_test = load_dataset('mbpp', 'sanitized', split='test')
humaneval = load_dataset('openai/openai_humaneval', split='test')

prompt_pool = []
for item in mbpp_train:
    prompt_pool.append({
        'task_id': item['task_id'],
        'prompt': item['prompt'],
        'code': item['code'],
        'test_list': item['test_list'],
        'test_imports': item.get('test_imports', []),
    })
for item in mbpp_test:
    prompt_pool.append({
        'task_id': item['task_id'],
        'prompt': item['prompt'],
        'code': item['code'],
        'test_list': item['test_list'],
        'test_imports': item.get('test_imports', []),
    })

print(f"  MBPP prompt pool: {len(prompt_pool)} problems", flush=True)
print(f"  HumanEval: {len(humaneval)} problems for evaluation", flush=True)

# === Code Descriptor (3D) — same as v7 ===
def get_code_descriptor(prompt, code=None):
    text = (prompt + " " + (code or "")).lower()
    loops = len(re.findall(r'\bfor\b|\bwhile\b', text))
    conds = len(re.findall(r'\bif\b|\belif\b', text))
    funcs = len(re.findall(r'\bdef\b', text))
    comprehensions = len(re.findall(r'\[.*for.*in.*\]', text))
    lines = len(text.split('\n'))
    complexity_score = loops * 2 + conds * 1 + funcs * 1 + comprehensions * 1.5 + lines * 0.1
    dim0 = min(int(complexity_score / 4), 4)
    if re.search(r'sort|order|rank|ascending|descending', text): dim1 = 0
    elif re.search(r'search|find|index|contains|exists', text): dim1 = 1
    elif re.search(r'sum|average|mean|max|min|count|total|product', text): dim1 = 2
    elif re.search(r'replace|split|join|strip|uppercase|lowercase|reverse|palindrome', text): dim1 = 3
    else: dim1 = 4
    if re.search(r'return.*\[|list|append|extend|pop|remove', text): dim2 = 0
    elif re.search(r'return.*true|return.*false|bool|is_|check|verify', text): dim2 = 1
    elif re.search(r'return.*\d|int|float|number|count|length|size', text): dim2 = 2
    elif re.search(r'return.*["\']|str|string|char', text): dim2 = 3
    else: dim2 = 4
    return (dim0, dim1, dim2)

# === Code Execution ===
def execute_code_safely(code_str, test_cases, timeout=5):
    passed = 0
    total = len(test_cases)
    for test in test_cases:
        try:
            full_code = code_str + "\n" + test
            result = subprocess.run(
                ['python3', '-c', full_code],
                timeout=timeout, capture_output=True, text=True
            )
            if result.returncode == 0:
                passed += 1
        except (subprocess.TimeoutExpired, Exception):
            pass
    return passed, total

def check_code_quality(prompt_text, generated_code, reference_code=None, test_list=None):
    descriptor = get_code_descriptor(prompt_text, generated_code)
    if not generated_code or len(generated_code.strip()) < 10:
        return 0.0, False, descriptor
    try:
        compile(generated_code, '<string>', 'exec')
    except SyntaxError:
        return 0.1, False, descriptor
    if test_list:
        passed, total = execute_code_safely(generated_code, test_list)
        quality = 0.3 + 0.7 * (passed / total) if total > 0 else 0.3
        return quality, passed == total, descriptor
    has_def = 'def ' in generated_code
    has_return = 'return ' in generated_code
    quality = 0.2 + (0.2 if has_def else 0) + (0.2 if has_return else 0)
    return quality, False, descriptor

# === Selection Strategies ===
def select_qd_code(solutions, n_select):
    archive = {}
    for sol in solutions:
        cell = sol['cell']
        if cell not in archive or sol['quality'] > archive[cell]['quality']:
            archive[cell] = sol
    selected = list(archive.values())
    if len(selected) < n_select:
        remaining = sorted(
            [s for s in solutions if s not in selected],
            key=lambda x: x['quality'], reverse=True
        )
        selected.extend(remaining[:n_select - len(selected)])
    elif len(selected) > n_select:
        selected = sorted(selected, key=lambda x: x['quality'], reverse=True)[:n_select]
    return selected[:n_select]

def select_greedy_code(solutions, n_select):
    return sorted(solutions, key=lambda x: x['quality'], reverse=True)[:n_select]

def shannon_entropy(items):
    counts = Counter(items)
    total = sum(counts.values())
    if total == 0: return 0.0
    return -sum((c/total) * np.log2(c/total) for c in counts.values() if c > 0)

# === Generation ===
def generate_code_solutions(model, tokenizer, problems, n_per_problem):
    solutions = []
    for idx, prob in enumerate(problems):
        prompt = f"Write a Python function to solve the following problem:\n\n{prob['prompt']}\n\nProvide the complete function implementation:"
        msgs = [
            {"role": "system", "content": "You are a Python programmer. Write clean, efficient code."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to(model.device)

        for attempt in range(n_per_problem):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_CODE_TOKENS,
                    do_sample=True if attempt > 0 else False,
                    temperature=0.7 if attempt > 0 else 0.0,
                    top_p=0.9 if attempt > 0 else 1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            code_blocks = re.findall(r'```python\s*(.*?)```', response, re.DOTALL)
            if code_blocks:
                code = code_blocks[0]
            elif 'def ' in response:
                match = re.search(r'(def\s+\w+.*?)(?=\n\S|\Z)', response, re.DOTALL)
                code = match.group(1) if match else response
            else:
                code = response
            quality, is_correct, desc = check_code_quality(
                prob['prompt'], code, prob.get('code'), prob.get('test_list')
            )
            solutions.append({
                'task_id': prob['task_id'],
                'prompt': prob['prompt'],
                'code': code.strip(),
                'quality': quality,
                'correct': is_correct,
                'cell': desc,
                'attempt': attempt,
            })

        if (idx + 1) % 20 == 0:
            cells_so_far = len(set(s['cell'] for s in solutions))
            correct_so_far = sum(1 for s in solutions if s['correct'])
            print(f"    Gen {idx+1}/{len(problems)} ({len(solutions)} sol, {correct_so_far} ok, {cells_so_far} cells)", flush=True)
    return solutions

# === Training ===
def train_model(base_model_path, train_data, rnd_key):
    """LoRA fine-tuning. Trains from base_model_path (which may be a merged checkpoint)."""
    print(f"  Training on {len(train_data)} code samples from {base_model_path}...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )

    train_texts = []
    for item in train_data:
        text = f"### Problem:\n{item['prompt']}\n\n### Solution:\n{item['code']}"
        train_texts.append(text)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

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
    training_args = TrainingArguments(
        output_dir=str(RESULTS_DIR / f"ckpt_{rnd_key}"),
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=LR, bf16=True,
        logging_steps=10, save_strategy="no", report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset,
                      data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False))
    trainer.train()

    # Merge LoRA into model weights
    merged_model = model.merge_and_unload()
    merged_path = str(RESULTS_DIR / f"merged_r{rnd_key}")
    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)

    del trainer, model, merged_model
    torch.cuda.empty_cache()

    return merged_path

# === Evaluation on HumanEval ===
def evaluate_humaneval(merged_path, n_eval=164):
    print(f"  Evaluating HumanEval ({n_eval})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        merged_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )
    model.eval()

    correct = 0; total = 0
    for idx, item in enumerate(humaneval):
        if idx >= n_eval: break
        prompt = item['prompt']
        msgs = [
            {"role": "system", "content": "Complete the Python function. Write only the function body."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=MAX_CODE_TOKENS,
                                     do_sample=False, pad_token_id=tokenizer.eos_token_id)
        completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        full_code = prompt + completion
        test_code = item['test']
        passed, n_tests = execute_code_safely(full_code, [test_code])
        if passed > 0: correct += 1
        total += 1
        if (idx + 1) % 20 == 0:
            print(f"    Eval {idx+1}/{total}, pass@1={correct/total:.4f}", flush=True)

    del model; torch.cuda.empty_cache()
    return correct / total if total > 0 else 0, correct, total

# === Main Loop — CUMULATIVE ===
results_file = RESULTS_DIR / f"{STRATEGY}_s{SEED}_v8.json"
all_results = {}
accumulated_data = []
current_model_path = MODEL_PATH  # Start from base; will be updated each round

for rnd in range(N_ROUNDS):
    t0 = time.time()
    rnd_key = f"r{rnd}"

    print(f"\n{'='*40}", flush=True)
    print(f"  ROUND {rnd} ({STRATEGY} seed={SEED}) [CUMULATIVE]", flush=True)
    print(f"  Generating from: {current_model_path}", flush=True)
    print(f"{'='*40}", flush=True)

    # === KEY: Generate from CURRENT model (not base) ===
    gen_tokenizer = AutoTokenizer.from_pretrained(current_model_path, trust_remote_code=True)
    gen_model = AutoModelForCausalLM.from_pretrained(
        current_model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )
    gen_model.eval()

    # Sample problems (same across rounds for fair comparison)
    rng = random.Random(SEED + rnd)
    sampled = rng.sample(prompt_pool, min(len(prompt_pool), len(prompt_pool)))

    solutions = generate_code_solutions(gen_model, gen_tokenizer, sampled, N_GENERATE_PER_PROBLEM)

    # Generation stats
    n_gen = len(solutions)
    n_correct = sum(1 for s in solutions if s['correct'])
    gen_cells = len(set(s['cell'] for s in solutions))
    gen_entropy = shannon_entropy([s['cell'] for s in solutions])
    gen_qual = np.mean([s['quality'] for s in solutions])

    print(f"  Gen (from {'base' if rnd == 0 else 'merged_r'+str(rnd-1)}): "
          f"{n_gen} sol, {n_correct} correct, {gen_cells} cells, H={gen_entropy:.2f}, q̄={gen_qual:.3f}", flush=True)

    del gen_model; torch.cuda.empty_cache()

    # Select
    if STRATEGY == "greedy":
        selected = select_greedy_code(solutions, N_SELECT)
    elif STRATEGY == "qd":
        selected = select_qd_code(solutions, N_SELECT)

    sel_cells = len(set(s['cell'] for s in selected))
    sel_correct = sum(1 for s in selected if s['correct'])
    sel_entropy = shannon_entropy([s['cell'] for s in selected])
    sel_qual = np.mean([s['quality'] for s in selected])

    print(f"  Sel: {len(selected)}, {sel_cells} cells, {sel_correct} correct, H={sel_entropy:.2f}, q̄={sel_qual:.3f}", flush=True)

    # Accumulate
    accumulated_data.extend(selected)

    # === KEY: Train from CURRENT model (cumulative LoRA) ===
    if len(accumulated_data) >= 10:
        # Train from the CURRENT model path (base for R0, merged for R1+)
        merged_path = train_model(current_model_path, accumulated_data, rnd)
        acc, correct, total = evaluate_humaneval(merged_path, N_EVAL_HUMANEVAL)

        # Update current model path for next round's generation
        current_model_path = merged_path
    else:
        acc, correct, total = 0, 0, 0

    elapsed = time.time() - t0
    print(f"  R{rnd}: pass@1={acc:.4f} ({correct}/{total}), accumulated={len(accumulated_data)}, {elapsed:.1f}s", flush=True)

    # Save round results
    round_result = {
        'round': rnd, 'seed': SEED, 'strategy': STRATEGY,
        'mode': 'CUMULATIVE',  # Mark this as cumulative
        'gen_model': 'base' if rnd == 0 else f'merged_r{rnd-1}',
        'n_accumulated_samples': len(accumulated_data),
        'n_generated': n_gen, 'n_correct_gen': n_correct,
        'n_cells_generated': gen_cells, 'gen_entropy': gen_entropy,
        'avg_quality_gen': gen_qual,
        'n_selected': len(selected), 'n_cells_selected': sel_cells,
        'n_correct_selected': sel_correct,
        'sel_entropy': sel_entropy, 'sel_avg_quality': sel_qual,
        'accuracy': acc, 'correct': correct, 'total': total,
        'elapsed_sec': elapsed, 'status': 'complete',
        'selected_data': [
            {'prompt': s['prompt'][:200], 'cell': s['cell'],
             'quality': s['quality'], 'correct': s['correct']}
            for s in selected
        ],
    }
    all_results[rnd_key] = round_result

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

# === Final Summary ===
print(f"\n{'='*60}", flush=True)
print(f"FINAL RESULTS: {STRATEGY.upper()} seed={SEED} (CUMULATIVE)", flush=True)
print(f"{'='*60}", flush=True)
for rnd_key, rnd_data in all_results.items():
    print(f"  {rnd_key}: pass@1={rnd_data['accuracy']:.4f} "
          f"gen_cells={rnd_data['n_cells_generated']} sel_cells={rnd_data['n_cells_selected']} "
          f"H={rnd_data['sel_entropy']:.3f}", flush=True)

print(f"\nResults saved to {results_file}", flush=True)
