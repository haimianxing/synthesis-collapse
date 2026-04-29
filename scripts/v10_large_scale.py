"""
V10: Large-Scale Code Synthesis (10K+ solutions)
=================================================
Addresses SAC W1 (scale): Prove collapse persists at 10K+ scale.

Design:
  Phase 1: Generate 30 solutions per MBPP problem (11,310 total) from base 7B
  Phase 2: Select at k=200, 500, 1000, 2000, 5000 using QD/Dedup, Greedy, Random
  Phase 3: Train LoRA for each (k, strategy), evaluate on HumanEval + MBPP
  Phase 4: Report coverage curves and downstream results

Usage:
  # Phase 1: Generation (split across GPUs 1-4)
  CUDA_VISIBLE_DEVICES=1 python3.9 -u v10_large_scale.py --phase gen --gen-start 0 --gen-end 95 &
  CUDA_VISIBLE_DEVICES=2 python3.9 -u v10_large_scale.py --phase gen --gen-start 95 --gen-end 190 &
  CUDA_VISIBLE_DEVICES=3 python3.9 -u v10_large_scale.py --phase gen --gen-start 190 --gen-end 285 &
  CUDA_VISIBLE_DEVICES=4 python3.9 -u v10_large_scale.py --phase gen --gen-start 285 --gen-end 377 &

  # Phase 2+3+4: After generation completes
  CUDA_VISIBLE_DEVICES=5 python3.9 -u v10_large_scale.py --phase eval
"""
import os, sys, json, random, re, torch, numpy as np, subprocess, time, argparse, shutil
from pathlib import Path
from collections import Counter, defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-7B-Instruct"
RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/scale_v10")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEQ_LENGTH = 512
MAX_CODE_TOKENS = 512
N_SOLUTIONS_PER_PROBLEM = 12  # 377 × 12 = 4,524 total (pool with V9 = 7.5K)

# === Parse Args ===
parser = argparse.ArgumentParser()
parser.add_argument('--phase', type=str, required=True, choices=['gen', 'eval'])
parser.add_argument('--gen-start', type=int, default=0)
parser.add_argument('--gen-end', type=int, default=377)
args = parser.parse_args()

# === Load Datasets ===
from datasets import load_dataset

mbpp_train = load_dataset('mbpp', 'sanitized', split='train')
mbpp_test = load_dataset('mbpp', 'sanitized', split='test')
humaneval = load_dataset('openai/openai_humaneval', split='test')

prompt_pool = []
for item in mbpp_train:
    prompt_pool.append({
        'task_id': item['task_id'], 'prompt': item['prompt'],
        'code': item['code'], 'test_list': item['test_list'],
        'test_imports': item.get('test_imports', []),
    })
for item in mbpp_test:
    prompt_pool.append({
        'task_id': item['task_id'], 'prompt': item['prompt'],
        'code': item['code'], 'test_list': item['test_list'],
        'test_imports': item.get('test_imports', []),
    })

print(f"=== V10 Large-Scale Code Synthesis ===", flush=True)
print(f"  MBPP prompt pool: {len(prompt_pool)} problems", flush=True)
print(f"  HumanEval: {len(humaneval)} problems", flush=True)
print(f"  Phase: {args.phase}", flush=True)

# === Code Descriptor (same as V7/V8/V9) ===
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
    for test in test_cases:
        try:
            full_code = code_str + "\n" + test
            result = subprocess.run(['python3', '-c', full_code], timeout=timeout,
                                    capture_output=True, text=True)
            if result.returncode == 0: passed += 1
        except: pass
    return passed, len(test_cases)

def check_code_quality(prompt_text, generated_code, test_list=None):
    descriptor = get_code_descriptor(prompt_text, generated_code)
    if not generated_code or len(generated_code.strip()) < 10:
        return 0.0, False, descriptor
    try: compile(generated_code, '<string>', 'exec')
    except SyntaxError: return 0.1, False, descriptor
    if test_list:
        passed, total = execute_code_safely(generated_code, test_list)
        quality = 0.3 + 0.7 * (passed / total) if total > 0 else 0.3
        return quality, passed == total, descriptor
    has_def = 'def ' in generated_code
    has_return = 'return ' in generated_code
    quality = 0.2 + (0.2 if has_def else 0) + (0.2 if has_return else 0)
    return quality, False, descriptor

# === Selection Strategies ===
def select_qd(solutions, n_select):
    archive = {}
    for sol in solutions:
        cell = sol['cell']
        if cell not in archive or sol['quality'] > archive[cell]['quality']:
            archive[cell] = sol
    selected = list(archive.values())
    if len(selected) < n_select:
        remaining = sorted([s for s in solutions if s not in selected],
                           key=lambda x: x['quality'], reverse=True)
        selected.extend(remaining[:n_select - len(selected)])
    elif len(selected) > n_select:
        selected = sorted(selected, key=lambda x: x['quality'], reverse=True)[:n_select]
    return selected[:n_select]

def select_greedy(solutions, n_select):
    return sorted(solutions, key=lambda x: x['quality'], reverse=True)[:n_select]

def select_random(solutions, n_select, seed=42):
    rng = random.Random(seed)
    return rng.sample(solutions, min(n_select, len(solutions)))

# === Phase 1: Generation ===
if args.phase == 'gen':
    print(f"\n--- Phase 1: Generation (problems {args.gen_start} to {args.gen_end-1}) ---", flush=True)
    gen_file = RESULTS_DIR / f"gen_pool_{args.gen_start}_{args.gen_end}.json"

    # Check if already done
    if gen_file.exists():
        existing = json.load(open(gen_file))
        print(f"  Already generated: {len(existing)} solutions. Skipping.", flush=True)
        sys.exit(0)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )
    model.eval()

    t0 = time.time()
    all_solutions = []
    problems = prompt_pool[args.gen_start:args.gen_end]

    for idx, prob in enumerate(problems):
        prompt = f"Write a Python function to solve the following problem:\n\n{prob['prompt']}\n\nProvide the complete function implementation:"
        msgs = [
            {"role": "system", "content": "You are a Python programmer. Write clean, efficient code."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to(model.device)

        for attempt in range(N_SOLUTIONS_PER_PROBLEM):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=MAX_CODE_TOKENS,
                    do_sample=True if attempt > 0 else False,
                    temperature=0.7 if attempt > 0 else 0.0,
                    top_p=0.9 if attempt > 0 else 1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            code_blocks = re.findall(r'```python\s*(.*?)```', response, re.DOTALL)
            if code_blocks: code = code_blocks[0]
            elif 'def ' in response:
                match = re.search(r'(def\s+\w+.*?)(?=\n\S|\Z)', response, re.DOTALL)
                code = match.group(1) if match else response
            else: code = response

            quality, is_correct, desc = check_code_quality(
                prob['prompt'], code, prob.get('test_list'))

            all_solutions.append({
                'task_id': prob['task_id'], 'prompt': prob['prompt'],
                'code': code.strip(), 'quality': quality, 'correct': is_correct,
                'cell': desc, 'attempt': attempt,
            })

        if (idx + 1) % 10 == 0:
            cells = len(set(s['cell'] for s in all_solutions))
            correct = sum(1 for s in all_solutions if s['correct'])
            elapsed = time.time() - t0
            print(f"  {idx+1}/{len(problems)} probs, {len(all_solutions)} sol, "
                  f"{correct} ok, {cells} cells, {elapsed:.0f}s", flush=True)

    # Save
    with open(gen_file, 'w') as f:
        json.dump(all_solutions, f, ensure_ascii=False)
    elapsed = time.time() - t0
    print(f"\n  Generated {len(all_solutions)} solutions in {elapsed:.0f}s", flush=True)
    print(f"  Saved to {gen_file}", flush=True)

    del model; torch.cuda.empty_cache()

# === Phase 2+3+4: Selection, Training, Evaluation ===
elif args.phase == 'eval':
    print(f"\n--- Phase 2: Merge generation pools ---", flush=True)

    # Merge all generation pool files
    all_solutions = []
    for f in sorted(RESULTS_DIR.glob("gen_pool_*.json")):
        chunk = json.load(open(f))
        all_solutions.extend(chunk)
        print(f"  Loaded {len(chunk)} from {f.name}", flush=True)

    # Also load V9 pool if available
    v9_pool_file = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/scale_v9/generation_pool.json")
    if v9_pool_file.exists():
        v9_pool = json.load(open(v9_pool_file))
        print(f"  V9 pool: {len(v9_pool)} solutions", flush=True)
        # Merge (avoid duplicates by task_id + code)
        existing_codes = set((s['task_id'], s['code'][:100]) for s in all_solutions)
        added = 0
        for s in v9_pool:
            key = (s['task_id'], s['code'][:100])
            if key not in existing_codes:
                # Convert V9 format
                all_solutions.append({
                    'task_id': s['task_id'], 'prompt': s['prompt'],
                    'code': s['code'], 'quality': s['quality'],
                    'correct': s['correct'], 'cell': s['cell_expert'],
                    'attempt': -1,
                })
                existing_codes.add(key)
                added += 1
        print(f"  Added {added} unique solutions from V9 pool", flush=True)

    # Normalize cell to tuple for hashability
    for s in all_solutions:
        if isinstance(s['cell'], list):
            s['cell'] = tuple(s['cell'])

    total_gen = len(all_solutions)
    total_cells = len(set(s['cell'] for s in all_solutions))
    total_correct = sum(1 for s in all_solutions if s['correct'])
    print(f"\n  Total pool: {total_gen} solutions, {total_correct} correct, {total_cells} cells", flush=True)

    # Save merged pool
    with open(RESULTS_DIR / "merged_pool.json", 'w') as f:
        json.dump(all_solutions, f, ensure_ascii=False)

    # === Phase 2: Selection at multiple k values ===
    print(f"\n--- Phase 2: Selection ---", flush=True)
    K_VALUES = [200, 500, 1000, 2000, 5000]
    STRATEGIES = ['qd', 'greedy', 'random']
    SEEDS = [42, 123, 456]

    selection_results = {}

    for k in K_VALUES:
        if k > len(all_solutions):
            print(f"  k={k} > pool size {len(all_solutions)}, skipping", flush=True)
            continue
        for strategy in STRATEGIES:
            for seed in SEEDS:
                key = f"{strategy}_k{k}_s{seed}"
                if strategy == 'qd':
                    selected = select_qd(all_solutions, k)
                elif strategy == 'greedy':
                    selected = select_greedy(all_solutions, k)
                elif strategy == 'random':
                    selected = select_random(all_solutions, k, seed=seed)

                n_cells = len(set(s['cell'] for s in selected))
                n_correct = sum(1 for s in selected if s['correct'])
                avg_quality = np.mean([s['quality'] for s in selected])
                entropy = -sum((c/len(selected)) * np.log2(c/len(selected))
                               for c in Counter(s['cell'] for s in selected).values() if c > 0)

                selection_results[key] = {
                    'strategy': strategy, 'k': k, 'seed': seed,
                    'n_selected': len(selected), 'n_cells': n_cells,
                    'n_correct': n_correct, 'avg_quality': avg_quality,
                    'entropy': entropy,
                    'selected_data': [{'prompt': s['prompt'][:200], 'cell': list(s['cell']) if isinstance(s['cell'], tuple) else s['cell'],
                                       'quality': s['quality'], 'correct': s['correct']}
                                      for s in selected],
                }
                print(f"  {key}: {n_cells} cells, H={entropy:.2f}, q={avg_quality:.3f}, "
                      f"{n_correct}/{len(selected)} correct", flush=True)

    # Save selection results
    with open(RESULTS_DIR / "selection_results.json", 'w') as f:
        json.dump(selection_results, f, indent=2, ensure_ascii=False)

    # === Phase 3: Training + Evaluation ===
    print(f"\n--- Phase 3: Training + Evaluation ---", flush=True)

    # Focus on key configs for training (not all combinations)
    # Train configs: k=500, 1000, 2000, 5000 × strategies × seed=42
    TRAIN_CONFIGS = []
    for k in [500, 1000, 2000, 5000]:
        for strategy in ['qd', 'greedy', 'random']:
            key = f"{strategy}_k{k}_s42"
            if key in selection_results:
                TRAIN_CONFIGS.append(key)

    downstream_results = {}

    for config_key in TRAIN_CONFIGS:
        sel = selection_results[config_key]
        print(f"\n  Training: {config_key} ({sel['n_selected']} samples, {sel['n_cells']} cells)", flush=True)

        # Prepare training data
        train_texts = []
        for item in sel['selected_data']:
            # Find full code for this item
            item_cell = tuple(item['cell']) if isinstance(item['cell'], list) else item['cell']
            matching = [s for s in all_solutions
                        if s['prompt'][:200] == item['prompt'] and s['cell'] == item_cell]
            if matching:
                code = matching[0]['code']
                text = f"### Problem:\n{matching[0]['prompt']}\n\n### Solution:\n{code}"
                train_texts.append(text)

        if len(train_texts) < 10:
            print(f"    Skipping (only {len(train_texts)} train texts)", flush=True)
            continue

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
        ckpt_dir = RESULTS_DIR / f"ckpt_{config_key}"
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
        merged_path = str(RESULTS_DIR / f"merged_{config_key}")
        merged.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)

        del trainer, model, merged; torch.cuda.empty_cache()

        # Evaluate on HumanEval
        print(f"    Evaluating HumanEval...", flush=True)
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
            msgs = [
                {"role": "system", "content": "Complete the Python function."},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=MAX_SEQ_LENGTH).to(eval_model.device)
            with torch.no_grad():
                outputs = eval_model.generate(**inputs, max_new_tokens=MAX_CODE_TOKENS,
                                               do_sample=False, pad_token_id=tokenizer.eos_token_id)
            completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=True)
            full_code = prompt + completion
            passed, n_tests = execute_code_safely(full_code, [test_code])
            if passed > 0: he_correct += 1
            he_total += 1

        he_acc = he_correct / he_total if he_total > 0 else 0

        # Evaluate on MBPP test
        print(f"    Evaluating MBPP test...", flush=True)
        mbpp_correct = 0; mbpp_total = 0
        for idx, item in enumerate(mbpp_test):
            prompt = item['prompt']
            test_list = item['test_list']
            msgs = [
                {"role": "system", "content": "Write a Python function to solve the problem."},
                {"role": "user", "content": f"Write a Python function:\n\n{prompt}\n\nProvide the implementation:"}
            ]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=MAX_SEQ_LENGTH).to(eval_model.device)
            with torch.no_grad():
                outputs = eval_model.generate(**inputs, max_new_tokens=MAX_CODE_TOKENS,
                                               do_sample=False, pad_token_id=tokenizer.eos_token_id)
            completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=True)
            code_blocks = re.findall(r'```python\s*(.*?)```', completion, re.DOTALL)
            code = code_blocks[0] if code_blocks else completion
            passed, n_tests = execute_code_safely(code, test_list)
            if passed == n_tests: mbpp_correct += 1
            mbpp_total += 1

        mbpp_acc = mbpp_correct / mbpp_total if mbpp_total > 0 else 0

        downstream_results[config_key] = {
            'strategy': sel['strategy'], 'k': sel['k'], 'seed': sel['seed'],
            'n_train': sel['n_selected'], 'n_cells': sel['n_cells'],
            'entropy': sel['entropy'],
            'humaneval_pass1': he_acc, 'he_correct': he_correct, 'he_total': he_total,
            'mbpp_acc': mbpp_acc, 'mbpp_correct': mbpp_correct, 'mbpp_total': mbpp_total,
        }

        print(f"    {config_key}: HumanEval={he_acc:.4f} ({he_correct}/{he_total}), "
              f"MBPP={mbpp_acc:.4f} ({mbpp_correct}/{mbpp_total})", flush=True)

        # Save intermediate results
        with open(RESULTS_DIR / "downstream_results.json", 'w') as f:
            json.dump(downstream_results, f, indent=2)

        # Clean up checkpoint to save disk
        del eval_model; torch.cuda.empty_cache()
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        shutil.rmtree(merged_path, ignore_errors=True)
        print(f"    Cleaned up checkpoint", flush=True)

    # === Phase 4: Summary ===
    print(f"\n{'='*70}", flush=True)
    print(f"V10 LARGE-SCALE RESULTS SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Pool: {total_gen} solutions, {total_cells} cells, {total_correct} correct", flush=True)

    print(f"\n  Selection Results:", flush=True)
    print(f"  {'Config':<30} {'Cells':>6} {'Entropy':>8} {'Quality':>8} {'Correct':>8}", flush=True)
    print(f"  {'-'*62}", flush=True)
    for k in K_VALUES:
        for strategy in STRATEGIES:
            key = f"{strategy}_k{k}_s42"
            if key in selection_results:
                r = selection_results[key]
                print(f"  {key:<30} {r['n_cells']:>6} {r['entropy']:>8.2f} "
                      f"{r['avg_quality']:>8.3f} {r['n_correct']:>8}", flush=True)
        print(f"  {'-'*62}", flush=True)

    print(f"\n  Downstream Results:", flush=True)
    print(f"  {'Config':<30} {'HE pass@1':>10} {'MBPP acc':>10} {'Cells':>6}", flush=True)
    print(f"  {'-'*58}", flush=True)
    for key, r in sorted(downstream_results.items()):
        print(f"  {key:<30} {r['humaneval_pass1']:>10.4f} {r['mbpp_acc']:>10.4f} {r['n_cells']:>6}", flush=True)

    # Save full results
    full_results = {
        'pool_stats': {'n_total': total_gen, 'n_cells': total_cells, 'n_correct': total_correct},
        'selection': {k: {kk: vv for kk, vv in v.items() if kk != 'selected_data'}
                      for k, v in selection_results.items()},
        'downstream': downstream_results,
    }
    with open(RESULTS_DIR / "v10_full_results.json", 'w') as f:
        json.dump(full_results, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}", flush=True)
