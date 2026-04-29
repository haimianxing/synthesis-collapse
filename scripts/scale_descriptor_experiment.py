"""
Scale + Descriptor Sensitivity Experiment (V9)
================================================
Addresses SAC concerns:
  1. Scale: Generate 10K+ solutions, select at k=200/500/1000
  2. Descriptor: Compare domain-expert vs length-based vs random descriptors
  3. k-scaling: Show coverage gap persists at larger k

Runs on GPU 5-7 (GPU 1-4 occupied by V8).

Usage:
  CUDA_VISIBLE_DEVICES=5 python3.9 -u scale_descriptor_experiment.py
"""
import os, sys, json, random, re, torch, numpy as np, subprocess, time, argparse
from pathlib import Path
from collections import Counter, defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-7B-Instruct"
RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/scale_v9")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_SOLUTIONS_PER_PROBLEM = 8  # ~3016 total from 377 MBPP problems (1.6x V7)
MAX_SEQ_LENGTH = 512
MAX_CODE_TOKENS = 512

torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

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

print(f"=== Scale + Descriptor V9 ===", flush=True)
print(f"  {len(prompt_pool)} problems, {N_SOLUTIONS_PER_PROBLEM} sol/problem = ~{len(prompt_pool)*N_SOLUTIONS_PER_PROBLEM} total", flush=True)

# === Descriptor Functions ===
def get_code_descriptor_expert(prompt, code=None):
    """Domain-expert: 3D (complexity × algorithm × io), 5×5×5=125 cells"""
    text = (prompt + " " + (code or "")).lower()
    loops = len(re.findall(r'\bfor\b|\bwhile\b', text))
    conds = len(re.findall(r'\bif\b|\belif\b', text))
    funcs = len(re.findall(r'\bdef\b', text))
    comprehensions = len(re.findall(r'\[.*for.*in.*\]', text))
    lines = len(text.split('\n'))
    complexity = min(int((loops*2 + conds + funcs + comprehensions*1.5 + lines*0.1) / 4), 4)
    if re.search(r'sort|order|rank|ascending|descending', text): alg = 0
    elif re.search(r'search|find|index|contains|exists', text): alg = 1
    elif re.search(r'sum|average|mean|max|min|count|total|product', text): alg = 2
    elif re.search(r'replace|split|join|strip|uppercase|lowercase|reverse', text): alg = 3
    else: alg = 4
    if re.search(r'return.*\[|list|append|extend|pop|remove', text): io = 0
    elif re.search(r'return.*true|return.*false|bool|is_|check|verify', text): io = 1
    elif re.search(r'return.*\d|int|float|number|count|length|size', text): io = 2
    elif re.search(r'return.*["\']|str|string|char', text): io = 3
    else: io = 4
    return (complexity, alg, io)

def get_code_descriptor_length(prompt, code=None):
    """Length-based: 3D (code_length × prompt_length × has_test), 5×5×2=50 cells"""
    code_text = code or ""
    prompt_text = prompt or ""
    code_len = min(int(len(code_text) / 100), 4)
    prompt_len = min(int(len(prompt_text) / 50), 4)
    has_test = 0 if not code_text else (1 if 'assert' in code_text or 'test' in code_text.lower() else 0)
    return (code_len, prompt_len, has_test)

def get_code_descriptor_random(prompt, code=None, seed=42):
    """Random: randomly assign to 125 cells (same cardinality as expert)"""
    rng = random.Random(hash(prompt) + seed)
    return (rng.randint(0, 4), rng.randint(0, 4), rng.randint(0, 4))

def shannon_entropy(items):
    counts = Counter(items)
    total = sum(counts.values())
    if total == 0: return 0.0
    return -sum((c/total) * np.log2(c/total) for c in counts.values() if c > 0)

# === Code Execution ===
def execute_code_safely(code_str, test_cases, timeout=5):
    passed = 0
    total = len(test_cases)
    for test in test_cases:
        try:
            full_code = code_str + "\n" + test
            result = subprocess.run(['python3', '-c', full_code],
                                    timeout=timeout, capture_output=True, text=True)
            if result.returncode == 0: passed += 1
        except (subprocess.TimeoutExpired, Exception): pass
    return passed, total

def check_code_quality(prompt_text, generated_code, test_list=None):
    descriptor = get_code_descriptor_expert(prompt_text, generated_code)
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
def select_qd(solutions, n_select, desc_key='cell_expert'):
    archive = {}
    for sol in solutions:
        cell = sol[desc_key]
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

# === Generation ===
def generate_all_solutions(model, tokenizer, problems, n_per_problem):
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
                    **inputs, max_new_tokens=MAX_CODE_TOKENS,
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

            quality, is_correct, _ = check_code_quality(prob['prompt'], code, prob.get('test_list'))

            # Compute all descriptor types
            cell_expert = get_code_descriptor_expert(prob['prompt'], code)
            cell_length = get_code_descriptor_length(prob['prompt'], code)
            cell_random = get_code_descriptor_random(prob['prompt'], code)

            solutions.append({
                'task_id': prob['task_id'], 'prompt': prob['prompt'],
                'code': code.strip(), 'quality': quality, 'correct': is_correct,
                'cell_expert': cell_expert, 'cell_length': cell_length,
                'cell_random': cell_random, 'attempt': attempt,
            })

        if (idx + 1) % 20 == 0:
            cells = len(set(s['cell_expert'] for s in solutions))
            ok = sum(1 for s in solutions if s['correct'])
            print(f"    Gen {idx+1}/{len(problems)} ({len(solutions)} sol, {ok} ok, {cells} expert-cells)", flush=True)

    return solutions

# === Training ===
def train_and_eval(train_data, label, k):
    """Train LoRA on selected data and evaluate on HumanEval."""
    print(f"  Training {label} (k={k}, {len(train_data)} samples)...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )

    train_texts = [f"### Problem:\n{item['prompt']}\n\n### Solution:\n{item['code']}" for item in train_data]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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
        output_dir=str(RESULTS_DIR / f"ckpt_{label}_k{k}"),
        num_train_epochs=3, per_device_train_batch_size=2,
        gradient_accumulation_steps=4, learning_rate=2e-4,
        bf16=True, logging_steps=10, save_strategy="no", report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset,
                      data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False))
    trainer.train()

    merged_model = model.merge_and_unload()
    merged_path = str(RESULTS_DIR / f"merged_{label}_k{k}")
    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)

    del trainer, model, merged_model
    torch.cuda.empty_cache()

    # Evaluate
    print(f"  Evaluating {label}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        merged_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )
    model.eval()

    correct = 0; total = 0
    for idx, item in enumerate(humaneval):
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
        passed, _ = execute_code_safely(full_code, [test_code])
        if passed > 0: correct += 1
        total += 1

    acc = correct / total if total > 0 else 0
    del model; torch.cuda.empty_cache()

    print(f"  {label} k={k}: pass@1={acc:.4f} ({correct}/{total})", flush=True)
    return acc, correct, total

# === MAIN ===
t0 = time.time()
all_results = {}

# Phase 1: Generate large pool
print(f"\n{'='*50}", flush=True)
print(f"  PHASE 1: Generation ({len(prompt_pool)} probs × {N_SOLUTIONS_PER_PROBLEM} sol)", flush=True)
print(f"{'='*50}", flush=True)

gen_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
gen_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True
)
gen_model.eval()

solutions = generate_all_solutions(gen_model, gen_tokenizer, prompt_pool, N_SOLUTIONS_PER_PROBLEM)

del gen_model; torch.cuda.empty_cache()

# Save generation pool
gen_pool_path = RESULTS_DIR / "generation_pool.json"
with open(gen_pool_path, 'w') as f:
    json.dump([{
        'task_id': s['task_id'],
        'prompt': s['prompt'][:200],
        'code': s['code'][:500],
        'quality': s['quality'],
        'correct': s['correct'],
        'cell_expert': list(s['cell_expert']),
        'cell_length': list(s['cell_length']),
        'cell_random': list(s['cell_random']),
    } for s in solutions], f, indent=2, ensure_ascii=False)

# Phase 2: k-scaling experiment (expert descriptor)
print(f"\n{'='*50}", flush=True)
print(f"  PHASE 2: k-Scaling (expert descriptor)", flush=True)
print(f"{'='*50}", flush=True)

n_total = len(solutions)
cells_expert = len(set(s['cell_expert'] for s in solutions))
cells_length = len(set(s['cell_length'] for s in solutions))
cells_random = len(set(s['cell_random'] for s in solutions))
print(f"  Pool: {n_total} solutions, expert={cells_expert} cells, length={cells_length} cells, random={cells_random} cells", flush=True)

K_VALUES = [200, 500]  # 200 replicates V7, 500 addresses scale concern

for k in K_VALUES:
    if k > n_total:
        print(f"  Skipping k={k} (pool size={n_total})", flush=True)
        continue

    print(f"\n  --- k={k} ---", flush=True)

    # QD selection (expert descriptor)
    qd_sel = select_qd(solutions, k, 'cell_expert')
    qd_cells = len(set(s['cell_expert'] for s in qd_sel))
    qd_entropy = shannon_entropy([s['cell_expert'] for s in qd_sel])
    qd_quality = np.mean([s['quality'] for s in qd_sel])

    # Greedy selection
    gr_sel = select_greedy(solutions, k)
    gr_cells = len(set(s['cell_expert'] for s in gr_sel))
    gr_entropy = shannon_entropy([s['cell_expert'] for s in gr_sel])
    gr_quality = np.mean([s['quality'] for s in gr_sel])

    # Random selection
    rn_sel = select_random(solutions, k)
    rn_cells = len(set(s['cell_expert'] for s in rn_sel))
    rn_entropy = shannon_entropy([s['cell_expert'] for s in rn_sel])
    rn_quality = np.mean([s['quality'] for s in rn_sel])

    print(f"    QD: {qd_cells} cells, H={qd_entropy:.2f}, q={qd_quality:.3f}", flush=True)
    print(f"    Greedy: {gr_cells} cells, H={gr_entropy:.2f}, q={gr_quality:.3f}", flush=True)
    print(f"    Random: {rn_cells} cells, H={rn_entropy:.2f}, q={rn_quality:.3f}", flush=True)

    # Train and evaluate
    qd_acc, qd_corr, qd_tot = train_and_eval(qd_sel, "qd", k)
    gr_acc, gr_corr, gr_tot = train_and_eval(gr_sel, "greedy", k)
    rn_acc, rn_corr, rn_tot = train_and_eval(rn_sel, "random", k)

    all_results[f'k{k}'] = {
        'k': k,
        'qd': {'cells': qd_cells, 'entropy': qd_entropy, 'quality': qd_quality,
                'pass@1': qd_acc, 'correct': qd_corr, 'total': qd_tot},
        'greedy': {'cells': gr_cells, 'entropy': gr_entropy, 'quality': gr_quality,
                   'pass@1': gr_acc, 'correct': gr_corr, 'total': gr_tot},
        'random': {'cells': rn_cells, 'entropy': rn_entropy, 'quality': rn_quality,
                   'pass@1': rn_acc, 'correct': rn_corr, 'total': rn_tot},
    }

    # Save intermediate results
    with open(RESULTS_DIR / "k_scaling_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

# Phase 3: Descriptor sensitivity (at k=200)
print(f"\n{'='*50}", flush=True)
print(f"  PHASE 3: Descriptor Sensitivity (k=200)", flush=True)
print(f"{'='*50}", flush=True)

k_desc = 200
for desc_name, desc_key in [('expert', 'cell_expert'), ('length', 'cell_length'), ('random', 'cell_random')]:
    print(f"\n  --- Descriptor: {desc_name} ---", flush=True)
    qd_sel = select_qd(solutions, k_desc, desc_key)
    cells = len(set(s[desc_key] for s in qd_sel))
    entropy = shannon_entropy([s[desc_key] for s in qd_sel])
    quality = np.mean([s['quality'] for s in qd_sel])
    print(f"    {desc_name}: {cells} cells, H={entropy:.2f}, q={quality:.3f}", flush=True)

    acc, corr, tot = train_and_eval(qd_sel, f"desc_{desc_name}", k_desc)

    all_results[f'desc_{desc_name}'] = {
        'descriptor': desc_name, 'k': k_desc,
        'cells': cells, 'entropy': entropy, 'quality': quality,
        'pass@1': acc, 'correct': corr, 'total': tot,
    }

    with open(RESULTS_DIR / "descriptor_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

# === Final Summary ===
elapsed = time.time() - t0
print(f"\n{'='*60}", flush=True)
print(f"FINAL RESULTS (Scale V9, {elapsed/3600:.1f}h)", flush=True)
print(f"{'='*60}", flush=True)

print(f"\nPool: {n_total} solutions, {cells_expert} expert cells", flush=True)

print(f"\n### k-Scaling ###")
for k in K_VALUES:
    key = f'k{k}'
    if key in all_results:
        r = all_results[key]
        print(f"  k={k}: QD {r['qd']['cells']}c/{r['qd']['entropy']:.2f}H/{r['qd']['pass@1']:.4f} "
              f"vs Greedy {r['greedy']['cells']}c/{r['greedy']['entropy']:.2f}H/{r['greedy']['pass@1']:.4f} "
              f"vs Random {r['random']['cells']}c/{r['random']['entropy']:.2f}H/{r['random']['pass@1']:.4f}", flush=True)

print(f"\n### Descriptor Sensitivity ###")
for desc in ['expert', 'length', 'random']:
    key = f'desc_{desc}'
    if key in all_results:
        r = all_results[key]
        print(f"  {desc}: {r['cells']}c/{r['entropy']:.2f}H/{r['pass@1']:.4f}", flush=True)

with open(RESULTS_DIR / "all_results.json", 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\nDone. Results in {RESULTS_DIR}", flush=True)
