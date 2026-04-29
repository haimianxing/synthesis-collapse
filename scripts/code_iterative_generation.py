"""
Code Domain Iterative Generation Experiment
Addresses: QD-Synth's core contribution is ITERATIVE GENERATION with directed mutation,
not just static selection. This experiment mirrors Table 5 (Dialogue collapse) for Code.

Design:
- Start from 20 seed MBPP problems
- 5 rounds of iterative generation (50 new problems per round)
- Greedy-Iter: select top-k exemplars by quality for next-round few-shot
- QD-Iter: select via MAP-Elites archive, mutate toward empty cells
- Track: coverage, entropy, cell count, quality per round
- Use Qwen3.5-122B API for generation

This is the KEY experiment: QD can generate for empty cells (Greedy cannot).
"""
import os, sys, json, random, re, time, ast, torch, numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from datasets import load_dataset
from openai import OpenAI

# ============ Config ============
API_KEY = "sk-bf04f622fcd94499833c70e98dac0803"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-plus"  # Use qwen-plus for faster generation

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-1___5B-Instruct"
GRID_RES = 10
N_SEED = 20
N_PER_ROUND = 50
N_ROUNDS = 5
GPU_ID = int(os.environ.get("GPU_ID", "1"))
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/code_iterative")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

print(f"=== Code Iterative Generation (GPU {GPU_ID}) ===", flush=True)
print(f"Config: {N_SEED} seeds, {N_PER_ROUND}/round, {N_ROUNDS} rounds", flush=True)

# ============ Load MBPP seed pool ============
print("Loading MBPP...", flush=True)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
try:
    mbpp = load_dataset("mbpp", "sanitized", split="test")
except:
    mbpp = load_dataset("mbpp", split="test")
print(f"MBPP: {len(mbpp)}", flush=True)

# Load HumanEval for final evaluation
print("Loading HumanEval...", flush=True)
humaneval = load_dataset("openai_humaneval", split="test")
print(f"HumanEval: {len(humaneval)}", flush=True)

# ============ Descriptors ============
def compute_code_descriptors(prompt, code, test_list=None):
    code_len = len(code) if code else 100
    difficulty = min(code_len / 1000.0, 1.0)
    api_count = 0
    try:
        tree = ast.parse(code) if code else None
        if tree:
            for node in ast.walk(tree):
                if isinstance(node, ast.Call): api_count += 1
                elif isinstance(node, ast.Import): api_count += len(node.names)
                elif isinstance(node, ast.ImportFrom): api_count += len(node.names)
    except:
        api_count = len(re.findall(r'\b\w+\.\w+\(', code)) if code else 0
    has_debug = 1 if (code and ('try:' in code or 'except' in code or 'assert' in code)) else 0
    return {'difficulty': difficulty, 'num_APIs': min(api_count / 10.0, 1.0), 'needs_debugging': has_debug}

def get_cell(desc):
    return (int(desc['difficulty'] * GRID_RES), int(desc['num_APIs'] * GRID_RES), int(desc['needs_debugging'] * GRID_RES))

def compute_quality(sample):
    code = sample.get('code', '')
    return min(len(code) / 500.0, 1.0) if code else 0.1

# Build full pool
pool = []
for ex in mbpp:
    code = ex.get('code', '')
    desc = compute_code_descriptors(ex.get('prompt', ''), code)
    pool.append({
        'prompt': ex.get('prompt', ''), 'code': code, 'text': ex.get('text', ''),
        'test_list': ex.get('test_list', []), 'descriptors': desc,
        'quality': compute_quality({'code': code})
    })

# Select 20 seeds (diverse)
random.seed(42)
seeds = []
used_cells = set()
for item in sorted(pool, key=lambda x: x['quality'], reverse=True):
    cell = get_cell(item['descriptors'])
    if cell not in used_cells:
        seeds.append(item)
        used_cells.add(cell)
        if len(seeds) >= N_SEED:
            break
print(f"Selected {len(seeds)} seed problems ({len(used_cells)} unique cells)", flush=True)

# ============ API Generation ============
def call_api(prompt_text, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a Python programming expert. Generate problems according to the specification. Output only the problem in the exact format requested."},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.85, max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"    API error (attempt {attempt+1}): {e}", flush=True)
            time.sleep(3 * (attempt + 1))
    return None

def generate_variant_greedy(parent, n=1):
    """Greedy mutation: ask LLM to create a similar problem (no cell targeting)"""
    results = []
    for _ in range(n):
        prompt = f"""Here is a Python programming problem:

```
{parent['prompt']}
```

Solution:
```python
{parent['code'][:500]}
```

Create a NEW, DIFFERENT Python programming problem with similar difficulty.
The problem should have a function signature, docstring, and be solvable in Python.

Output format:
**Problem:**
[problem description with function signature]

**Solution:**
```python
[solution code]
```"""
        resp = call_api(prompt)
        if resp:
            parsed = parse_code_problem(resp)
            if parsed:
                results.append(parsed)
    return results

def generate_variant_qd(parent, target_cell, grid_state):
    """QD mutation: ask LLM to create a problem for a SPECIFIC target cell"""
    d, a, b = target_cell
    d_label = "easy" if d < 3 else ("medium" if d < 7 else "hard")
    n_apis = "few (0-3)" if a < 3 else ("several (4-7)" if a < 7 else "many (8+)")
    debug_str = "with error handling (try/except/assert)" if b >= 5 else "without explicit error handling"

    # Find an empty cell to target
    empty_cells = []
    for di in range(GRID_RES):
        for ai in range(GRID_RES):
            for bi in range(GRID_RES):
                if (di, ai, bi) not in grid_state:
                    empty_cells.append((di, ai, bi))

    if empty_cells:
        target = random.choice(empty_cells)
        d_val = target[0] / GRID_RES
        a_val = target[1] / GRID_RES
        b_val = target[2] / GRID_RES
        d_label = "easy (short solution)" if d_val < 0.3 else ("medium" if d_val < 0.7 else "hard (long solution)")
        n_apis = "few (0-2)" if a_val < 0.3 else ("several (3-6)" if a_val < 0.7 else "many (7+)")
        debug_str = "with try/except or assert statements" if b_val >= 0.5 else "without explicit error handling"

    prompt = f"""Here is a reference Python programming problem:

```
{parent['prompt'][:300]}
```

Create a NEW Python programming problem with these SPECIFIC characteristics:
- Difficulty: {d_label}
- Number of standard library API calls: {n_apis}
- Error handling: {debug_str}

The problem must be solvable, have a clear function signature, and include test cases.

Output format:
**Problem:**
[problem description with function signature and docstring]

**Solution:**
```python
[solution code]
```"""
    resp = call_api(prompt)
    if resp:
        parsed = parse_code_problem(resp)
        if parsed:
            return [parsed]
    return []

def parse_code_problem(text):
    """Parse generated text into a structured problem"""
    # Extract problem and solution
    code_blocks = re.findall(r'```python\s*(.*?)\s*```', text, re.DOTALL)
    if not code_blocks:
        code_blocks = re.findall(r'```\s*(.*?)\s*```', text, re.DOTALL)

    # Try to find function signature
    func_match = re.search(r'def\s+(\w+)\s*\([^)]*\)\s*:', text)
    if not func_match and code_blocks:
        func_match = re.search(r'def\s+(\w+)\s*\([^)]*\)\s*:', code_blocks[0])

    if not func_match:
        return None

    func_name = func_match.group(1)

    # Extract code from the last code block (usually the solution)
    code = code_blocks[-1].strip() if code_blocks else ""

    # Extract problem description (text before first code block)
    problem_text = text.split('```')[0].strip() if '```' in text else text[:500]
    # Clean up
    problem_text = re.sub(r'\*\*Problem:\*\*\s*', '', problem_text)
    problem_text = re.sub(r'\*\*Solution:\*\*.*', '', problem_text, flags=re.DOTALL)

    if len(code) < 20:
        return None

    # Verify code is parseable
    try:
        ast.parse(code)
    except:
        return None

    return {
        'prompt': problem_text[:500],
        'code': code[:1000],
        'text': problem_text[:300],
        'test_list': [],
    }

# ============ Run Iterative Generation ============
def run_iteration(seeds_list, strategy="greedy", round_num=0, prev_archive=None):
    """Run one round of iterative generation"""
    grid = dict(prev_archive) if prev_archive else {}

    # Add seeds to grid
    for item in seeds_list:
        if 'descriptors' not in item:
            item['descriptors'] = compute_code_descriptors(item.get('prompt', ''), item.get('code', ''))
        if 'quality' not in item:
            item['quality'] = compute_quality(item)
        cell = get_cell(item['descriptors'])
        if cell not in grid or item['quality'] > grid[cell]['quality']:
            grid[cell] = item

    # Generate new problems
    new_items = []
    grid_cells = set(grid.keys())

    for i in range(N_PER_ROUND):
        if strategy == "greedy":
            # Pick best quality parent
            parents = sorted(grid.values(), key=lambda x: x['quality'], reverse=True)[:5]
            parent = random.choice(parents)
            variants = generate_variant_greedy(parent)
        else:  # QD
            # Pick random parent, target empty cell
            parents = list(grid.values())
            parent = random.choice(parents)
            variants = generate_variant_qd(parent, None, grid_cells)

        for v in variants:
            v['descriptors'] = compute_code_descriptors(v.get('prompt', ''), v.get('code', ''))
            v['quality'] = compute_quality(v)
            cell = get_cell(v['descriptors'])
            if strategy == "qd":
                # QD: accept if new cell OR better quality in existing cell
                if cell not in grid or v['quality'] > grid[cell]['quality']:
                    grid[cell] = v
                    grid_cells.add(cell)
            else:
                # Greedy: just collect all, will filter by quality later
                pass
            new_items.append(v)

        if (i + 1) % 10 == 0:
            print(f"    Round {round_num}, {strategy}: {i+1}/{N_PER_ROUND} generated, grid size={len(grid)}", flush=True)

    if strategy == "greedy":
        # Greedy: select top-N_PER_ROUND by quality
        all_items = list(grid.values()) + new_items
        all_items.sort(key=lambda x: x['quality'], reverse=True)
        for item in all_items[:N_PER_ROUND]:
            cell = get_cell(item['descriptors'])
            if cell not in grid or item['quality'] > grid[cell]['quality']:
                grid[cell] = item

    return grid, new_items

def compute_metrics(grid):
    """Compute coverage, entropy, cell count"""
    n_cells = len(grid)
    total_cells = GRID_RES ** 3
    coverage = n_cells / total_cells

    # Entropy over quality distribution
    qualities = [v['quality'] for v in grid.values() if v.get('quality')]
    if len(qualities) > 1:
        q_arr = np.array(qualities)
        q_arr = q_arr / q_arr.sum()
        entropy = -np.sum(q_arr * np.log(q_arr + 1e-10))
    else:
        entropy = 0.0

    # Unique difficulty levels
    difficulties = set()
    api_counts = set()
    debug_flags = set()
    for v in grid.values():
        d = v['descriptors']
        difficulties.add(d['difficulty'])
        api_counts.add(d['num_APIs'])
        debug_flags.add(d['needs_debugging'])

    return {
        'coverage': round(coverage, 4),
        'n_cells': n_cells,
        'entropy': round(entropy, 4),
        'n_difficulty_levels': len(difficulties),
        'n_api_levels': len(api_counts),
        'n_debug_types': len(debug_flags),
        'avg_quality': round(np.mean(qualities), 4) if qualities else 0,
    }

# ============ Main Loop ============
print("\n" + "="*60, flush=True)
print("RUNNING ITERATIVE GENERATION EXPERIMENT", flush=True)
print("="*60, flush=True)

all_round_results = {"greedy": {}, "qd": {}}

for strategy in ["greedy", "qd"]:
    print(f"\n{'='*40}", flush=True)
    print(f"STRATEGY: {strategy.upper()}", flush=True)
    print(f"{'='*40}", flush=True)

    archive = {}
    current_seeds = list(seeds)  # Start from same 20 seeds

    # Initialize with seeds
    for item in current_seeds:
        cell = get_cell(item['descriptors'])
        if cell not in archive or item['quality'] > archive[cell]['quality']:
            archive[cell] = item

    round_data = []
    # Round 0 (seeds)
    metrics = compute_metrics(archive)
    metrics['round'] = 0
    metrics['strategy'] = strategy
    round_data.append(metrics)
    print(f"  Round 0 (seeds): cells={metrics['n_cells']}, cov={metrics['coverage']}, ent={metrics['entropy']}", flush=True)

    for round_num in range(1, N_ROUNDS + 1):
        print(f"\n  Round {round_num}/{N_ROUNDS} ({strategy})", flush=True)
        t0 = time.time()

        if strategy == "greedy":
            # Select exemplars: top-k by quality
            exemplars = sorted(archive.values(), key=lambda x: x['quality'], reverse=True)[:20]
            archive, new_items = run_iteration(exemplars, "greedy", round_num, archive)
        else:
            # QD: use archive for exemplar selection, mutate toward empty cells
            archive, new_items = run_iteration(list(archive.values()), "qd", round_num, archive)

        metrics = compute_metrics(archive)
        metrics['round'] = round_num
        metrics['strategy'] = strategy
        metrics['n_new'] = len(new_items)
        metrics['time_s'] = round(time.time() - t0)
        round_data.append(metrics)
        print(f"  Round {round_num}: cells={metrics['n_cells']}, cov={metrics['coverage']}, ent={metrics['entropy']}, n_new={metrics['n_new']}, time={metrics['time_s']}s", flush=True)

        # Save intermediate metrics
        all_round_results[strategy] = round_data
        # Save archive items for later fine-tuning
        archive_save = []
        for cell_key, item in archive.items():
            save_item = {k: v for k, v in item.items() if k != 'descriptors'}
            save_item['cell'] = list(cell_key)
            archive_save.append(save_item)
        with open(RESULTS_DIR / f"{strategy}_archive.json", "w") as f:
            json.dump(archive_save, f, indent=2, default=str, ensure_ascii=False)
        with open(RESULTS_DIR / "iterative_results.json", "w") as f:
            json.dump(all_round_results, f, indent=2, default=str)

# ============ Fine-tune & Evaluate Final Archives ============
print(f"\n{'='*60}", flush=True)
print("FINE-TUNING FINAL ARCHIVES ON HumanEval", flush=True)
print(f"{'='*60}", flush=True)

DEVICE = "cuda:0"

def finetune_and_eval(train_samples, config_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    output_dir = RESULTS_DIR / f"model_{config_name}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"], lora_dropout=0.05, task_type="CAUSAL_LM"))

    def fmt(s):
        return f"<|im_start|>system\nComplete the Python function.<|im_end|>\n<|im_start|>user\n{s['prompt'][:512]}<|im_end|>\n<|im_start|>assistant\n{s['code'][:768]}<|im_end|>"

    ds = Dataset.from_dict({"text": [fmt(s) for s in train_samples]})
    trainer = SFTTrainer(model=model, args=SFTConfig(output_dir=str(output_dir), num_train_epochs=3, per_device_train_batch_size=4, gradient_accumulation_steps=4, learning_rate=2e-4, logging_steps=50, save_strategy="no", bf16=True, report_to="none", max_length=768, dataset_text_field="text", packing=False), train_dataset=ds, processing_class=tokenizer)
    trainer.train()
    model.save_pretrained(output_dir / "lora"); tokenizer.save_pretrained(output_dir / "lora")

    # Evaluate
    model.eval()
    correct = total = 0
    for ex in humaneval:
        msgs = [{"role":"system","content":"Complete the Python function."},{"role":"user","content":ex['prompt']}]
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(txt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=256, temperature=0.0, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        try:
            exec_globals = {}; exec(ex['prompt'] + resp, exec_globals); exec(ex['test'], exec_globals)
            correct += 1
        except: pass
        total += 1

    del model; torch.cuda.empty_cache()
    return {"pass_at_1": round(correct/total, 4), "correct": correct, "total": total, "n_train": len(train_samples)}

# Load final archives
for strategy in ["greedy", "qd"]:
    # Reload archive from results
    round_data = all_round_results[strategy]
    print(f"\n--- {strategy.upper()} final fine-tuning ---", flush=True)

    # We need to reconstruct the archive - use the saved grid state
    # For simplicity, re-run with saved state
    # Actually, let's just collect all generated items from the JSON
    pass

# Re-construct archives from saved results and fine-tune
print("Reconstructing archives for fine-tuning...", flush=True)

# We need to save the archive items too. Let me re-approach:
# Save archives to disk during generation, then load for fine-tuning
print("NOTE: Archives were saved during generation. Loading...", flush=True)

# Load saved archives
greedy_archive_path = RESULTS_DIR / "greedy_archive.json"
qd_archive_path = RESULTS_DIR / "qd_archive.json"

# Check if archives were saved
if greedy_archive_path.exists() and qd_archive_path.exists():
    for strategy, path in [("greedy", greedy_archive_path), ("qd", qd_archive_path)]:
        with open(path) as f:
            archive_items = json.load(f)
        valid = [item for item in archive_items if item.get('code') and len(item['code']) > 20]
        print(f"  {strategy}: {len(valid)} valid training samples", flush=True)
        if valid:
            result = finetune_and_eval(valid, f"{strategy}_iter")
            print(f"  {strategy} pass@1: {result['pass_at_1']} ({result['correct']}/{result['total']})", flush=True)
            all_round_results[f"{strategy}_downstream"] = result
else:
    print("WARNING: Archives not found. Need to re-run generation.", flush=True)

# Save final results
with open(RESULTS_DIR / "iterative_results.json", "w") as f:
    json.dump(all_round_results, f, indent=2, default=str)

print(f"\nResults saved to {RESULTS_DIR / 'iterative_results.json'}", flush=True)
