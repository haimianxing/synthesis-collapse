"""
Code Iterative QD Re-run
We have Greedy results (41 cells, saturation visible) but QD results were lost.
Re-run ONLY the QD portion to complete 3-domain collapse evidence.

Enhanced: 100 samples/round × 7 rounds (more data than original 50×5).
Saves per-round archives for downstream fine-tuning analysis.
"""
import os, sys, json, random, re, time, ast, torch, numpy as np
from pathlib import Path
from collections import defaultdict

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset
from openai import OpenAI

API_KEY = "sk-bf04f622fcd94499833c70e98dac0803"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-plus"

GRID_RES = 10
N_SEED = 20
N_PER_ROUND = 100  # More data!
N_ROUNDS = 7       # More rounds!

RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/code_iterative_v2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

print(f"=== Code Iterative v2 (100/round × 7 rounds) ===", flush=True)

# Load MBPP
print("Loading MBPP...", flush=True)
try:
    mbpp = load_dataset("mbpp", "sanitized", split="test")
except:
    mbpp = load_dataset("mbpp", split="test")
print(f"MBPP: {len(mbpp)}", flush=True)

# Descriptors
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

# Build pool and select seeds
pool = []
for ex in mbpp:
    code = ex.get('code', '')
    desc = compute_code_descriptors(ex.get('prompt', ''), code)
    pool.append({
        'prompt': ex.get('prompt', ''), 'code': code, 'text': ex.get('text', ''),
        'descriptors': desc, 'quality': compute_quality({'code': code})
    })

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
print(f"Seeds: {len(seeds)} ({len(used_cells)} cells)", flush=True)

# API Generation
def call_api(prompt_text, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a Python programming expert. Generate problems as specified."},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.85, max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"    API error: {e}", flush=True)
            time.sleep(3 * (attempt + 1))
    return None

def generate_variant_qd(parent, grid_state):
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
    else:
        d_label = "medium"
        n_apis = "several"
        debug_str = "without explicit error handling"

    prompt = f"""Here is a reference Python programming problem:
```
{parent['prompt'][:300]}
```

Create a NEW Python programming problem with these SPECIFIC characteristics:
- Difficulty: {d_label}
- Number of standard library API calls: {n_apis}
- Error handling: {debug_str}

Output format:
**Problem:**
[problem description with function signature]

**Solution:**
```python
[solution code]
```"""
    resp = call_api(prompt)
    if resp:
        return parse_code_problem(resp)
    return None

def generate_variant_greedy(parent):
    prompt = f"""Here is a Python programming problem:
```
{parent['prompt']}
```

Create a NEW, DIFFERENT Python programming problem with similar difficulty.

Output format:
**Problem:**
[problem description]

**Solution:**
```python
[solution code]
```"""
    resp = call_api(prompt)
    if resp:
        return parse_code_problem(resp)
    return None

def parse_code_problem(text):
    code_blocks = re.findall(r'```python\s*(.*?)\s*```', text, re.DOTALL)
    if not code_blocks:
        code_blocks = re.findall(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if not code_blocks:
        return None
    code = code_blocks[-1].strip()
    if len(code) < 20:
        return None
    try:
        ast.parse(code)
    except:
        return None
    problem_text = text.split('```')[0].strip()[:500]
    return {'prompt': problem_text, 'code': code[:1000], 'text': problem_text[:300], 'test_list': []}

def compute_metrics(grid):
    n_cells = len(grid)
    total_cells = GRID_RES ** 3
    coverage = n_cells / total_cells
    qualities = [v['quality'] for v in grid.values()]
    if len(qualities) > 1:
        q_arr = np.array(qualities)
        q_arr = q_arr / q_arr.sum()
        entropy = -np.sum(q_arr * np.log(q_arr + 1e-10))
    else:
        entropy = 0.0
    return {
        'coverage': round(coverage, 4), 'n_cells': n_cells,
        'entropy': round(entropy, 4),
        'avg_quality': round(np.mean(qualities), 4) if qualities else 0
    }

# Run BOTH strategies
all_results = {}

for strategy in ["greedy", "qd"]:
    print(f"\n{'='*50}\nSTRATEGY: {strategy.upper()} (100/round × 7 rounds)\n{'='*50}", flush=True)

    archive = {}
    # Initialize with seeds
    for item in seeds:
        cell = get_cell(item['descriptors'])
        if cell not in archive or item['quality'] > archive[cell]['quality']:
            archive[cell] = dict(item)

    round_data = []
    # Round 0
    m = compute_metrics(archive)
    m['round'] = 0
    m['strategy'] = strategy
    round_data.append(m)
    print(f"  R0: cells={m['n_cells']}, cov={m['coverage']}", flush=True)

    # Save R0 archive
    archive_save = [{k: v for k, v in item.items() if k != 'descriptors'} for item in archive.values()]
    with open(RESULTS_DIR / f"{strategy}_archive_r0.json", "w") as f:
        json.dump(archive_save, f, indent=2, ensure_ascii=False)

    for rnd in range(1, N_ROUNDS + 1):
        t0 = time.time()
        new_items = []
        grid_cells = set(archive.keys())

        for i in range(N_PER_ROUND):
            parents = list(archive.values())

            if strategy == "greedy":
                top_parents = sorted(parents, key=lambda x: x['quality'], reverse=True)[:5]
                parent = random.choice(top_parents)
                result = generate_variant_greedy(parent)
            else:
                parent = random.choice(parents)
                result = generate_variant_qd(parent, grid_cells)

            if result:
                result['descriptors'] = compute_code_descriptors(result.get('prompt', ''), result.get('code', ''))
                result['quality'] = compute_quality(result)
                cell = get_cell(result['descriptors'])
                if cell not in archive or result['quality'] > archive[cell]['quality']:
                    archive[cell] = result
                    grid_cells.add(cell)
                new_items.append(result)

            if (i + 1) % 20 == 0:
                print(f"    R{rnd}: {i+1}/{N_PER_ROUND}, cells={len(archive)}", flush=True)

        m = compute_metrics(archive)
        m['round'] = rnd
        m['strategy'] = strategy
        m['n_new'] = len(new_items)
        m['time_s'] = round(time.time() - t0)
        round_data.append(m)
        print(f"  R{rnd}: cells={m['n_cells']}, cov={m['coverage']}, ent={m['entropy']}, n_new={len(new_items)}, {m['time_s']}s", flush=True)

        # Save per-round archive
        archive_save = [{k: v for k, v in item.items() if k != 'descriptors'} for item in archive.values()]
        with open(RESULTS_DIR / f"{strategy}_archive_r{rnd}.json", "w") as f:
            json.dump(archive_save, f, indent=2, ensure_ascii=False)

        # Save metrics
        all_results[strategy] = round_data
        with open(RESULTS_DIR / "iterative_v2_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

print(f"\nDone. Results: {RESULTS_DIR / 'iterative_v2_results.json'}", flush=True)
