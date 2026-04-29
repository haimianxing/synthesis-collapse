"""
Multi-Seed Iterative Collapse Validation
Proves collapse is reproducible across random seeds (NOT a single-run artifact).

Design: 3 seeds × 2 strategies × 3 rounds × 50 samples/round = 900 API calls
- Seeds change initial selection from MBPP pool → different starting archives
- 3 rounds sufficient to show greedy freeze pattern
- Saves per-seed metrics for statistical analysis
"""
import os, sys, json, random, re, time, ast, numpy as np
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from datasets import load_dataset
from openai import OpenAI

API_KEY = "sk-bf04f622fcd94499833c70e98dac0803"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-plus"

GPU_ID = int(os.environ.get("GPU_ID", "6"))
SEEDS = [42, 123, 271]
GRID_RES = 10
N_SEED_ITEMS = 20
N_PER_ROUND = 50
N_ROUNDS = 3

RESULTS_DIR = Path(f"/mnt/data2/zcz/neurIps-emnlp/neurips/results/code_multi_seed")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

print(f"=== Multi-Seed Collapse Validation (GPU {GPU_ID}) ===", flush=True)
print(f"Seeds: {SEEDS}, Rounds: {N_ROUNDS}, Samples/round: {N_PER_ROUND}", flush=True)

# Load MBPP (offline)
print("Loading MBPP (cached)...", flush=True)
try:
    mbpp = load_dataset("mbpp", "sanitized", split="test")
except:
    mbpp = load_dataset("mbpp", split="test")
print(f"MBPP: {len(mbpp)}", flush=True)

# Descriptors (same as iterative_v2)
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

# Build pool
pool = []
for ex in mbpp:
    code = ex.get('code', '')
    desc = compute_code_descriptors(ex.get('prompt', ''), code)
    pool.append({
        'prompt': ex.get('prompt', ''), 'code': code, 'text': ex.get('text', ''),
        'descriptors': desc, 'quality': compute_quality({'code': code})
    })

# API functions (same as iterative_v2)
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

# ============ Main: Multi-seed experiment ============
all_seed_results = {}

for seed in SEEDS:
    print(f"\n{'='*60}\nSEED {seed}\n{'='*60}", flush=True)
    random.seed(seed)
    np.random.seed(seed)

    # Select different seed items based on seed
    shuffled_pool = list(pool)
    random.shuffle(shuffled_pool)
    seed_items = []
    used_cells = set()
    for item in shuffled_pool:
        cell = get_cell(item['descriptors'])
        if cell not in used_cells:
            seed_items.append(item)
            used_cells.add(cell)
            if len(seed_items) >= N_SEED_ITEMS:
                break
    print(f"  Seed items: {len(seed_items)} ({len(used_cells)} unique cells)", flush=True)

    seed_data = {}
    for strategy in ["greedy", "qd"]:
        print(f"\n  --- Strategy: {strategy.upper()} ---", flush=True)
        archive = {}
        for item in seed_items:
            cell = get_cell(item['descriptors'])
            if cell not in archive or item['quality'] > archive[cell]['quality']:
                archive[cell] = dict(item)

        round_data = []
        m = compute_metrics(archive)
        m['round'] = 0
        round_data.append(m)
        print(f"    R0: cells={m['n_cells']}, cov={m['coverage']}", flush=True)

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

                if (i + 1) % 25 == 0:
                    print(f"      R{rnd}: {i+1}/{N_PER_ROUND}, cells={len(archive)}", flush=True)

            m = compute_metrics(archive)
            m['round'] = rnd
            m['n_new'] = len(new_items)
            m['time_s'] = round(time.time() - t0)
            round_data.append(m)
            print(f"    R{rnd}: cells={m['n_cells']}, cov={m['coverage']}, ent={m['entropy']}, n_new={len(new_items)}, {m['time_s']}s", flush=True)

        seed_data[strategy] = round_data
        # Save per-seed archive
        archive_save = [{k: v for k, v in item.items() if k != 'descriptors'} for item in archive.values()]
        with open(RESULTS_DIR / f"seed{seed}_{strategy}_archive.json", "w") as f:
            json.dump(archive_save, f, indent=2, ensure_ascii=False)

    all_seed_results[str(seed)] = seed_data

    # Save intermediate
    with open(RESULTS_DIR / "multi_seed_results.json", "w") as f:
        json.dump(all_seed_results, f, indent=2, default=str)

# ============ Statistical Analysis ============
print(f"\n{'='*60}\nSTATISTICAL ANALYSIS\n{'='*60}", flush=True)

for rnd in range(N_ROUNDS + 1):
    greedy_cells = []
    qd_cells = []
    for seed in SEEDS:
        sk = str(seed)
        if sk in all_seed_results:
            greedy_cells.append(all_seed_results[sk]['greedy'][rnd]['n_cells'])
            qd_cells.append(all_seed_results[sk]['qd'][rnd]['n_cells'])

    g_mean = np.mean(greedy_cells)
    g_std = np.std(greedy_cells)
    q_mean = np.mean(qd_cells)
    q_std = np.std(qd_cells)
    diff = q_mean - g_mean

    print(f"  R{rnd}: Greedy={g_mean:.1f}±{g_std:.1f}, QD={q_mean:.1f}±{q_std:.1f}, Δ={diff:.1f}", flush=True)

# Compute growth rates
greedy_growth = []
qd_growth = []
for seed in SEEDS:
    sk = str(seed)
    if sk in all_seed_results:
        g_r0 = all_seed_results[sk]['greedy'][0]['n_cells']
        g_r3 = all_seed_results[sk]['greedy'][N_ROUNDS]['n_cells']
        q_r0 = all_seed_results[sk]['qd'][0]['n_cells']
        q_r3 = all_seed_results[sk]['qd'][N_ROUNDS]['n_cells']
        greedy_growth.append((g_r3 - g_r0) / max(g_r0, 1))
        qd_growth.append((q_r3 - q_r0) / max(q_r0, 1))

print(f"\n  Growth rates (R0→R3):", flush=True)
print(f"    Greedy: {np.mean(greedy_growth):.3f} ± {np.std(greedy_growth):.3f}", flush=True)
print(f"    QD:     {np.mean(qd_growth):.3f} ± {np.std(qd_growth):.3f}", flush=True)

# Wilcoxon test
if len(greedy_cells) >= 3 and len(qd_cells) >= 3:
    from scipy.stats import wilcoxon
    try:
        stat, p_val = wilcoxon([qd_growth[i] - greedy_growth[i] for i in range(len(SEEDS))])
        print(f"    Wilcoxon: W={stat:.1f}, p={p_val:.4f}", flush=True)
    except:
        print(f"    Wilcoxon: insufficient variation for test", flush=True)

with open(RESULTS_DIR / "multi_seed_results.json", "w") as f:
    json.dump(all_seed_results, f, indent=2, default=str)
print(f"\nDone. Results: {RESULTS_DIR / 'multi_seed_results.json'}", flush=True)
