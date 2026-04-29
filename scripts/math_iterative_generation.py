"""
Math Domain Iterative Generation Experiment
Mirror of Code iterative experiment, for Math domain (GSM8K).
Shows collapse prevention across 3 domains (Dialogue + Code + Math).

Design: Same as Code iterative - 20 seeds, 5 rounds, 50/round.
QD targets empty cells; Greedy selects top quality.
"""
import os, sys, json, random, re, time, torch, numpy as np
from pathlib import Path
from collections import defaultdict
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset
from openai import OpenAI

API_KEY = "sk-bf04f622fcd94499833c70e98dac0803"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-plus"
MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-1___5B-Instruct"
GRID_RES = 10
N_SEED = 20
N_PER_ROUND = 50
N_ROUNDS = 5
GPU_ID = int(os.environ.get("GPU_ID", "3"))
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/math_iterative")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

print(f"=== Math Iterative Generation (GPU {GPU_ID}) ===", flush=True)

# Load GSM8K
print("Loading GSM8K...", flush=True)
train_ds = load_dataset("gsm8k", "main", split="train")
test_ds = load_dataset("gsm8k", "main", split="test")
print(f"Train: {len(train_ds)}, Test: {len(test_ds)}", flush=True)

def extract_answer(text):
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    return match.group(1).replace(',', '') if match else None

def compute_descriptors(question, answer):
    steps = answer.count('<<') + 1
    sol_len = len(answer)
    difficulty = min(sol_len / 500.0, 1.0)
    is_multi = 1 if steps >= 3 else 0
    return {'difficulty': difficulty, 'num_steps': min(steps / 10.0, 1.0), 'is_multi_step': is_multi}

def get_cell(desc):
    return (int(desc['difficulty'] * GRID_RES), int(desc['num_steps'] * GRID_RES), int(desc['is_multi_step'] * GRID_RES))

def compute_quality(sample):
    return min(len(sample.get('answer', '')) / 300.0, 1.0)

# Build pool
pool = []
for ex in train_ds:
    q, a = ex['question'], ex['answer']
    ans = extract_answer(a)
    if ans is None: continue
    desc = compute_descriptors(q, a)
    pool.append({'question': q, 'answer': a, 'answer_num': ans, 'descriptors': desc, 'quality': compute_quality({'answer': a})})
print(f"Pool: {len(pool)}", flush=True)

# Select 20 diverse seeds
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

# API generation
def call_api(prompt_text, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a math education expert. Generate math word problems with step-by-step solutions."},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.85, max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"    API error: {e}", flush=True)
            time.sleep(3 * (attempt + 1))
    return None

def generate_greedy(parent, n=1):
    results = []
    for _ in range(n):
        prompt = f"""Here is a math word problem:

{parent['question']}

Step-by-step solution:
{parent['answer'][:500]}

Create a NEW, DIFFERENT math word problem of similar difficulty.
Include a step-by-step solution ending with #### [answer].

Format:
**Problem:**
[problem text]

**Solution:**
[step-by-step solution]
#### [numerical answer]"""
        resp = call_api(prompt)
        if resp:
            parsed = parse_math(resp)
            if parsed:
                results.append(parsed)
    return results

def generate_qd(parent, grid_cells):
    # Find empty cells to target
    empty_cells = []
    for di in range(GRID_RES):
        for si in range(GRID_RES):
            for mi in range(2):  # binary multi_step
                if (di, si, mi) not in grid_cells:
                    empty_cells.append((di, si, mi))

    if empty_cells:
        target = random.choice(empty_cells)
        d_val = target[0] / GRID_RES
        s_val = target[1] / GRID_RES
        m_val = target[2]
        d_label = "easy (1-2 arithmetic operations)" if d_val < 0.3 else ("medium" if d_val < 0.7 else "hard (5+ operations)")
        s_label = f"{max(1, int(s_val * 10))} steps"
        m_label = "multi-step reasoning" if m_val else "single-step calculation"
    else:
        d_label = "medium"
        s_label = "3 steps"
        m_label = "multi-step"

    prompt = f"""Create a NEW math word problem with these characteristics:
- Difficulty: {d_label}
- Solution steps: {s_label}
- Type: {m_label}

Include a step-by-step solution ending with #### [answer].

Format:
**Problem:**
[problem text]

**Solution:**
[step-by-step solution]
#### [numerical answer]"""
    resp = call_api(prompt)
    if resp:
        parsed = parse_math(resp)
        if parsed:
            return [parsed]
    return []

def parse_math(text):
    # Extract answer
    ans_match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if not ans_match:
        return None
    answer_num = ans_match.group(1).replace(',', '')

    # Extract problem and solution
    parts = text.split('**Problem:**')
    if len(parts) < 2:
        parts = text.split('**Problem:**')
    if len(parts) < 2:
        question = text[:300]
        answer = text
    else:
        rest = parts[1]
        sol_parts = rest.split('**Solution:**')
        question = sol_parts[0].strip()[:500] if sol_parts else rest[:300]
        answer = sol_parts[1].strip() if len(sol_parts) > 1 else rest

    if len(question) < 20:
        return None

    return {'question': question, 'answer': answer, 'answer_num': answer_num}

# ============ Run ============
print("\n" + "="*60, flush=True)
print("RUNNING MATH ITERATIVE GENERATION", flush=True)
print("="*60, flush=True)

all_round_results = {"greedy": {}, "qd": {}}

for strategy in ["greedy", "qd"]:
    print(f"\n{'='*40}\nSTRATEGY: {strategy.upper()}\n{'='*40}", flush=True)

    archive = {}
    for item in seeds:
        if 'descriptors' not in item:
            item['descriptors'] = compute_descriptors(item['question'], item['answer'])
        if 'quality' not in item:
            item['quality'] = compute_quality(item)
        cell = get_cell(item['descriptors'])
        if cell not in archive or item['quality'] > archive[cell]['quality']:
            archive[cell] = item

    round_data = []

    def compute_metrics(grid):
        n_cells = len(grid)
        total_cells = GRID_RES ** 3
        coverage = n_cells / total_cells
        qualities = [v['quality'] for v in grid.values()]
        if len(qualities) > 1:
            q_arr = np.array(qualities); q_arr = q_arr / q_arr.sum()
            entropy = -np.sum(q_arr * np.log(q_arr + 1e-10))
        else:
            entropy = 0.0
        return {'coverage': round(coverage, 4), 'n_cells': n_cells, 'entropy': round(entropy, 4),
                'avg_quality': round(np.mean(qualities), 4) if qualities else 0}

    metrics = compute_metrics(archive)
    metrics['round'] = 0
    metrics['strategy'] = strategy
    round_data.append(metrics)
    print(f"  Round 0: cells={metrics['n_cells']}, cov={metrics['coverage']}", flush=True)

    for round_num in range(1, N_ROUNDS + 1):
        print(f"\n  Round {round_num}/{N_ROUNDS} ({strategy})", flush=True)
        t0 = time.time()

        new_items = []
        grid_cells = set(archive.keys())

        for i in range(N_PER_ROUND):
            parents = list(archive.values())

            if strategy == "greedy":
                parent = max(parents, key=lambda x: x['quality'])
                variants = generate_greedy(parent)
            else:
                parent = random.choice(parents)
                variants = generate_qd(parent, grid_cells)

            for v in variants:
                v['descriptors'] = compute_descriptors(v['question'], v['answer'])
                v['quality'] = compute_quality(v)
                cell = get_cell(v['descriptors'])
                if cell not in archive or v['quality'] > archive[cell]['quality']:
                    archive[cell] = v
                    grid_cells.add(cell)
                new_items.append(v)

            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{N_PER_ROUND}, grid={len(archive)}", flush=True)

        metrics = compute_metrics(archive)
        metrics['round'] = round_num
        metrics['strategy'] = strategy
        metrics['n_new'] = len(new_items)
        metrics['time_s'] = round(time.time() - t0)
        round_data.append(metrics)
        print(f"  Round {round_num}: cells={metrics['n_cells']}, cov={metrics['coverage']}, ent={metrics['entropy']}", flush=True)

        all_round_results[strategy] = round_data
        # Save archive
        archive_save = [{k: v for k, v in item.items() if k != 'descriptors'} for item in archive.values()]
        with open(RESULTS_DIR / f"{strategy}_archive.json", "w") as f:
            json.dump(archive_save, f, indent=2, ensure_ascii=False)
        with open(RESULTS_DIR / "iterative_results.json", "w") as f:
            json.dump(all_round_results, f, indent=2, default=str)

print(f"\nDone. Results: {RESULTS_DIR / 'iterative_results.json'}", flush=True)
