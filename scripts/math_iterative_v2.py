"""
Math Iterative v2: 100 samples/round × 7 rounds
Saves per-round archives for downstream fine-tuning analysis.
Shows collapse prevention at larger scale.
"""
import os, sys, json, random, re, time, numpy as np
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset
from openai import OpenAI

API_KEY = "sk-bf04f622fcd94499833c70e98dac0803"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-plus"

GRID_RES = 10
N_SEED = 20
N_PER_ROUND = 100
N_ROUNDS = 7

RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/math_iterative_v2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

print(f"=== Math Iterative v2 (100/round × 7 rounds) ===", flush=True)

# Load GSM8K
print("Loading GSM8K...", flush=True)
train_ds = load_dataset("gsm8k", "main", split="test")  # Use test as pool to avoid contamination
print(f"Pool: {len(train_ds)}", flush=True)

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
    pool.append({'question': q, 'answer': a, 'answer_num': ans,
                 'descriptors': desc, 'quality': compute_quality({'answer': a})})
print(f"Valid pool: {len(pool)}", flush=True)

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
                    {"role": "system", "content": "You are a math education expert."},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.85, max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"    API error: {e}", flush=True)
            time.sleep(3 * (attempt + 1))
    return None

def generate_greedy(parent):
    prompt = f"""Here is a math word problem:
{parent['question']}

Solution:
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
        return parse_math(resp)
    return None

def generate_qd(parent, grid_state):
    empty_cells = []
    for di in range(GRID_RES):
        for si in range(GRID_RES):
            for mi in range(2):
                if (di, si, mi) not in grid_state:
                    empty_cells.append((di, si, mi))

    if empty_cells:
        target = random.choice(empty_cells)
        d_val = target[0] / GRID_RES
        s_val = target[1] / GRID_RES
        m_val = target[2]
        d_label = "easy (1-2 operations)" if d_val < 0.3 else ("medium" if d_val < 0.7 else "hard (5+ operations)")
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
        return parse_math(resp)
    return None

def parse_math(text):
    ans_match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if not ans_match:
        return None
    answer_num = ans_match.group(1).replace(',', '')
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
    return {'coverage': round(coverage, 4), 'n_cells': n_cells, 'entropy': round(entropy, 4),
            'avg_quality': round(np.mean(qualities), 4) if qualities else 0}

# Run both strategies
all_results = {}

for strategy in ["greedy", "qd"]:
    print(f"\n{'='*50}\nSTRATEGY: {strategy.upper()} (100/round × 7 rounds)\n{'='*50}", flush=True)

    archive = {}
    for item in seeds:
        cell = get_cell(item['descriptors'])
        if cell not in archive or item['quality'] > archive[cell]['quality']:
            archive[cell] = dict(item)

    round_data = []
    m = compute_metrics(archive)
    m['round'] = 0
    m['strategy'] = strategy
    round_data.append(m)
    print(f"  R0: cells={m['n_cells']}, cov={m['coverage']}", flush=True)

    # Save R0
    with open(RESULTS_DIR / f"{strategy}_archive_r0.json", "w") as f:
        json.dump([{k: v for k, v in item.items() if k != 'descriptors'} for item in archive.values()], f, indent=2, ensure_ascii=False)

    for rnd in range(1, N_ROUNDS + 1):
        t0 = time.time()
        new_items = []
        grid_cells = set(archive.keys())

        for i in range(N_PER_ROUND):
            parents = list(archive.values())

            if strategy == "greedy":
                parent = max(parents, key=lambda x: x['quality'])
                result = generate_greedy(parent)
            else:
                parent = random.choice(parents)
                result = generate_qd(parent, grid_cells)

            if result:
                result['descriptors'] = compute_descriptors(result['question'], result['answer'])
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
        print(f"  R{rnd}: cells={m['n_cells']}, cov={m['coverage']}, ent={m['entropy']}, n_new={len(new_items)}", flush=True)

        # Save per-round archive
        with open(RESULTS_DIR / f"{strategy}_archive_r{rnd}.json", "w") as f:
            json.dump([{k: v for k, v in item.items() if k != 'descriptors'} for item in archive.values()], f, indent=2, ensure_ascii=False)

        all_results[strategy] = round_data
        with open(RESULTS_DIR / "math_iterative_v2_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

print(f"\nDone. Results: {RESULTS_DIR / 'math_iterative_v2_results.json'}", flush=True)
