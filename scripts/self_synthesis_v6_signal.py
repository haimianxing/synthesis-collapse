#!/usr/bin/env python3
"""
Self-Synthesis v6: Signal Amplification Experiment
===================================================

DIAGNOSIS of why v3/v4 didn't show clear QD advantage:
  1. Descriptor is effectively 1D (dim2 constant, dim1 80%+ same value)
  2. Selection pressure too weak (30%) → 98.7% QD/Greedy overlap
  3. Accuracy metric doesn't capture QD's actual advantage (entropy/coverage)

KEY INNOVATIONS:
  A. Improved 3D descriptor with BALANCED dimensions:
     dim0: difficulty (0-4, based on step count + number count + op count + keywords)
     dim1: operation_diversity (0-4, count of unique op types: +,-,*,/,comparison)
     dim2: problem_type (0-4: equation/rate/geometry/ratio/arithmetic)
     → 5×5×5 = 125 cells (vs old 10×10×10 with 2 dead dimensions)

  B. Higher selection pressure: 5000 gen → 300 select (6% vs 30%)
     → Forces QD and Greedy to make VERY different choices

  C. TRIZ S6: Wasserstein-driven cell selection
     → QD selects parents from cells maximally different from archive centroid

  D. TRIZ S3: Golden Ratio archive protection
     → ≥61.8% of R0 seeds preserved across rounds

  E. Shared generation pool: All strategies select from SAME generated data
     → Isolates selection effect from generation quality

  F. PRIMARY METRIC: Entropy trajectory (not just accuracy)
     → QD should show monotonically increasing entropy; Greedy should decrease

DESIGN:
  Phase 1: Generate 5000 samples from base model (1 GPU, ~4h)
  Phase 2: Select + Train + Eval for 4 strategies (4 GPUs, ~30min each)
  Repeat for R0, R1, R2 (3 rounds)

STRATEGIES:
  1. qd_v6      : QD with improved descriptor + Wasserstein + Golden Ratio
  2. greedy_v6  : Greedy top-quality selection (baseline)
  3. random_v6  : Random selection (baseline)
  4. qd_old_desc: QD with OLD descriptor (ablation: does new descriptor help?)

Usage:
  # Phase 1: Generate shared pool
  CUDA_VISIBLE_DEVICES=X python -u self_synthesis_v6_signal.py --phase generate --round 0

  # Phase 2: Select + Train + Eval (one strategy)
  CUDA_VISIBLE_DEVICES=X python -u self_synthesis_v6_signal.py --phase train \
    --strategy qd_v6 --round 0

  # Or run all phases for one strategy
  CUDA_VISIBLE_DEVICES=X python -u self_synthesis_v6_signal.py --phase all \
    --strategy qd_v6
"""

import os, sys, json, random, re, torch, numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import time, math, argparse

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-7B-Instruct"
DEVICE = "cuda:0"

# === v6 Hyperparameters ===
GRID_RES = 5          # 5×5×5 = 125 cells (improved from 10×10×10)
N_ROUNDS = 3
N_GENERATE = 3000     # 3× more generation (faster iteration)
N_SELECT = 300        # 6% selection pressure (vs 30%)
N_EVAL = 500
GOLDEN_RATIO = 0.618  # S3: ≥61.8% seed protection

RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/self_synthesis_v6_signal")
SHARED_POOL_DIR = RESULTS_DIR / "shared_pool"

# ============================================================
# IMPROVED 3D DESCRIPTOR (Key Innovation A)
# ============================================================
def compute_descriptor_improved(question, answer=None):
    """
    Balanced 3D descriptor where each dimension discriminates well.

    dim0: difficulty (0-4) based on reasoning steps, number count, operations, keywords
    dim1: operation_diversity (0-4) based on count of unique arithmetic ops
    dim2: problem_type (0-4) based on problem structure
    """
    q_lower = question.lower()

    # dim0: Difficulty estimation
    steps = 0
    if answer:
        lines = [l.strip() for l in answer.split('\n') if l.strip()]
        steps = len([l for l in lines if any(c.isdigit() for c in l) or '=' in l])

    q_len = len(question.split())
    ops = len(re.findall(r'[+\-*/×÷]', question))
    numbers = re.findall(r'\d+\.?\d*', question)
    n_numbers = len(numbers)

    diff_score = 0
    if steps >= 6: diff_score += 2
    elif steps >= 3: diff_score += 1
    if n_numbers >= 5: diff_score += 1
    if ops >= 2: diff_score += 1
    if any(w in q_lower for w in ['each', 'every', 'per', 'ratio', 'fraction', 'percent']):
        diff_score += 1

    dim0 = min(diff_score, 4)  # 0-4

    # dim1: Operation diversity
    op_types = set()
    if '+' in question or 'add' in q_lower or 'sum' in q_lower or 'total' in q_lower:
        op_types.add('add')
    if '-' in question or 'subtract' in q_lower or 'difference' in q_lower or 'remain' in q_lower or 'left' in q_lower:
        op_types.add('sub')
    if '*' in question or '×' in question or 'multiply' in q_lower or 'product' in q_lower or 'times' in q_lower:
        op_types.add('mul')
    if '/' in question or '÷' in question or 'divide' in q_lower or 'split' in q_lower or 'share' in q_lower:
        op_types.add('div')
    if any(w in q_lower for w in ['more than', 'less than', 'greater', 'fewer', 'at least', 'at most']):
        op_types.add('cmp')

    dim1 = min(len(op_types), 4)  # 0-4

    # dim2: Problem type
    if any(w in q_lower for w in ['equation', 'solve for', 'find the value', '=']):
        dim2 = 0  # equation
    elif any(w in q_lower for w in ['rate', 'speed', 'per hour', 'per minute', 'mph', 'km/h']):
        dim2 = 1  # rate/speed
    elif any(w in q_lower for w in ['area', 'perimeter', 'volume', 'length', 'width',
                                      'rectangle', 'circle', 'triangle']):
        dim2 = 2  # geometry
    elif any(w in q_lower for w in ['ratio', 'fraction', 'proportion', 'percent']):
        dim2 = 3  # ratio/fraction
    else:
        dim2 = 4  # arithmetic/word problem

    return (dim0, dim1, dim2)

def compute_descriptor_old(question, answer=None):
    """Old v3 descriptor for ablation comparison."""
    if not answer or len(answer) < 20: return None
    steps = answer.count('<<') + answer.count('\n') + 1
    difficulty = min(len(answer) / 800.0, 1.0)
    struct = 0 if steps <= 2 else (1 if steps <= 6 else 2)
    grid_res = 10
    return (int(difficulty * grid_res), int(min(steps / 15.0, 1.0) * grid_res), struct * (grid_res // 2))

# ============================================================
# QUALITY & ANSWER FUNCTIONS
# ============================================================
def extract_answer(text):
    m = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    return m.group(1).replace(',', '') if m else None

def extract_flexible(text):
    for pattern in [
        r'####\s*(-?[\d,]+\.?\d*)',
        r'(?:answer is|answer:)\s*\$?(-?[\d,]+\.?\d*)',
        r'\\boxed\{(-?[\d,]+\.?\d*)\}',
        r'(?:=|is)\s*(-?[\d,]+\.?\d*)',
    ]:
        m = re.search(pattern, text, re.IGNORECASE if 'answer' in pattern else 0)
        if m: return m.group(1).replace(',', '')
    nums = re.findall(r'-?[\d,]+\.?\d*', text.strip())
    return nums[-1].replace(',', '') if nums else None

def check_correct(pred, gold):
    if not pred or not gold: return False
    try:
        return abs(float(pred.strip()) - float(gold.strip())) < 1e-6
    except:
        return pred.strip() == gold.strip()

def quality_score(answer, gold_answer=None):
    if not answer or len(answer) < 20: return 0
    score = 0
    if '####' in answer: score += 0.2
    score += min(answer.count('<<') / 5.0, 0.2)
    score += min(len(answer) / 800.0, 0.15)
    pred = extract_flexible(answer)
    if pred: score += 0.15
    if gold_answer and pred:
        try:
            if abs(float(pred.strip()) - float(gold_answer.strip())) < 1e-6: score += 0.3
        except:
            if pred.strip() == gold_answer.strip(): score += 0.3
    return min(score, 1.0)

# ============================================================
# DIVERSITY METRICS
# ============================================================
def compute_cell_entropy(solutions):
    cells = [s['cell'] for s in solutions if s.get('cell')]
    if not cells: return 0
    counts = Counter(cells)
    total = len(cells)
    return -sum((c/total) * math.log2(c/total) for c in counts.values() if c > 0)

def compute_vocab_diversity(solutions):
    ratios = []
    for s in solutions:
        ans = s.get('answer', '')
        if not ans: continue
        tokens = ans.lower().split()
        if tokens: ratios.append(len(set(tokens)) / len(tokens))
    return np.mean(ratios) if ratios else 0

def compute_self_bleu(solutions, max_n=100):
    """Simplified Self-BLEU: average pairwise 1-gram overlap."""
    if len(solutions) < 2: return 0
    samples = solutions[:max_n]
    texts = [s.get('answer', '').lower().split() for s in samples if s.get('answer')]
    if len(texts) < 2: return 0

    total_overlap = 0
    n_pairs = 0
    for i in range(min(len(texts), 50)):
        for j in range(i+1, min(len(texts), 50)):
            if not texts[i] or not texts[j]: continue
            set_i, set_j = set(texts[i]), set(texts[j])
            if not set_i or not set_j: continue
            overlap = len(set_i & set_j) / min(len(set_i), len(set_j))
            total_overlap += overlap
            n_pairs += 1

    return total_overlap / n_pairs if n_pairs > 0 else 0

# ============================================================
# SELECTION STRATEGIES
# ============================================================
def select_greedy(solutions, n):
    """Greedy: top-n by quality."""
    return sorted([s for s in solutions if s['quality'] > 0.1],
                  key=lambda x: x['quality'], reverse=True)[:n]

def select_random(solutions, n, seed=42):
    """Random selection baseline."""
    valid = [s for s in solutions if s['quality'] > 0.1]
    rng = random.Random(seed)
    return rng.sample(valid, min(n, len(valid)))

def select_qd_v6(solutions, n, archive_cells=None, wasserstein=True):
    """
    QD with improved selection:
    1. MAP-Elites: one best per cell
    2. Surprisal: prioritize empty cells (archive-guided)
    3. S6 Wasserstein: prefer cells furthest from archive centroid
    """
    if archive_cells is None: archive_cells = set()

    cell_to_items = defaultdict(list)
    for sol in solutions:
        if sol.get('cell') and sol['quality'] > 0.1:
            cell_to_items[sol['cell']].append(sol)

    if not cell_to_items: return select_greedy(solutions, n)

    # Phase 1: Fill empty cells first (surprisal)
    empty_cell_best = []
    for cell, items in cell_to_items.items():
        if cell not in archive_cells:
            empty_cell_best.append(max(items, key=lambda x: x['quality']))

    # S6 Wasserstein: sort empty cell candidates by distance from archive centroid
    if wasserstein and archive_cells and empty_cell_best:
        # Compute archive centroid
        archive_cells_list = list(archive_cells)
        if archive_cells_list:
            centroid = np.mean([np.array(c) for c in archive_cells_list], axis=0)
            # Sort by distance from centroid (furthest first)
            empty_cell_best.sort(
                key=lambda x: -np.linalg.norm(np.array(x['cell']) - centroid))
        else:
            empty_cell_best.sort(key=lambda x: x['quality'], reverse=True)
    else:
        empty_cell_best.sort(key=lambda x: x['quality'], reverse=True)

    selected = list(empty_cell_best[:int(n * 0.5)])  # 50% for new cells

    # Phase 2: Fill from filled cells (best per cell)
    filled_cell_best = []
    for cell, items in cell_to_items.items():
        if cell in archive_cells:
            best = max(items, key=lambda x: x['quality'])
            if best not in selected:
                filled_cell_best.append(best)

    # S6: sort by distance from centroid for filled cells too
    if wasserstein and archive_cells and filled_cell_best:
        archive_cells_list = list(archive_cells)
        if archive_cells_list:
            centroid = np.mean([np.array(c) for c in archive_cells_list], axis=0)
            filled_cell_best.sort(
                key=lambda x: -np.linalg.norm(np.array(x['cell']) - centroid))

    for item in filled_cell_best:
        if len(selected) >= n: break
        if item not in selected:
            selected.append(item)

    # Phase 3: Fill by quality
    if len(selected) < n:
        remaining = [s for s in solutions if s not in selected and s['quality'] > 0.1]
        remaining.sort(key=lambda x: x['quality'], reverse=True)
        selected.extend(remaining[:n - len(selected)])

    return selected[:n]

def select_qd_old_descriptor(solutions, n, archive_cells=None):
    """QD with OLD descriptor (ablation: does new descriptor help?)."""
    # Re-compute cells using old descriptor
    for s in solutions:
        s['cell_old'] = compute_descriptor_old(s['question'], s.get('answer'))

    cell_to_items = defaultdict(list)
    for sol in solutions:
        if sol.get('cell_old') and sol['quality'] > 0.1:
            cell_to_items[sol['cell_old']].append(sol)

    if not cell_to_items: return select_greedy(solutions, n)

    if archive_cells is None: archive_cells = set()

    selected = []
    # Empty cells first
    for cell, items in cell_to_items.items():
        if cell not in archive_cells:
            selected.append(max(items, key=lambda x: x['quality']))
    # Fill remaining
    if len(selected) < n:
        for cell in sorted(cell_to_items.keys()):
            best = max(cell_to_items[cell], key=lambda x: x['quality'])
            if best not in selected:
                selected.append(best)
                if len(selected) >= n: break
    if len(selected) < n:
        remaining = [s for s in solutions if s not in selected and s['quality'] > 0.1]
        remaining.sort(key=lambda x: x['quality'], reverse=True)
        selected.extend(remaining[:n - len(selected)])

    # Restore original cell
    for s in solutions:
        s.pop('cell_old', None)

    return selected[:n]

# ============================================================
# GENERATION
# ============================================================
def generate_solutions(model, tokenizer, prompts, gold_answers, desc_fn=None):
    """Generate solutions and compute descriptors."""
    if desc_fn is None:
        desc_fn = compute_descriptor_improved

    solutions = []
    SYS = ("You are a mathematics expert. Solve the given math problem step by step.\n\n"
           "Instructions:\n1. Read the problem carefully.\n2. Break it down into steps.\n"
           "3. Show all calculations.\n4. Write your final numerical answer after ####\n\n"
           "Example format:\nJohn has 5 apples and buys 3 more.\n"
           "Step 1: Total = 5 + 3 = 8\n#### 8")

    for i, prompt in enumerate(prompts):
        gold = gold_answers[i]
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": prompt}]
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(txt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=1024, temperature=0.8,
                                 do_sample=True, top_p=0.9,
                                 pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        pred = extract_flexible(resp)
        cell = desc_fn(prompt, resp)

        solutions.append({
            'question': prompt, 'answer': resp,
            'cell': cell,
            'quality': quality_score(resp, gold),
            'correct': check_correct(pred, gold),
        })

        if (i + 1) % 200 == 0:
            nv = sum(1 for s in solutions if s['quality'] > 0.1)
            nc = sum(1 for s in solutions if s['correct'])
            cells = len(set(s['cell'] for s in solutions if s.get('cell')))
            print(f"    Gen {i+1}/{len(prompts)} ({nv}v, {nc}c, {cells} cells)", flush=True)

    return solutions

def evaluate_gsm8k(model, tokenizer, test_data, n=None, seed=42):
    """Evaluate on GSM8K test set."""
    if n:
        rng = random.Random(seed)
        test_data = rng.sample(test_data, min(n, len(test_data)))
    correct = total = 0
    for i, ex in enumerate(test_data):
        msgs = [{"role": "system", "content": "Solve the math problem step by step. Put your final answer after ####."},
                {"role": "user", "content": ex['question']}]
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(txt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=1024, temperature=0.0,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        pred = extract_flexible(resp)
        gold = extract_answer(ex['answer'])
        if check_correct(pred, gold): correct += 1
        total += 1
        if (i + 1) % 100 == 0:
            print(f"    Eval {i+1}/{len(test_data)}, acc={correct/total:.4f}", flush=True)
    return correct, total

def evaluate_math(model, tokenizer, n=500, seed=42):
    """Evaluate on MATH-500 benchmark for cross-domain."""
    try:
        from datasets import load_dataset
        ds = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))
    except:
        print("    MATH-500 not available, skipping", flush=True)
        return None, 0

    rng = random.Random(seed)
    ds = rng.sample(ds, min(n, len(ds)))
    correct = total = 0
    for i, ex in enumerate(ds):
        msgs = [{"role": "system", "content": "Solve the math problem step by step. Put your final answer after ####."},
                {"role": "user", "content": ex['problem']}]
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(txt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=1024, temperature=0.0,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        pred = extract_flexible(resp)
        gold = extract_flexible(ex.get('answer', ''))
        if not gold:
            gold = str(ex.get('answer', ''))
        if check_correct(pred, gold): correct += 1
        total += 1
        if (i + 1) % 100 == 0:
            print(f"    MATH Eval {i+1}/{len(ds)}, acc={correct/total:.4f}", flush=True)
    return correct, total

def fmt_sample(item):
    return (f"<|im_start|>system\nSolve the math problem step by step.<|im_end|>\n"
            f"<|im_start|>user\n{item['question'][:512]}<|im_end|>\n"
            f"<|im_start|>assistant\n{item['answer'][:2048]}<|im_end|>")

# ============================================================
# PHASE 1: GENERATE SHARED POOL
# ============================================================
def phase_generate(round_num, seed=42):
    """Generate shared pool of N_GENERATE solutions."""
    print(f"\n{'='*60}", flush=True)
    print(f"PHASE 1: Generating {N_GENERATE} solutions for Round {round_num}", flush=True)
    print(f"{'='*60}", flush=True)

    SHARED_POOL_DIR.mkdir(parents=True, exist_ok=True)
    pool_file = SHARED_POOL_DIR / f"pool_r{round_num}.json"

    if pool_file.exists():
        print(f"  Pool already exists: {pool_file}", flush=True)
        return json.load(open(pool_file))

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading GSM8K...", flush=True)
    gsm8k_train = list(load_dataset("gsm8k", "main", split="train"))
    prompt_pool = [(ex['question'], extract_answer(ex['answer'])) for ex in gsm8k_train]

    print("Loading base model for generation...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
                                                  device_map=DEVICE, trust_remote_code=True)
    model.eval()

    # Sample prompts (with replacement if needed)
    rng = random.Random(seed + round_num)
    sampled = rng.choices(prompt_pool, k=N_GENERATE)  # choices allows replacement
    prompts = [q for q, a in sampled]
    golds = [a for q, a in sampled]

    solutions = generate_solutions(model, tokenizer, prompts, golds, compute_descriptor_improved)

    # Also compute old descriptor for ablation
    for s in solutions:
        s['cell_old'] = compute_descriptor_old(s['question'], s.get('answer'))
        if s['cell_old']:
            s['cell_old'] = list(s['cell_old'])
        if s['cell']:
            s['cell'] = list(s['cell'])

    # Save pool
    with open(pool_file, 'w') as f:
        json.dump(solutions, f)

    n_valid = sum(1 for s in solutions if s['quality'] > 0.1)
    n_correct = sum(1 for s in solutions if s['correct'])
    cells_new = len(set(tuple(s['cell']) for s in solutions if s.get('cell')))
    cells_old = len(set(tuple(s['cell_old']) for s in solutions if s.get('cell_old')))
    entropy_new = compute_cell_entropy(
        [{**s, 'cell': tuple(s['cell'])} for s in solutions if s.get('cell')])
    entropy_old = compute_cell_entropy(
        [{**s, 'cell': tuple(s['cell_old'])} for s in solutions if s.get('cell_old')])

    print(f"\n  Pool R{round_num} summary:", flush=True)
    print(f"    {len(solutions)} solutions, {n_valid} valid, {n_correct} correct", flush=True)
    print(f"    New descriptor: {cells_new} cells, entropy={entropy_new:.3f}", flush=True)
    print(f"    Old descriptor: {cells_old} cells, entropy={entropy_old:.3f}", flush=True)
    print(f"    Saved to {pool_file}", flush=True)

    del model; torch.cuda.empty_cache()
    return solutions

# ============================================================
# PHASE 2: SELECT + TRAIN + EVAL
# ============================================================
def phase_train(strategy, round_num, seed=42):
    """Select, train, and evaluate for one strategy."""
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    print(f"\n{'='*60}", flush=True)
    print(f"PHASE 2: {strategy} R{round_num} (seed={seed})", flush=True)
    print(f"{'='*60}", flush=True)

    strat_dir = RESULTS_DIR / strategy
    strat_dir.mkdir(parents=True, exist_ok=True)

    # Load results file
    results_file = strat_dir / f"{strategy}_results.json"
    all_results = json.load(open(results_file)) if results_file.exists() else {}

    # Load shared pool
    pool_file = SHARED_POOL_DIR / f"pool_r{round_num}.json"
    if not pool_file.exists():
        print(f"  ERROR: Pool not found at {pool_file}", flush=True)
        return
    pool = json.load(open(pool_file))

    # Restore tuples for cells
    for s in pool:
        if s.get('cell'): s['cell'] = tuple(s['cell'])
        if s.get('cell_old'): s['cell_old'] = tuple(s['cell_old'])

    # Load archive cells from previous rounds
    archive_cells = set()
    archive_cells_old = set()
    accumulated_data = []
    r0_seeds = []  # S3 Golden Ratio: R0 seeds are protected

    for k in sorted(all_results.keys()):
        r = all_results[k]
        if r.get("status") == "completed":
            for c in r.get("archive_cells", []):
                archive_cells.add(tuple(c))
            for c in r.get("archive_cells_old", []):
                archive_cells_old.add(tuple(c))
            if "selected_data" in r:
                accumulated_data.extend(r["selected_data"])
            if r.get("round") == 0 and "selected_data" in r:
                r0_seeds = r["selected_data"]

    # Step 1: SELECT from shared pool
    if strategy == "greedy_v6":
        selected = select_greedy(pool, N_SELECT)
    elif strategy == "random_v6":
        selected = select_random(pool, N_SELECT, seed=seed + round_num)
    elif strategy == "qd_v6":
        selected = select_qd_v6(pool, N_SELECT, archive_cells, wasserstein=True)
    elif strategy == "qd_old_desc":
        selected = select_qd_old_descriptor(pool, N_SELECT, archive_cells_old)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Compute selection metrics
    sel_cells = set(s['cell'] for s in selected if s.get('cell'))
    sel_cells_old = set(s.get('cell_old') for s in selected if s.get('cell_old'))
    sel_correct = sum(1 for s in selected if s['correct'])
    sel_entropy = compute_cell_entropy(selected)
    sel_avg_quality = np.mean([s['quality'] for s in selected]) if selected else 0
    sel_vocab_div = compute_vocab_diversity(selected)
    sel_self_bleu = compute_self_bleu(selected)

    archive_cells.update(sel_cells)
    archive_cells_old.update(sel_cells_old)

    # S3 Golden Ratio: protect R0 seeds
    if round_num > 0 and r0_seeds:
        n_seeds = len(r0_seeds)
        n_total = len(accumulated_data) + len(selected)
        seed_ratio = n_seeds / n_total if n_total > 0 else 0
        if seed_ratio < GOLDEN_RATIO:
            # Add back some seeds to maintain ratio
            n_needed = int(n_total * GOLDEN_RATIO) - n_seeds
            if n_needed > 0:
                rng = random.Random(seed)
                extra_seeds = rng.sample(r0_seeds, min(n_needed, len(r0_seeds)))
                accumulated_data.extend(extra_seeds)
                print(f"  S3: Added {len(extra_seeds)} seeds to maintain golden ratio ({seed_ratio:.1%} → {GOLDEN_RATIO:.1%})", flush=True)

    accumulated_data.extend(selected)

    print(f"  Selected: {len(selected)} samples, {len(sel_cells)} cells, {sel_correct} correct", flush=True)
    print(f"  Entropy={sel_entropy:.3f}, VocabDiv={sel_vocab_div:.3f}, SelfBLEU={sel_self_bleu:.3f}", flush=True)
    print(f"  Accumulated: {len(accumulated_data)} samples from {round_num+1} rounds", flush=True)

    # Step 2: TRAIN from base model
    print(f"  Loading base model for training on {len(accumulated_data)} samples...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    train_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
                                                        device_map=DEVICE, trust_remote_code=True)

    texts = [fmt_sample(s) for s in accumulated_data if s.get('quality', 0) > 0.1]
    if len(texts) < 10:
        print(f"  Too few ({len(texts)}), skipping", flush=True)
        del train_model; torch.cuda.empty_cache(); return

    train_model = get_peft_model(train_model, LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, task_type="CAUSAL_LM"))

    rnd_key = f"{strategy}_r{round_num}"
    ds = Dataset.from_dict({"text": texts})
    trainer = SFTTrainer(
        model=train_model,
        args=SFTConfig(
            output_dir=str(strat_dir / f"ckpt_{rnd_key}"),
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            logging_steps=50,
            save_strategy="no",
            bf16=True,
            report_to="none",
            max_length=1024,
            dataset_text_field="text",
            packing=False),
        train_dataset=ds,
        processing_class=tokenizer)
    trainer.train()

    # Merge and save
    train_model.eval()
    merged = train_model.merge_and_unload()
    merged_path = str(strat_dir / f"merged_{rnd_key}")
    merged.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    del train_model, merged; torch.cuda.empty_cache()

    # Step 3: EVALUATE
    print(f"  Evaluating GSM8K ({N_EVAL})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    eval_model = AutoModelForCausalLM.from_pretrained(merged_path, torch_dtype=torch.bfloat16,
                                                       device_map=DEVICE, trust_remote_code=True)
    eval_model.eval()

    gsm8k_test = list(load_dataset("gsm8k", "main", split="test"))
    correct, total = evaluate_gsm8k(eval_model, tokenizer, gsm8k_test, N_EVAL, seed=seed)
    accuracy = round(correct / total, 4) if total > 0 else 0

    # Cross-domain eval on MATH
    print(f"  Evaluating MATH-500 (cross-domain)...", flush=True)
    math_correct, math_total = evaluate_math(eval_model, tokenizer, n=500, seed=seed)
    math_acc = round(math_correct / math_total, 4) if math_total > 0 else None

    del eval_model; torch.cuda.empty_cache()

    # Save serializable selected data
    serializable_selected = [
        {'question': s['question'][:512], 'answer': s.get('answer', '')[:2048],
         'cell': list(s['cell']) if isinstance(s.get('cell'), tuple) else s.get('cell'),
         'cell_old': list(s['cell_old']) if isinstance(s.get('cell_old'), tuple) else s.get('cell_old'),
         'quality': s['quality'], 'correct': s['correct']}
        for s in selected
    ]

    all_results[rnd_key] = {
        "round": round_num, "seed": seed, "strategy": strategy,
        "experiment": "v6_signal_amplification",
        "n_accumulated": len(accumulated_data),
        "n_selected": len(selected), "n_cells_selected": len(sel_cells),
        "n_correct_selected": sel_correct,
        "sel_entropy": round(sel_entropy, 4),
        "sel_avg_quality": round(sel_avg_quality, 4),
        "sel_vocab_diversity": round(sel_vocab_div, 4),
        "sel_self_bleu": round(sel_self_bleu, 4),
        "archive_cells": [list(c) for c in archive_cells],
        "archive_cells_old": [list(c) for c in archive_cells_old],
        "archive_size": len(archive_cells),
        "gsm8k_accuracy": accuracy, "gsm8k_correct": correct, "gsm8k_total": total,
        "math_accuracy": math_acc, "math_correct": math_correct, "math_total": math_total,
        "status": "completed",
        "selected_data": serializable_selected,
    }
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  R{round_num} {strategy}: GSM8K={accuracy} ({correct}/{total})", flush=True)
    if math_acc is not None:
        print(f"  R{round_num} {strategy}: MATH={math_acc} ({math_correct}/{math_total})", flush=True)
    print(f"  Cells: {len(sel_cells)}, Entropy: {sel_entropy:.3f}, SelfBLEU: {sel_self_bleu:.3f}", flush=True)

    return all_results

# ============================================================
# ANALYSIS: Selection Overlap and QD-Only Cells
# ============================================================
def analyze_selection_overlap(round_num):
    """Analyze overlap between QD and Greedy selections."""
    strategies_data = {}
    for strat in ["qd_v6", "greedy_v6", "random_v6", "qd_old_desc"]:
        rf = RESULTS_DIR / strat / f"{strat}_results.json"
        if rf.exists():
            data = json.load(open(rf))
            key = f"{strat}_r{round_num}"
            if key in data and data[key]["status"] == "completed":
                strategies_data[strat] = data[key]

    if len(strategies_data) < 2:
        print("Not enough strategies completed for overlap analysis", flush=True)
        return

    print(f"\n{'='*60}", flush=True)
    print(f"SELECTION OVERLAP ANALYSIS (Round {round_num})", flush=True)
    print(f"{'='*60}", flush=True)

    # Compare cell coverage
    for s1 in strategies_data:
        for s2 in strategies_data:
            if s1 >= s2: continue
            cells1 = set(tuple(c) for c in strategies_data[s1].get("archive_cells", []))
            cells2 = set(tuple(c) for c in strategies_data[s2].get("archive_cells", []))
            overlap = cells1 & cells2
            only1 = cells1 - cells2
            only2 = cells2 - cells1
            print(f"  {s1} vs {s2}:", flush=True)
            print(f"    Cells: {len(cells1)} vs {len(cells2)}", flush=True)
            print(f"    Overlap: {len(overlap)} cells ({len(overlap)/max(len(cells1),len(cells2),1)*100:.1f}%)", flush=True)
            print(f"    {s1}-only: {len(only1)} cells, {s2}-only: {len(only2)} cells", flush=True)

    # Entropy comparison
    print(f"\n  Entropy comparison:", flush=True)
    for strat, data in strategies_data.items():
        print(f"    {strat}: {data.get('sel_entropy', 'N/A')}", flush=True)

    # Self-BLEU comparison
    print(f"\n  Self-BLEU comparison (lower = more diverse):", flush=True)
    for strat, data in strategies_data.items():
        print(f"    {strat}: {data.get('sel_self_bleu', 'N/A')}", flush=True)

# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["generate", "train", "all", "analyze"], required=True)
    parser.add_argument("--strategy", type=str, default="qd_v6")
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    STRATEGY = args.strategy
    SEED = args.seed

    if args.phase == "generate":
        rnd = args.round if args.round is not None else 0
        phase_generate(rnd, SEED)

    elif args.phase == "train":
        rnd = args.round if args.round is not None else 0
        phase_train(STRATEGY, rnd, SEED)

    elif args.phase == "all":
        # Full pipeline: generate → train → eval for all rounds
        for rnd in range(N_ROUNDS):
            print(f"\n{'#'*60}", flush=True)
            print(f"# ROUND {rnd}", flush=True)
            print(f"{'#'*60}", flush=True)

            # Check if already completed
            rf = RESULTS_DIR / STRATEGY / f"{STRATEGY}_results.json"
            if rf.exists():
                data = json.load(open(rf))
                key = f"{STRATEGY}_r{rnd}"
                if key in data and data[key]["status"] == "completed":
                    print(f"  {key} already done, skipping", flush=True)
                    continue

            # Generate shared pool (same for all strategies)
            phase_generate(rnd, SEED)

            # Select + Train + Eval
            phase_train(STRATEGY, rnd, SEED)

        # Final analysis
        analyze_selection_overlap(N_ROUNDS - 1)

    elif args.phase == "analyze":
        for rnd in range(N_ROUNDS):
            analyze_selection_overlap(rnd)

if __name__ == "__main__":
    main()
