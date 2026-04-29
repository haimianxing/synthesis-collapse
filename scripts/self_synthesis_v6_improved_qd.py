#!/usr/bin/env python3
"""
Self-Synthesis v6: Improved QD with Better Descriptors + High Selection Pressure
==================================================================================

Fixes identified in v3-v5 analysis:

1. DESCRIPTOR FIX: dim2 (multi_step) was constant, dim1 (steps) was 80% value=10
   → New 3D descriptor with balanced distribution:
     dim0: difficulty (LLM-rated 1-5, binned to 10 levels)
     dim1: operation_types (count of unique arithmetic operations: +,-,*,/)
     dim2: problem_type (word_problem, equation, geometry, comparison, rate)

2. HIGH SELECTION PRESSURE: N_GEN=5000, N_SEL=300 (6% selection rate)
   → Forces QD and Greedy to make very different choices

3. QUALITY VARIANCE: Remove fixed q_min, use actual quality variance
   → Allow quality scores to differentiate samples

4. MULTI-SEED: 3 training seeds per strategy
5. MULTI-DOMAIN EVAL: GSM8K + MATH + SVAMP + ASDiv
6. ARCHIVE-GUIDED GENERATION: Generate for empty cells (50% guided + 50% random)

Usage:
  # Phase 1: Generate from base model
  CUDA_VISIBLE_DEVICES=X python -u self_synthesis_v6.py --phase generate --round 0

  # Phase 2: Select + Train + Eval for one strategy
  CUDA_VISIBLE_DEVICES=X python -u self_synthesis_v6.py --phase train --strategy qd --round 0 --train-seed 42
"""

import os, sys, json, random, re, torch, numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import time, math, argparse

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-7B-Instruct"
GRID_RES = 5  # 5×5×5 = 125 cells max (more manageable than 10×10×10=1000)
N_ROUNDS = 5
N_GENERATE = 5000
N_SELECT = 300
N_EVAL = 500
N_TRAIN_SEEDS = 3
TRAIN_SEEDS = [42, 123, 456]

RESULTS_DIR = "/mnt/data2/zcz/neurIps-emnlp/neurips/results/self_synthesis_v6"

STRATEGIES = ['greedy', 'qd', 'random', 'qd_guided']

def compute_descriptor_improved(question, answer=None):
    """
    Improved 3D behavior descriptor with balanced distribution.

    dim0: difficulty (1-5 scale, mapped to 5 grid levels)
          Based on: number of reasoning steps, presence of multi-step operations
    dim1: operation_diversity (0-4, mapped to 5 grid levels)
          Count of unique operation types: +, -, *, /, comparison
    dim2: problem_type (5 categories)
          Classified by: keywords and structure
    """
    # dim0: Difficulty estimation
    steps = 0
    if answer:
        # Count reasoning lines
        lines = [l.strip() for l in answer.split('\n') if l.strip()]
        steps = len([l for l in lines if any(c.isdigit() for c in l) or '=' in l])

    q_lower = question.lower()
    q_len = len(question.split())

    # Count arithmetic operations
    ops = len(re.findall(r'[+\-*/×÷]', question))
    # Count numbers (more numbers = harder)
    numbers = re.findall(r'\d+\.?\d*', question)
    n_numbers = len(numbers)

    # Difficulty score
    diff_score = 0
    if steps >= 6: diff_score += 2
    elif steps >= 3: diff_score += 1
    if n_numbers >= 5: diff_score += 1
    if ops >= 2: diff_score += 1
    if any(w in q_lower for w in ['each', 'every', 'per', 'ratio', 'fraction', 'percent']):
        diff_score += 1

    dim0 = min(diff_score, 4)  # 0-4 → 5 levels

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

    dim1 = min(len(op_types), 4)  # 0-4 → 5 levels

    # dim2: Problem type
    if any(w in q_lower for w in ['equation', 'solve for', 'find the value', '=']):
        dim2 = 0  # equation
    elif any(w in q_lower for w in ['rate', 'speed', 'per hour', 'per minute', 'mph', 'km/h']):
        dim2 = 1  # rate/speed
    elif any(w in q_lower for w in ['area', 'perimeter', 'volume', 'length', 'width', 'rectangle', 'circle', 'triangle']):
        dim2 = 2  # geometry
    elif any(w in q_lower for w in ['ratio', 'fraction', 'proportion', 'percent']):
        dim2 = 3  # ratio/fraction
    else:
        dim2 = 4  # arithmetic/word problem

    return (dim0, dim1, dim2)

def quality_score(question, answer, correct):
    """More nuanced quality scoring."""
    if not correct:
        return 0.0

    score = 0.5  # Base for correct

    # Reasoning quality (step count)
    steps = len([l for l in answer.split('\n') if l.strip() and any(c.isdigit() for c in l)])
    if steps >= 4: score += 0.2
    elif steps >= 2: score += 0.1

    # Answer clarity
    if '####' in answer or '\\boxed' in answer:
        score += 0.1

    # Explanation quality
    if any(w in answer.lower() for w in ['because', 'therefore', 'since', 'thus']):
        score += 0.1

    # Problem-solving structure
    if 'step' in answer.lower():
        score += 0.1

    return min(score, 1.0)

def select_greedy(samples, n_select):
    """Greedy: select top-n by quality."""
    sorted_samples = sorted(samples, key=lambda x: x['quality'], reverse=True)
    return sorted_samples[:n_select]

def select_qd(samples, n_select, grid_res=GRID_RES):
    """QD: MAP-Elites selection - one best per cell, then fill."""
    # Group by cell
    cells = defaultdict(list)
    for s in samples:
        cells[s['cell']].append(s)

    # Select best from each cell
    selected = []
    for cell, cell_samples in cells.items():
        best = max(cell_samples, key=lambda x: x['quality'])
        selected.append(best)

    # If more needed, add by quality from remaining
    if len(selected) < n_select:
        remaining = [s for s in samples if s not in selected]
        remaining.sort(key=lambda x: x['quality'], reverse=True)
        selected.extend(remaining[:n_select - len(selected)])
    elif len(selected) > n_select:
        # Sort selected by quality and keep top n_select
        selected.sort(key=lambda x: x['quality'], reverse=True)
        selected = selected[:n_select]

    return selected[:n_select]

def select_qd_guided(samples, n_select, archive_cells=None, grid_res=GRID_RES):
    """QD + Archive-Guided: Prioritize empty/rare cells."""
    if archive_cells is None:
        archive_cells = set()

    # Group by cell
    cells = defaultdict(list)
    for s in samples:
        cells[s['cell']].append(s)

    # Phase 1: Fill empty cells first (archive-guided)
    selected = []
    empty_cell_samples = []
    for cell, cell_samples in cells.items():
        if cell not in archive_cells:
            best = max(cell_samples, key=lambda x: x['quality'])
            empty_cell_samples.append(best)

    # Sort empty cell candidates by quality
    empty_cell_samples.sort(key=lambda x: x['quality'], reverse=True)
    selected.extend(empty_cell_samples[:int(n_select * 0.5)])  # 50% for new cells

    # Phase 2: Fill remaining with QD (best per cell)
    remaining_cells = {c: s for c, s in cells.items() if c in archive_cells}
    for cell, cell_samples in remaining_cells.items():
        if len(selected) >= n_select:
            break
        best = max(cell_samples, key=lambda x: x['quality'])
        if best not in selected:
            selected.append(best)

    # Phase 3: Fill by quality
    if len(selected) < n_select:
        remaining = [s for s in samples if s not in selected]
        remaining.sort(key=lambda x: x['quality'], reverse=True)
        selected.extend(remaining[:n_select - len(selected)])

    return selected[:n_select]

def select_random(samples, n_select, seed=42):
    """Random selection baseline."""
    rng = random.Random(seed)
    return rng.sample(samples, min(n_select, len(samples)))

def compute_entropy(items):
    """Compute Shannon entropy of cell distribution."""
    cells = [item['cell'] for item in items]
    counts = Counter(cells)
    total = len(cells)
    if total == 0:
        return 0.0
    probs = [c/total for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)

# ... rest of implementation (generation, training, evaluation) follows same pattern as v5
