"""
Descriptor Sensitivity Analysis
Tests whether QD-Synth's advantage depends on specific descriptor choice.

Design: Run QD iterative with 3 alternative descriptor sets for Code domain:
1. Default: (difficulty, num_APIs, needs_debugging) - current
2. Length-based: (code_length, function_count, class_usage)
3. Random projection: (random_1, random_2, random_3) - random features

If QD's advantage persists across descriptor sets → advantage is from grid mechanism,
not specific descriptor engineering. If advantage disappears with random descriptors →
descriptors must capture meaningful structure.

Uses existing code_iterative_v2 archives (doesn't regenerate API data).
Re-evaluates archives with alternative descriptors to compute coverage.
"""
import os, json, re, ast, random, numpy as np
from pathlib import Path
from itertools import product

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from datasets import load_dataset

GRID_RES = 10
RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/descriptor_sensitivity")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=== Descriptor Sensitivity Analysis ===", flush=True)

# Load MBPP
print("Loading MBPP (cached)...", flush=True)
try:
    mbpp = load_dataset("mbpp", "sanitized", split="test")
except:
    mbpp = load_dataset("mbpp", split="test")
print(f"MBPP: {len(mbpp)}", flush=True)

# Load existing archives
ITER_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/code_iterative_v2")

def load_archive(strategy, rnd):
    path = ITER_DIR / f"{strategy}_archive_r{rnd}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

# ============ Alternative descriptor functions ============

def desc_default(code, prompt=""):
    """Original descriptors: difficulty, num_APIs, needs_debugging"""
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
    return {'d1': difficulty, 'd2': min(api_count / 10.0, 1.0), 'd3': has_debug}

def desc_length(code, prompt=""):
    """Length-based: code_length, function_count, class_usage"""
    code_len = len(code) if code else 100
    func_count = 0
    class_count = 0
    try:
        tree = ast.parse(code) if code else None
        if tree:
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef): func_count += 1
                elif isinstance(node, ast.ClassDef): class_count += 1
    except:
        func_count = len(re.findall(r'\bdef\s+\w+', code)) if code else 0
        class_count = len(re.findall(r'\bclass\s+\w+', code)) if code else 0
    return {'d1': min(code_len / 2000.0, 1.0), 'd2': min(func_count / 5.0, 1.0), 'd3': min(class_count / 3.0, 1.0)}

def desc_syntactic(code, prompt=""):
    """Syntactic: nesting_depth, loop_count, string_count"""
    if not code:
        return {'d1': 0, 'd2': 0, 'd3': 0}
    max_nesting = 0
    current_nesting = 0
    loop_count = 0
    string_count = 0
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While, ast.If, ast.With)):
                loop_count += 1
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                string_count += 1
    except:
        loop_count = len(re.findall(r'\b(for|while|if)\b', code))
        string_count = len(re.findall(r'["\']', code)) // 2
    return {'d1': min(max_nesting / 5.0, 1.0), 'd2': min(loop_count / 10.0, 1.0), 'd3': min(string_count / 20.0, 1.0)}

def desc_random(code, prompt="", seed=42):
    """Random projection: hash-based random features"""
    rng = np.random.RandomState(hash(code) % 2**31 if code else seed)
    return {'d1': rng.random(), 'd2': rng.random(), 'd3': rng.random()}

DESCRIPTOR_SETS = {
    'default': desc_default,
    'length': desc_length,
    'syntactic': desc_syntactic,
}

# ============ Re-evaluate archives with alternative descriptors ============

results = {}

for desc_name, desc_fn in DESCRIPTOR_SETS.items():
    print(f"\n--- Descriptor: {desc_name} ---", flush=True)
    desc_results = {'greedy': {}, 'qd': {}}

    for strategy in ['greedy', 'qd']:
        for rnd in range(8):
            archive = load_archive(strategy, rnd)
            if archive is None:
                continue

            # Compute cells using alternative descriptors
            grid = {}
            for item in archive:
                code = item.get('code', '')
                prompt = item.get('prompt', '')
                desc = desc_fn(code, prompt)
                cell = (int(desc['d1'] * GRID_RES), int(desc['d2'] * GRID_RES), int(desc['d3'] * GRID_RES))
                if cell not in grid:
                    grid[cell] = item

            n_cells = len(grid)
            total_cells = GRID_RES ** 3
            coverage = n_cells / total_cells

            desc_results[strategy][rnd] = {
                'round': rnd,
                'n_cells': n_cells,
                'coverage': round(coverage, 4),
                'n_items': len(archive)
            }

            print(f"  {strategy.upper()} R{rnd}: cells={n_cells}, cov={coverage:.4f}, items={len(archive)}", flush=True)

    results[desc_name] = desc_results

    # Compute greedy freeze vs QD growth
    greedy_final = max(desc_results['greedy'].keys()) if desc_results['greedy'] else 0
    qd_final = max(desc_results['qd'].keys()) if desc_results['qd'] else 0
    if greedy_final in desc_results['greedy'] and qd_final in desc_results['qd']:
        g_cells = desc_results['greedy'][greedy_final]['n_cells']
        q_cells = desc_results['qd'][qd_final]['n_cells']
        ratio = q_cells / max(g_cells, 1)
        print(f"  → Final: Greedy={g_cells}, QD={q_cells}, QD/Greedy={ratio:.2f}x", flush=True)

# ============ Random descriptor baseline ============
print(f"\n--- Random Descriptors (hash-based) ---", flush=True)
random.seed(42)
np.random.seed(42)

random_results = {'greedy': {}, 'qd': {}}
for strategy in ['greedy', 'qd']:
    for rnd in range(8):
        archive = load_archive(strategy, rnd)
        if archive is None:
            continue

        # Random hash-based cells
        grid = {}
        for item in archive:
            code = item.get('code', '')
            desc = desc_random(code)
            cell = (int(desc['d1'] * GRID_RES), int(desc['d2'] * GRID_RES), int(desc['d3'] * GRID_RES))
            if cell not in grid:
                grid[cell] = item

        n_cells = len(grid)
        total_cells = GRID_RES ** 3
        coverage = n_cells / total_cells

        random_results[strategy][rnd] = {
            'round': rnd,
            'n_cells': n_cells,
            'coverage': round(coverage, 4)
        }

        print(f"  {strategy.upper()} R{rnd}: cells={n_cells}, cov={coverage:.4f}", flush=True)

results['random_hash'] = random_results

# ============ Analysis ============
print(f"\n{'='*60}", flush=True)
print("DESCRIPTOR SENSITIVITY SUMMARY", flush=True)
print(f"{'='*60}", flush=True)

for desc_name, desc_data in results.items():
    print(f"\n  {desc_name}:", flush=True)
    for strategy in ['greedy', 'qd']:
        rounds = sorted(desc_data[strategy].keys())
        if rounds:
            cells = [desc_data[strategy][r]['n_cells'] for r in rounds]
            growth = (cells[-1] - cells[0]) / max(cells[0], 1) * 100
            print(f"    {strategy.upper()}: {cells[0]}→{cells[-1]} cells ({growth:+.0f}%)", flush=True)

# Key question: does the greedy freeze pattern persist across descriptors?
print(f"\n  Key Question: Does Greedy freeze across ALL descriptor sets?", flush=True)
for desc_name in ['default', 'length', 'syntactic']:
    desc_data = results[desc_name]
    greedy_cells_by_round = [desc_data['greedy'].get(r, {}).get('n_cells', 0) for r in range(8)]
    # Check if cells stop growing after R3
    if len(greedy_cells_by_round) > 4:
        growth_after_r3 = greedy_cells_by_round[min(4, len(greedy_cells_by_round)-1)] - greedy_cells_by_round[min(3, len(greedy_cells_by_round)-1)]
        print(f"    {desc_name}: growth after R3 = {growth_after_r3} cells", flush=True)

# Save
with open(RESULTS_DIR / "descriptor_sensitivity.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults: {RESULTS_DIR}", flush=True)
