#!/usr/bin/env python3
"""
Per-Problem HumanEval Evaluation for V7 Code models.
Records pass/fail per problem, maps to code descriptor cells,
and identifies which cells are failure-prone for each strategy.

Usage: CUDA_VISIBLE_DEVICES=3 python -u eval_per_problem_humaneval.py
"""
import os, json, re, torch, tempfile, subprocess, time
import numpy as np
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from collections import defaultdict

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-7B-Instruct"
RESULTS_BASE = "/mnt/data2/zcz/neurIps-emnlp/neurips/results/self_synthesis_v7_code"
GRID_RES = 5

# Models to evaluate
MODELS = {
    "base": MODEL_PATH,
    "qd_s42_r0": f"{RESULTS_BASE}/qd_s42/merged_qd_s42_r0",
    "qd_s42_r1": f"{RESULTS_BASE}/qd_s42/merged_qd_s42_r1",
    "greedy_s42_r0": f"{RESULTS_BASE}/greedy_s42/merged_greedy_s42_r0",
    "greedy_s42_r1": f"{RESULTS_BASE}/greedy_s42/merged_greedy_s42_r1",
}

# Try to load more R1 models if available
for name in ["simple_dedup_s42_r0", "simple_dedup_s42_r1"]:
    strat_seed = name.rsplit("_r", 1)[0]
    rnd = name.rsplit("_r", 1)[1]
    path = f"{RESULTS_BASE}/{strat_seed}/merged_{strat_seed}_r{rnd}"
    if os.path.exists(path):
        MODELS[name] = path

OUTPUT_FILE = f"{RESULTS_BASE}/per_problem_analysis.json"

def code_descriptor_complexity(code):
    """Estimate code complexity (0-4 scale)."""
    lines = code.strip().split('\n')
    n_lines = len(lines)
    n_loops = len(re.findall(r'\b(for|while)\b', code))
    n_ifs = len(re.findall(r'\bif\b', code))
    n_ops = n_loops * 2 + n_ifs
    complexity = min(4, max(0, n_ops))
    return complexity

def code_descriptor_algorithm(code):
    """Estimate algorithm type (0-4 scale)."""
    code_lower = code.lower()
    if any(x in code_lower for x in ['sort', 'sorted']):
        return 0  # sorting
    elif any(x in code_lower for x in ['search', 'find', 'index', 'count']):
        return 1  # search
    elif any(x in code_lower for x in ['math', 'sum', 'abs', 'min', 'max', 'sqrt']):
        return 2  # math
    elif any(x in code_lower for x in ['string', 'str', 'split', 'join', 'replace']):
        return 3  # string
    else:
        return 4  # other

def code_descriptor_io(code):
    """Estimate I/O type (0-4 scale)."""
    code_lower = code.lower()
    if 'list' in code_lower or '[' in code:
        return 0  # list
    elif 'dict' in code_lower or '{' in code:
        return 1  # dict
    elif any(x in code_lower for x in ['int', 'float', 'bool']):
        return 2  # numeric
    elif any(x in code_lower for x in ['str', 'string']):
        return 3  # string
    else:
        return 4  # mixed

def get_code_cell(prompt):
    """Get 3D cell for a HumanEval problem."""
    c = code_descriptor_complexity(prompt)
    a = code_descriptor_algorithm(prompt)
    io = code_descriptor_io(prompt)
    return (c, a, io)

def execute_code_safely(code, test_code, timeout=5):
    """Execute code with test and return (passed, n_tests)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code + '\n' + test_code)
        f.flush()
        fname = f.name
    try:
        result = subprocess.run(
            ['python3', fname],
            capture_output=True, text=True, timeout=timeout
        )
        os.unlink(fname)
        return 1 if result.returncode == 0 else 0
    except (subprocess.TimeoutExpired, Exception):
        try:
            os.unlink(fname)
        except:
            pass
        return 0

def evaluate_model_per_problem(model_path, humaneval_data, model_name):
    """Evaluate model on HumanEval, recording per-problem results."""
    print(f"  Loading {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )
    model.eval()

    results = []
    for idx, item in enumerate(humaneval_data):
        prompt = item['prompt']
        entry_point = item['entry_point']
        test = item['test']
        cell = get_code_cell(prompt)

        msgs = [
            {"role": "system", "content": "Complete the Python function. Write only the function body."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        full_code = prompt + completion

        passed = execute_code_safely(full_code, test)
        results.append({
            'problem_idx': idx,
            'entry_point': entry_point,
            'cell': list(cell),
            'passed': passed,
        })

        if (idx + 1) % 40 == 0:
            n_pass = sum(r['passed'] for r in results)
            print(f"    {model_name}: {idx+1}/{len(humaneval_data)}, pass@1={n_pass/(idx+1):.4f}", flush=True)

    del model
    torch.cuda.empty_cache()

    n_pass = sum(r['passed'] for r in results)
    print(f"  {model_name}: FINAL pass@1={n_pass/len(results):.4f} ({n_pass}/{len(results)})", flush=True)
    return results

def analyze_per_cell(all_results, humaneval_data):
    """Analyze failures by cell across models."""
    # Map problems to cells
    problem_cells = {}
    for i, item in enumerate(humaneval_data):
        problem_cells[i] = get_code_cell(item['prompt'])

    # Group by cell
    cell_stats = defaultdict(lambda: defaultdict(list))
    for model_name, results in all_results.items():
        for r in results:
            cell = tuple(r['cell'])
            cell_stats[cell][model_name].append(r['passed'])

    # Find cells where Greedy fails but QD passes
    print("\n=== CELL-LEVEL ANALYSIS ===")
    print(f"{'Cell':<15} {'Problems':>8} {'Base':>6} {'QD_R1':>6} {'Greedy_R1':>10} {'Gap':>6}")
    print("-" * 60)

    interesting_cells = []
    for cell in sorted(cell_stats.keys()):
        problems = len(cell_stats[cell].get('base', []))
        if problems == 0:
            continue
        base_rate = np.mean(cell_stats[cell].get('base', [0]))
        qd_rate = np.mean(cell_stats[cell].get('qd_s42_r1', [0]))
        greedy_rate = np.mean(cell_stats[cell].get('greedy_s42_r1', [0]))
        gap = qd_rate - greedy_rate

        if abs(gap) > 0.01:  # Only show cells with a gap
            print(f"  {str(cell):<13} {problems:>8} {base_rate:>5.0%} {qd_rate:>5.0%} {greedy_rate:>9.0%} {gap:>+5.0%}")
            interesting_cells.append({
                'cell': list(cell),
                'n_problems': problems,
                'base_rate': base_rate,
                'qd_rate': qd_rate,
                'greedy_rate': greedy_rate,
                'gap': gap,
            })

    return interesting_cells

def main():
    print("=== Per-Problem HumanEval Analysis ===", flush=True)

    # Load HumanEval
    humaneval = load_dataset("openai/openai_humaneval", "openai_humaneval", split="test")
    print(f"Loaded {len(humaneval)} HumanEval problems", flush=True)

    # Evaluate all models
    all_results = {}
    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            print(f"  Skipping {model_name}: path not found", flush=True)
            continue
        results = evaluate_model_per_problem(model_path, humaneval, model_name)
        all_results[model_name] = results

    # Per-cell analysis
    interesting = analyze_per_cell(all_results, humaneval)

    # Save results
    output = {
        'models_evaluated': list(all_results.keys()),
        'per_problem': {name: results for name, results in all_results.items()},
        'interesting_cells': interesting,
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {OUTPUT_FILE}", flush=True)

if __name__ == "__main__":
    main()
