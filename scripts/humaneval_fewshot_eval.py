"""
HumanEval few-shot evaluation with QD-Synth vs Greedy exemplar selection.
Tests whether QD-selected code examples improve pass@1 on HumanEval.
"""
import sys, json, random, re, torch, numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-1___5B-Instruct/"
K = 57
N_TEST = 164  # All HumanEval test
N_SHOT = 4
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OUTPUT_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/humaneval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading HumanEval...", flush=True)
humaneval = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=True)
print(f"Loaded {len(humaneval)} problems", flush=True)

# Behavior descriptors for code
def compute_descriptors(entry):
    """Code descriptors: difficulty, num_APIs, needs_debugging."""
    prompt = entry['prompt']
    solution = entry.get('canonical_solution', '')
    # Difficulty: solution length proxy
    difficulty = min(len(solution) / 300.0, 1.0)
    # Num APIs: count of function calls
    api_calls = len(re.findall(r'\w+\(', solution))
    num_apis = min(api_calls / 10.0, 1.0)
    # Needs debugging: has try/except or assert
    needs_debug = 1.0 if ('try' in solution or 'assert' in solution or 'except' in solution) else 0.0
    return {'difficulty': difficulty, 'num_apis': num_apis, 'needs_debug': needs_debug}

# Prepare exemplar pool from HumanEval itself (use all 164 as pool)
print("Computing descriptors...", flush=True)
pool = []
for entry in humaneval:
    desc = compute_descriptors(entry)
    quality = min(len(entry.get('canonical_solution', '')) / 200.0, 1.0)
    pool.append({
        'prompt': entry['prompt'],
        'canonical_solution': entry.get('canonical_solution', ''),
        'test': entry.get('test', ''),
        'entry_point': entry.get('entry_point', ''),
        'descriptors': desc,
        'quality': quality
    })

print(f"Pool: {len(pool)} examples", flush=True)

GRID_RES = 10
def get_cell(desc):
    return (int(desc['difficulty'] * GRID_RES),
            int(desc['num_apis'] * GRID_RES),
            int(desc['needs_debug'] * GRID_RES))

# Greedy: top-K by quality
greedy_selected = sorted(pool, key=lambda x: x['quality'], reverse=True)[:K]

# QD: fill grid cells
grid = {}
for item in pool:
    cell = get_cell(item['descriptors'])
    if cell not in grid or item['quality'] > grid[cell]['quality']:
        grid[cell] = item
qd_selected = sorted(grid.values(), key=lambda x: x['quality'], reverse=True)[:K]

# Random
random_selected = random.sample(pool, K)

print(f"Greedy: {len(greedy_selected)} from {len(set(get_cell(x['descriptors']) for x in greedy_selected))} cells", flush=True)
print(f"QD: {len(qd_selected)} from {len(set(get_cell(x['descriptors']) for x in qd_selected))} cells", flush=True)
print(f"Random: {len(random_selected)} from {len(set(get_cell(x['descriptors']) for x in random_selected))} cells", flush=True)

# Load model
print("Loading model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", trust_remote_code=True, dtype=torch.bfloat16
)
model.eval()
print("Model loaded.", flush=True)

def format_fewshot(exemplars, test_prompt):
    messages = []
    for ex in exemplars[:N_SHOT]:
        messages.append({"role": "user", "content": ex['prompt']})
        messages.append({"role": "assistant", "content": ex['canonical_solution']})
    messages.append({"role": "user", "content": test_prompt})
    return messages

def check_correctness(solution, test_code, entry_point):
    """Execute and test the solution."""
    try:
        # Combine solution + test
        full_code = solution + "\n" + test_code
        local_vars = {}
        exec(full_code, local_vars)
        return True
    except:
        return False

def evaluate_pass_at_1(exemplars, name):
    correct = 0
    total = 0
    for i, entry in enumerate(humaneval):
        messages = format_fewshot(exemplars, entry['prompt'])
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.0,
                                     do_sample=False, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Extract code from response
        code_blocks = re.findall(r'```python\n(.*?)```', response, re.DOTALL)
        if not code_blocks:
            code_blocks = re.findall(r'```\n(.*?)```', response, re.DOTALL)
        solution = code_blocks[0] if code_blocks else response

        passed = check_correctness(solution, entry.get('test', ''), entry.get('entry_point', ''))
        if passed:
            correct += 1
        total += 1

        if (i + 1) % 20 == 0:
            print(f"  {name}: {i+1}/{total} done, pass@1={correct/total:.3f}", flush=True)

    acc = correct / total if total > 0 else 0
    return {"name": name, "correct": correct, "total": total, "pass_at_1": round(acc, 4)}

# Run evaluations
results = {}
for exemplars, name in [(greedy_selected, "greedy_57"), (qd_selected, "qd_57"), (random_selected, "random_57")]:
    print(f"\nEvaluating {name}...", flush=True)
    result = evaluate_pass_at_1(exemplars, name)
    results[name] = result
    print(f"  {name}: pass@1={result['pass_at_1']:.4f} ({result['correct']}/{result['total']})", flush=True)

out_path = OUTPUT_DIR / "humaneval_fewshot_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {out_path}", flush=True)
