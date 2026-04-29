"""
GSM8K few-shot evaluation with QD-Synth vs Greedy exemplar selection.
Tests whether QD-selected few-shot examples improve math reasoning.
Uses Qwen2.5-1.5B-Instruct for inference (no fine-tuning needed).
"""
import sys, json, random, re, torch, numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-1___5B-Instruct/"
K = 57  # Number of few-shot exemplars
N_TEST = 200  # Subset of test for stronger statistics
N_SHOT = 4   # Few-shot count per prompt
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OUTPUT_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/gsm8k")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading GSM8K...", flush=True)
gsm8k = load_dataset("gsm8k", "main", split="test")
train = load_dataset("gsm8k", "main", split="train")

# Extract answer from GSM8K format (#### number)
def extract_answer(text):
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '')
    return None

# Compute behavior descriptors for math problems
def compute_descriptors(problem, solution):
    """Simple descriptors: solution length, number of steps, has multiple steps."""
    steps = solution.count('<<') + 1  # Count intermediate calculations
    sol_len = len(solution)
    is_multi_step = 1 if steps >= 3 else 0
    # Difficulty proxy: solution length
    difficulty = min(sol_len / 500.0, 1.0)
    return {
        'difficulty': difficulty,
        'num_steps': min(steps / 10.0, 1.0),
        'is_multi_step': is_multi_step
    }

# Prepare train pool with descriptors
print(f"Computing descriptors for {len(train)} training examples...", flush=True)
pool = []
for ex in train:
    q = ex['question']
    a = ex['answer']
    ans = extract_answer(a)
    if ans is None:
        continue
    desc = compute_descriptors(q, a)
    # Quality proxy: solution completeness (has steps and final answer)
    quality = min(len(a) / 300.0, 1.0)
    pool.append({
        'question': q,
        'answer': a,
        'answer_num': ans,
        'descriptors': desc,
        'quality': quality
    })

print(f"Pool: {len(pool)} examples", flush=True)

# QD Selection: grid-based selection
GRID_RES = 10
DIM = 3

def get_cell(desc):
    """Map descriptor to grid cell."""
    d = desc['difficulty']
    s = desc['num_steps']
    m = desc['is_multi_step']
    return (int(d * GRID_RES), int(s * GRID_RES), int(m * GRID_RES))

# Greedy selection: top-K by quality
greedy_selected = sorted(pool, key=lambda x: x['quality'], reverse=True)[:K]

# QD selection: fill grid cells with highest quality per cell
grid = {}
for item in pool:
    cell = get_cell(item['descriptors'])
    if cell not in grid or item['quality'] > grid[cell]['quality']:
        grid[cell] = item

# Sort cells by quality and take top-K
cells_sorted = sorted(grid.values(), key=lambda x: x['quality'], reverse=True)
qd_selected = cells_sorted[:K]

# Random selection
random_selected = random.sample(pool, K)

print(f"Greedy: {len(greedy_selected)} from {len(set(get_cell(x['descriptors']) for x in greedy_selected))} cells", flush=True)
print(f"QD: {len(qd_selected)} from {len(set(get_cell(x['descriptors']) for x in qd_selected))} cells", flush=True)
print(f"Random: {len(random_selected)} from {len(set(get_cell(x['descriptors']) for x in random_selected))} cells", flush=True)

# Format few-shot prompt
def format_fewshot(exemplars, test_question):
    messages = []
    for ex in exemplars[:N_SHOT]:  # Use N_SHOT few-shot examples
        messages.append({"role": "user", "content": ex['question']})
        messages.append({"role": "assistant", "content": ex['answer']})
    messages.append({"role": "user", "content": test_question})
    return messages

# Load model
print("Loading model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", trust_remote_code=True, dtype=torch.bfloat16
)
model.eval()

# Evaluate
def evaluate_pass_at_k(exemplars, name, n_test=N_TEST):
    correct = 0
    total = 0
    test_subset = list(range(min(n_test, len(gsm8k))))

    for i, idx in enumerate(test_subset):
        test_ex = gsm8k[idx]
        test_q = test_ex['question']
        test_ans = extract_answer(test_ex['answer'])

        messages = format_fewshot(exemplars, test_q)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.0,
                                     do_sample=False, pad_token_id=tokenizer.eos_token_id)

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Extract number from response
        numbers = re.findall(r'-?[\d,]+\.?\d*', response.replace(',', ''))
        pred = numbers[-1] if numbers else None

        if pred and test_ans:
            try:
                if abs(float(pred) - float(test_ans)) < 0.01:
                    correct += 1
            except:
                pass
        total += 1

        if (i + 1) % 10 == 0:
            print(f"  {name}: {i+1}/{total} done, acc={correct/total:.3f}", flush=True)

    acc = correct / total if total > 0 else 0
    return {"name": name, "correct": correct, "total": total, "accuracy": round(acc, 4)}

# Run evaluations
results = {}
for exemplars, name in [(greedy_selected, "greedy_57"), (qd_selected, "qd_57"), (random_selected, "random_57")]:
    print(f"\nEvaluating {name}...", flush=True)
    result = evaluate_pass_at_k(exemplars, name)
    results[name] = result
    print(f"  {name}: {result['accuracy']:.4f} ({result['correct']}/{result['total']})", flush=True)

# Save
out_path = OUTPUT_DIR / "gsm8k_fewshot_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {out_path}", flush=True)
