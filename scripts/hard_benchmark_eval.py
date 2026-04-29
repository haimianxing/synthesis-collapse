"""
Hard Benchmark Evaluation: MBPP Test Hard Subset
===================================================
Evaluates V7 base-reset LoRA adapters on hard MBPP problems.
V7 models are saved as LoRA adapters (not merged), so we load
base model + PEFT adapter for each evaluation.

V7 models: QD/Greedy/Simple-Dedup x 2 seeds x 4 rounds = 24 checkpoints
Eval: MBPP test set filtered to problems where base model fails (hard subset)
+ HumanEval (for reference)

GPU 6 -- no training needed, just inference.
"""
import os, re, torch, json, subprocess, time
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-7B-Instruct"
V7_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/self_synthesis_v7_code")
RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/hard_benchmark")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEQ_LENGTH = 512
MAX_CODE_TOKENS = 512

# Load eval datasets
mbpp_test = load_dataset('mbpp', 'sanitized', split='test')
humaneval = load_dataset('openai/openai_humaneval', split='test')

print(f"=== Hard Benchmark Evaluation ===", flush=True)
print(f"  MBPP test: {len(mbpp_test)} problems", flush=True)
print(f"  HumanEval: {len(humaneval)} problems", flush=True)

def execute_code(code_str, test_cases, timeout=5):
    for test in test_cases:
        try:
            full = code_str + "\n" + test
            r = subprocess.run(['python3', '-c', full], timeout=timeout,
                              capture_output=True, text=True)
            if r.returncode != 0:
                return False
        except:
            return False
    return True

def load_base_model():
    """Load and cache the base model (shared across evaluations)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )
    return model, tokenizer

def load_adapter_model(adapter_path):
    """Load base model + LoRA adapter for V7 checkpoints."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.eval()
    return model, tokenizer

def eval_model_on_dataset(model, tokenizer, dataset, name, is_humaneval=False):
    """Evaluate a model on a dataset."""
    print(f"  Evaluating {name}...", flush=True)
    model.eval()

    correct = 0; total = 0
    results = []
    for idx, item in enumerate(dataset):
        if is_humaneval:
            prompt = item['prompt']
            test_code = item['test']
            msgs = [
                {"role": "system", "content": "Complete the Python function."},
                {"role": "user", "content": prompt}
            ]
        else:
            prompt = item['prompt']
            test_list = item['test_list']
            msgs = [
                {"role": "system", "content": "Write a Python function to solve the problem."},
                {"role": "user", "content": f"Write a Python function:\n\n{prompt}\n\nProvide the implementation:"}
            ]

        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=MAX_SEQ_LENGTH).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=MAX_CODE_TOKENS,
                                    do_sample=False, pad_token_id=tokenizer.eos_token_id)
        completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)

        if is_humaneval:
            full_code = prompt + completion
            passed = execute_code(full_code, [test_code])
        else:
            code_blocks = re.findall(r'```python\s*(.*?)```', completion, re.DOTALL)
            code = code_blocks[0] if code_blocks else completion
            passed = execute_code(code, test_list)

        if passed: correct += 1
        total += 1
        results.append({'idx': idx, 'passed': passed, 'prompt': prompt[:100]})

        if (idx + 1) % 50 == 0:
            print(f"    {idx+1}/{total} pass={correct}/{total} ({correct/total:.4f})", flush=True)

    acc = correct / total if total > 0 else 0
    print(f"  {name}: {acc:.4f} ({correct}/{total})", flush=True)
    return acc, correct, total, results

# Step 1: Identify hard MBPP problems (where base model fails)
print(f"\n--- Step 1: Find hard MBPP problems ---", flush=True)
base_model, base_tokenizer = load_base_model()
base_acc, base_corr, base_total, base_results = eval_model_on_dataset(
    base_model, base_tokenizer, mbpp_test, "base_model_mbpp"
)

hard_indices = [r['idx'] for r in base_results if not r['passed']]
print(f"\n  Base model fails {len(hard_indices)}/{base_total} MBPP problems", flush=True)

# Save base results
with open(RESULTS_DIR / "base_mbpp_results.json", 'w') as f:
    json.dump({'accuracy': base_acc, 'correct': base_corr, 'total': base_total,
               'n_hard': len(hard_indices), 'hard_indices': hard_indices}, f, indent=2)

# Free base model memory
del base_model; torch.cuda.empty_cache()

# Create hard dataset
hard_mbpp = mbpp_test.select(hard_indices)
print(f"  Hard MBPP subset: {len(hard_mbpp)} problems", flush=True)

# Step 2: Evaluate all V7 adapter models on hard MBPP + HumanEval
print(f"\n--- Step 2: Evaluate V7 adapter models ---", flush=True)
all_results = {}

# Base model on hard subset (already computed: 0 by definition)
all_results['base'] = {
    'hard_mbpp': 0.0,
    'hard_correct': 0,
    'hard_total': len(hard_indices),
    'full_mbpp': base_acc,
    'label': 'base_model'
}
print(f"  base on hard: 0/{len(hard_indices)} (by construction)", flush=True)

# Evaluate each V7 adapter
models_to_eval = []
for strategy in ['qd', 'greedy', 'simple_dedup']:
    for seed in [42, 123]:
        for rnd in [0, 3]:  # R0 (start) and R3 (end) -- most informative contrast
            name = f"{strategy}_s{seed}"
            adapter_path = V7_DIR / name / f"merged_{name}_r{rnd}"
            if adapter_path.exists():
                models_to_eval.append({
                    'strategy': strategy, 'seed': seed, 'round': rnd,
                    'path': str(adapter_path),
                    'label': f"{strategy}_s{seed}_r{rnd}"
                })

print(f"  Found {len(models_to_eval)} V7 adapter models to evaluate", flush=True)

for m in models_to_eval:
    label = m['label']
    print(f"\n  === {label} ===", flush=True)

    # Load adapter
    model, tokenizer = load_adapter_model(m['path'])

    # Hard MBPP
    h_acc, hc, ht, _ = eval_model_on_dataset(model, tokenizer, hard_mbpp, f"{label}_hard")
    # HumanEval
    he_acc, hec, het, _ = eval_model_on_dataset(model, tokenizer, humaneval, f"{label}_humaneval",
                                                  is_humaneval=True)

    all_results[label] = {
        'strategy': m['strategy'], 'seed': m['seed'], 'round': m['round'],
        'hard_mbpp': h_acc, 'hard_correct': hc, 'hard_total': ht,
        'humaneval': he_acc, 'he_correct': hec, 'he_total': het,
    }

    with open(RESULTS_DIR / "hard_benchmark_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    del model; torch.cuda.empty_cache()

# Step 3: Summary
print(f"\n{'='*60}", flush=True)
print(f"HARD BENCHMARK RESULTS", flush=True)
print(f"{'='*60}", flush=True)
print(f"\nHard MBPP subset: {len(hard_mbpp)} problems (base model fails these)", flush=True)
print(f"\n  {'Model':<30} {'Hard MBPP':>10} {'HumanEval':>10}", flush=True)
print(f"  {'-'*52}", flush=True)

# Base
print(f"  {'base_model':<30} {'0.0000':>10} {'--':>10}", flush=True)

# Per strategy, average across seeds
for strategy in ['qd', 'greedy', 'simple_dedup']:
    for rnd in [0, 3]:
        key_matches = [k for k, v in all_results.items()
                      if isinstance(v, dict) and v.get('strategy') == strategy
                      and v.get('round') == rnd]
        if key_matches:
            avg_hard = sum(all_results[k]['hard_mbpp'] for k in key_matches) / len(key_matches)
            avg_he = sum(all_results[k]['humaneval'] for k in key_matches) / len(key_matches)
            print(f"  {strategy}_R{rnd}{'':<25} {avg_hard:>10.4f} {avg_he:>10.4f}", flush=True)

print(f"\nResults saved to {RESULTS_DIR}", flush=True)
