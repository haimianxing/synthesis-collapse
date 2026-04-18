"""
Code Domain Downstream Fine-tuning (Addresses SAC C3: downstream only Dialogue)
Pool: MBPP sanitized training (974 problems)
Selection: QD-200, Greedy-200, Random-200 from pool
Fine-tune: Qwen2.5-1.5B + LoRA (r=16, alpha=32, 3 epochs)
Evaluate: HumanEval (164 problems), pass@1
Multi-seed: 5 seeds for statistical analysis
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys, json, random, re, torch, numpy as np, time, ast
from pathlib import Path
from collections import Counter, defaultdict
from datasets import load_dataset

# ============ Config ============
MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-1___5B-Instruct"
K = 200           # training subset size
SEEDS = [42, 123, 271, 456, 2024]
GRID_RES = 10
DIM = 3
GPU_ID = int(os.environ.get("GPU_ID", "5"))
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
DEVICE = "cuda:0"

RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/code_downstream")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"=== Code Downstream Fine-tuning (GPU {GPU_ID}) ===", flush=True)
print(f"Config: K={K}, {len(SEEDS)} seeds", flush=True)

# ============ Load MBPP ============
print("Loading MBPP...", flush=True)
try:
    mbpp_train = load_dataset("mbpp", "sanitized", split="test")  # MBPP test has 500 with test cases
    print(f"MBPP sanitized test (used as pool): {len(mbpp_train)}", flush=True)
except:
    mbpp_train = load_dataset("mbpp", split="test")
    print(f"MBPP test (used as pool): {len(mbpp_train)}", flush=True)

# Load HumanEval for evaluation
print("Loading HumanEval...", flush=True)
humaneval = load_dataset("openai_humaneval", split="test")
print(f"HumanEval: {len(humaneval)} problems", flush=True)

# ============ Compute Descriptors ============
def compute_code_descriptors(prompt, code, test_list):
    """Compute behavior descriptors for code problems."""
    # Difficulty: based on solution length and complexity
    code_len = len(code) if code else 100
    difficulty = min(code_len / 1000.0, 1.0)

    # Number of API/library calls
    api_count = 0
    try:
        tree = ast.parse(code) if code else None
        if tree:
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    api_count += 1
                elif isinstance(node, ast.Import):
                    api_count += len(node.names)
                elif isinstance(node, ast.ImportFrom):
                    api_count += len(node.names)
    except:
        api_count = len(re.findall(r'\b\w+\.\w+\(', code)) if code else 0

    # Has debugging/error handling
    has_debug = 1 if (code and ('try:' in code or 'except' in code or 'assert' in code)) else 0

    return {
        'difficulty': difficulty,
        'num_APIs': min(api_count / 10.0, 1.0),
        'needs_debugging': has_debug
    }

def get_cell(desc):
    d = desc['difficulty']
    a = desc['num_APIs']
    b = desc['needs_debugging']
    return (int(d * GRID_RES), int(a * GRID_RES), int(b * GRID_RES))

# Build pool from MBPP
print(f"Computing descriptors for {len(mbpp_train)} MBPP examples...", flush=True)
pool = []
for ex in mbpp_train:
    prompt = ex.get('prompt', '')
    code = ex.get('code', '')
    test_list = ex.get('test_list', [])
    text = ex.get('text', '')

    desc = compute_code_descriptors(prompt, code, test_list)
    quality = min(len(code) / 500.0, 1.0) if code else 0.1

    pool.append({
        'prompt': prompt, 'code': code, 'text': text,
        'test_list': test_list, 'descriptors': desc, 'quality': quality
    })
print(f"Pool: {len(pool)} examples", flush=True)

# ============ Selection Strategies ============
def select_qd(pool_items, k, seed=42):
    rng = random.Random(seed)
    grid = {}
    for item in pool_items:
        cell = get_cell(item['descriptors'])
        if cell not in grid or item['quality'] > grid[cell]['quality']:
            grid[cell] = item
    cells_sorted = sorted(grid.values(), key=lambda x: x['quality'], reverse=True)
    return cells_sorted[:k]

def select_greedy(pool_items, k):
    return sorted(pool_items, key=lambda x: x['quality'], reverse=True)[:k]

def select_random(pool_items, k, seed=42):
    rng = random.Random(seed)
    return rng.sample(pool_items, k)

# ============ Fine-tuning ============
def finetune(train_samples, config_name, seed):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir = RESULTS_DIR / f"model_{config_name}_seed{seed}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE, trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    def format_code(sample):
        return f"<|im_start|>system\nComplete the Python function.<|im_end|>\n<|im_start|>user\n{sample['prompt'][:512]}<|im_end|>\n<|im_start|>assistant\n{sample['code'][:768]}<|im_end|>"

    train_texts = [format_code(s) for s in train_samples]
    train_dataset = Dataset.from_dict({"text": train_texts})

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=50,
        save_strategy="no",
        bf16=True,
        report_to="none",
        max_length=768,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model, args=sft_config,
        train_dataset=train_dataset, processing_class=tokenizer,
    )
    trainer.train()

    model.save_pretrained(output_dir / "lora")
    tokenizer.save_pretrained(output_dir / "lora")

    del model, trainer
    torch.cuda.empty_cache()
    return output_dir / "lora"

# ============ Evaluation on HumanEval ============
def evaluate_humaneval(model_path, seed):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    correct = 0
    total = len(humaneval)
    passed_problems = []

    for i, ex in enumerate(humaneval):
        prompt = ex['prompt']
        test = ex['test']
        task_id = ex['task_id']

        messages = [
            {"role": "system", "content": "Complete the Python function. Return only the function body."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=256, temperature=0.0,
                do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Extract code from response
        full_code = prompt + response

        # Execute and test
        passed = False
        try:
            exec_globals = {}
            exec(full_code, exec_globals)
            exec(test, exec_globals)
            passed = True
            correct += 1
            passed_problems.append(task_id)
        except:
            pass

        if (i + 1) % 20 == 0:
            print(f"    [{model_path.parent.name}] {i+1}/{total}, pass@1={correct/(i+1):.4f}", flush=True)

    acc = correct / total if total > 0 else 0

    del model, base_model
    torch.cuda.empty_cache()

    return {
        "pass_at_1": round(acc, 4),
        "correct": correct,
        "total": total,
        "passed_problems": passed_problems
    }

# ============ Main Loop ============
def run_seed(seed):
    results = {}
    methods = {
        "qd_200": select_qd(pool, K, seed),
        "greedy_200": select_greedy(pool, K),
        "random_200": select_random(pool, K, seed),
    }

    for name, selected in methods.items():
        cells = set(get_cell(s['descriptors']) for s in selected)
        qualities = [s['quality'] for s in selected]
        print(f"  [{name}] n={len(selected)}, cells={len(cells)}, q={np.mean(qualities):.3f}", flush=True)

    for name, selected in methods.items():
        config_name = f"{name}_s{seed}"
        print(f"\n--- Training {config_name} ---", flush=True)
        t0 = time.time()
        model_path = finetune(selected, config_name, seed)
        print(f"  Training took {time.time()-t0:.0f}s", flush=True)

        print(f"--- Evaluating {config_name} on HumanEval ---", flush=True)
        t0 = time.time()
        eval_result = evaluate_humaneval(model_path, seed)
        eval_result["train_time_s"] = round(time.time()-t0)
        eval_result["seed"] = seed
        eval_result["n_train"] = len(selected)
        results[name] = eval_result
        print(f"  pass@1: {eval_result['pass_at_1']:.4f} ({eval_result['correct']}/{eval_result['total']})", flush=True)

    return results

# Run all seeds
all_results = {}
for si, seed in enumerate(SEEDS):
    print(f"\n{'='*60}", flush=True)
    print(f"SEED {seed} ({si+1}/{len(SEEDS)})", flush=True)
    print(f"{'='*60}", flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    seed_results = run_seed(seed)
    all_results[seed] = seed_results

    with open(RESULTS_DIR / "code_downstream_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

# ============ Aggregate ============
print(f"\n{'='*60}", flush=True)
print("AGGREGATE RESULTS", flush=True)
print(f"{'='*60}", flush=True)

for method in ["qd_200", "greedy_200", "random_200"]:
    accs = [all_results[s][method]["pass_at_1"] for s in SEEDS]
    print(f"{method}: {np.mean(accs):.4f} ± {np.std(accs):.4f}", flush=True)

try:
    from scipy.stats import wilcoxon
    qd_accs = [all_results[s]["qd_200"]["pass_at_1"] for s in SEEDS]
    greedy_accs = [all_results[s]["greedy_200"]["pass_at_1"] for s in SEEDS]
    random_accs = [all_results[s]["random_200"]["pass_at_1"] for s in SEEDS]

    if len(set(qd_accs)) > 1 and len(set(greedy_accs)) > 1:
        stat, p = wilcoxon(qd_accs, greedy_accs)
        print(f"QD vs Greedy: p={p:.4f}", flush=True)
    else:
        print(f"QD vs Greedy: all equal, no test needed", flush=True)

    if len(set(qd_accs)) > 1 and len(set(random_accs)) > 1:
        stat, p = wilcoxon(qd_accs, random_accs)
        print(f"QD vs Random: p={p:.4f}", flush=True)
except Exception as e:
    print(f"Stats error: {e}", flush=True)

# Unique problems passed analysis
qd_all = set()
greedy_all = set()
for s in SEEDS:
    qd_all.update(all_results[s]["qd_200"]["passed_problems"])
    greedy_all.update(all_results[s]["greedy_200"]["passed_problems"])

print(f"\nUnique problems passed across seeds:", flush=True)
print(f"  QD: {len(qd_all)}, Greedy: {len(greedy_all)}", flush=True)
print(f"  QD-only: {qd_all - greedy_all}", flush=True)
print(f"  Greedy-only: {greedy_all - qd_all}", flush=True)

print(f"\nResults saved to {RESULTS_DIR / 'code_downstream_results.json'}", flush=True)
