"""
Math Domain Downstream Fine-tuning (Addresses SAC C3: downstream only Dialogue)
Pool: GSM8K training (7,473 problems)
Selection: QD-500, Greedy-500, Random-500 from pool
Fine-tune: Qwen2.5-1.5B + LoRA (r=16, alpha=32, 3 epochs)
Evaluate: GSM8K test (1,319 problems), accuracy
Multi-seed: 8 seeds for statistical significance
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys, json, random, re, torch, numpy as np, time
from pathlib import Path
from collections import Counter, defaultdict
from datasets import load_dataset

# ============ Config ============
MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-1___5B-Instruct"
K = 500           # training subset size (bigger than Dialogue's 57)
N_TEST = 1319     # full GSM8K test
SEEDS = [42, 123, 159, 271, 314, 456, 789, 2024]
GRID_RES = 10
DIM = 3
GPU_ID = int(os.environ.get("GPU_ID", "1"))
# Use CUDA_VISIBLE_DEVICES for proper GPU isolation
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
DEVICE = "cuda:0"  # Will be the only visible GPU

RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/math_downstream")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"=== Math Downstream Fine-tuning (GPU {GPU_ID}) ===", flush=True)
print(f"Config: K={K}, N_TEST={N_TEST}, {len(SEEDS)} seeds", flush=True)

# ============ Load GSM8K ============
print("Loading GSM8K...", flush=True)
train_ds = load_dataset("gsm8k", "main", split="train")
test_ds = load_dataset("gsm8k", "main", split="test")
print(f"Train: {len(train_ds)}, Test: {len(test_ds)}", flush=True)

def extract_answer(text):
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '')
    return None

def compute_descriptors(problem, solution):
    steps = solution.count('<<') + 1
    sol_len = len(solution)
    is_multi_step = 1 if steps >= 3 else 0
    difficulty = min(sol_len / 500.0, 1.0)
    return {
        'difficulty': difficulty,
        'num_steps': min(steps / 10.0, 1.0),
        'is_multi_step': is_multi_step
    }

def get_cell(desc):
    d = desc['difficulty']
    s = desc['num_steps']
    m = desc['is_multi_step']
    return (int(d * GRID_RES), int(s * GRID_RES), int(m * GRID_RES))

# Build pool
print(f"Computing descriptors for {len(train_ds)} training examples...", flush=True)
pool = []
for ex in train_ds:
    q = ex['question']
    a = ex['answer']
    ans = extract_answer(a)
    if ans is None:
        continue
    desc = compute_descriptors(q, a)
    quality = min(len(a) / 300.0, 1.0)
    pool.append({
        'question': q, 'answer': a, 'answer_num': ans,
        'descriptors': desc, 'quality': quality
    })
print(f"Pool: {len(pool)} examples", flush=True)

# ============ Selection Strategies ============
def select_qd(pool_items, k, seed=42):
    """QD-Synth: grid-based selection with per-cell elitism"""
    rng = random.Random(seed)
    grid = {}
    for item in pool_items:
        cell = get_cell(item['descriptors'])
        if cell not in grid or item['quality'] > grid[cell]['quality']:
            grid[cell] = item
    cells_sorted = sorted(grid.values(), key=lambda x: x['quality'], reverse=True)
    return cells_sorted[:k]

def select_greedy(pool_items, k):
    """Greedy: top-k by quality"""
    return sorted(pool_items, key=lambda x: x['quality'], reverse=True)[:k]

def select_random(pool_items, k, seed=42):
    """Random: uniform sampling"""
    rng = random.Random(seed)
    return rng.sample(pool_items, k)

# ============ Fine-tuning ============
def finetune(train_samples, config_name, seed):
    """Fine-tune Qwen2.5-1.5B with LoRA on math problems"""
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

    # Format: instruction → solution with reasoning
    def format_math(sample):
        return f"<|im_start|>system\nSolve the math problem step by step.<|im_end|>\n<|im_start|>user\n{sample['question']}<|im_end|>\n<|im_start|>assistant\n{sample['answer'][:768]}<|im_end|>"

    train_texts = [format_math(s) for s in train_samples]
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

# ============ Evaluation ============
def evaluate(model_path, test_data, seed):
    """Evaluate fine-tuned model on GSM8K test"""
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
    total = 0
    per_difficulty = defaultdict(lambda: {"correct": 0, "total": 0})

    for i, ex in enumerate(test_data):
        test_q = ex['question']
        test_ans = extract_answer(ex['answer'])

        # Compute difficulty bucket for per-difficulty analysis
        desc = compute_descriptors(test_q, ex['answer'])
        diff_bucket = "easy" if desc['difficulty'] < 0.33 else ("medium" if desc['difficulty'] < 0.66 else "hard")

        messages = [
            {"role": "system", "content": "Solve the math problem step by step."},
            {"role": "user", "content": test_q}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=256, temperature=0.0,
                do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        numbers = re.findall(r'-?[\d,]+\.?\d*', response.replace(',', ''))
        pred = numbers[-1] if numbers else None

        is_correct = False
        if pred and test_ans:
            try:
                if abs(float(pred) - float(test_ans)) < 0.01:
                    is_correct = True
                    correct += 1
            except:
                pass
        total += 1
        per_difficulty[diff_bucket]["total"] += 1
        if is_correct:
            per_difficulty[diff_bucket]["correct"] += 1

        if (i + 1) % 100 == 0:
            print(f"    [{model_path.parent.name}] {i+1}/{total}, acc={correct/total:.4f}", flush=True)

    acc = correct / total if total > 0 else 0
    diff_acc = {k: v["correct"]/max(v["total"],1) for k, v in per_difficulty.items()}

    del model, base_model
    torch.cuda.empty_cache()

    return {
        "accuracy": round(acc, 4),
        "correct": correct,
        "total": total,
        "per_difficulty": diff_acc,
        "per_difficulty_counts": {k: v for k, v in per_difficulty.items()}
    }

# ============ Main Loop ============
def run_seed(seed):
    """Run one seed: select → train → evaluate for all 3 methods"""
    results = {}
    methods = {
        "qd_500": select_qd(pool, K, seed),
        "greedy_500": select_greedy(pool, K),
        "random_500": select_random(pool, K, seed),
    }

    # Selection stats
    for name, selected in methods.items():
        cells = set(get_cell(s['descriptors']) for s in selected)
        qualities = [s['quality'] for s in selected]
        difficulties = [s['descriptors']['difficulty'] for s in selected]
        print(f"  [{name}] n={len(selected)}, cells={len(cells)}, "
              f"q={np.mean(qualities):.3f}, diff={np.mean(difficulties):.3f}", flush=True)

    for name, selected in methods.items():
        config_name = f"{name}_s{seed}"
        print(f"\n--- Training {config_name} ---", flush=True)
        t0 = time.time()
        model_path = finetune(selected, config_name, seed)
        print(f"  Training took {time.time()-t0:.0f}s", flush=True)

        print(f"--- Evaluating {config_name} ---", flush=True)
        t0 = time.time()
        eval_result = evaluate(model_path, test_ds, seed)
        eval_result["train_time_s"] = round(time.time()-t0)
        eval_result["seed"] = seed
        eval_result["n_train"] = len(selected)
        results[name] = eval_result
        print(f"  Accuracy: {eval_result['accuracy']:.4f} ({eval_result['correct']}/{eval_result['total']})", flush=True)
        print(f"  Per-diff: {eval_result['per_difficulty']}", flush=True)

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

    # Save after each seed (crash-safe)
    with open(RESULTS_DIR / "math_downstream_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

# ============ Aggregate Statistics ============
print(f"\n{'='*60}", flush=True)
print("AGGREGATE RESULTS", flush=True)
print(f"{'='*60}", flush=True)

from scipy.stats import wilcoxon

for method in ["qd_500", "greedy_500", "random_500"]:
    accs = [all_results[s][method]["accuracy"] for s in SEEDS]
    print(f"{method}: {np.mean(accs):.4f} ± {np.std(accs):.4f} "
          f"(seeds: {[round(a,4) for a in accs]})", flush=True)

# Wilcoxon tests
qd_accs = [all_results[s]["qd_500"]["accuracy"] for s in SEEDS]
greedy_accs = [all_results[s]["greedy_500"]["accuracy"] for s in SEEDS]
random_accs = [all_results[s]["random_500"]["accuracy"] for s in SEEDS]

try:
    stat1, p1 = wilcoxon(qd_accs, greedy_accs)
    d1 = (np.mean(qd_accs) - np.mean(greedy_accs)) / np.std([a-b for a,b in zip(qd_accs, greedy_accs)])
    print(f"\nQD vs Greedy: Wilcoxon p={p1:.4f}, Cohen's d={d1:.2f}", flush=True)
except:
    print("\nQD vs Greedy: Wilcoxon test failed (all equal?)", flush=True)

try:
    stat2, p2 = wilcoxon(qd_accs, random_accs)
    d2 = (np.mean(qd_accs) - np.mean(random_accs)) / np.std([a-b for a,b in zip(qd_accs, random_accs)])
    print(f"QD vs Random: Wilcoxon p={p2:.4f}, Cohen's d={d2:.2f}", flush=True)
except:
    print("QD vs Random: Wilcoxon test failed", flush=True)

# Per-difficulty analysis
print(f"\n=== Per-Difficulty Accuracy ===", flush=True)
for diff in ["easy", "medium", "hard"]:
    for method in ["qd_500", "greedy_500", "random_500"]:
        accs = [all_results[s][method]["per_difficulty"].get(diff, 0) for s in SEEDS]
        print(f"  {method} {diff}: {np.mean(accs):.4f} ± {np.std(accs):.4f}", flush=True)

print(f"\nResults saved to {RESULTS_DIR / 'math_downstream_results.json'}", flush=True)
