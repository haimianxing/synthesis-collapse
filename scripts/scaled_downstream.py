"""
Scaled Downstream Evaluation — Fixed-Size Fair Comparison

Key insight: Previous math comparison was UNFAIR (QD-75 vs Greedy-500).
This script does FAIR comparison at equal sample sizes (N=500 for math, N=200 for code).

Also runs per-round analysis with ALL matched pool samples (not just best-1 per cell).

GPU allocation:
  GPU 2: Math scaled (QD-500 vs Greedy-500 vs Random-500, per-round ALL matched)
  GPU 4: Code scaled (QD-200 vs Greedy-200 vs Random-200, per-round ALL matched)
"""
import os, sys, json, random, re, torch, numpy as np, ast, time
from pathlib import Path
from collections import defaultdict

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

GPU_ID = int(os.environ.get("GPU_ID", "2"))
DOMAIN = os.environ.get("DOMAIN", "math")  # "code" or "math"
N_SEEDS = 5

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-1___5B-Instruct"
DEVICE = "cuda:0"
GRID_RES = 10

RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/scaled_downstream")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"=== Scaled Downstream: GPU {GPU_ID}, DOMAIN={DOMAIN} ===", flush=True)

# ============ Descriptor Functions ============
def get_code_cell(code, prompt=''):
    if not code or len(code) < 20:
        return None
    code_len = len(code)
    difficulty = min(code_len / 1000.0, 1.0)
    api_count = 0
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call): api_count += 1
            elif isinstance(node, ast.Import): api_count += len(node.names)
            elif isinstance(node, ast.ImportFrom): api_count += len(node.names)
    except:
        api_count = len(re.findall(r'\b\w+\.\w+\(', code))
    has_debug = 1 if ('try:' in code or 'except' in code or 'assert' in code) else 0
    return (int(difficulty * GRID_RES), int(min(api_count / 10.0, 1.0) * GRID_RES), int(has_debug * GRID_RES))

def get_math_cell(answer):
    if not answer or len(answer) < 20:
        return None
    steps = answer.count('<<') + 1
    difficulty = min(len(answer) / 500.0, 1.0)
    is_multi = 1 if steps >= 3 else 0
    s = min(steps / 10.0, 1.0)
    return (int(difficulty * GRID_RES), int(s * GRID_RES), int(is_multi * GRID_RES))

def extract_answer(text):
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    return match.group(1).replace(',', '') if match else None

# ============ Build Pool & Load Archives ============
if DOMAIN == "code":
    ITER_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/code_iterative_v2")
    FIXED_N = 200  # Fair comparison at 200 samples

    print("Loading MBPP pool...", flush=True)
    try: mbpp_train = load_dataset("mbpp", "sanitized", split="train")
    except: mbpp_train = load_dataset("mbpp", split="train")
    try: mbpp_test = load_dataset("mbpp", "sanitized", split="test")
    except: mbpp_test = load_dataset("mbpp", split="test")
    try: mbpp_val = load_dataset("mbpp", "sanitized", split="validation")
    except: mbpp_val = []

    pool = []
    for ex in list(mbpp_train) + list(mbpp_test) + list(mbpp_val):
        code = ex.get('code', '')
        prompt = ex.get('prompt', ex.get('text', ''))
        if code and len(code) > 20:
            c = get_code_cell(code, prompt)
            if c:
                pool.append({'prompt': prompt, 'code': code, 'cell': c,
                             'quality': min(len(code) / 500.0, 1.0)})
    print(f"Code pool: {len(pool)} MBPP samples", flush=True)

    print("Loading HumanEval...", flush=True)
    humaneval = load_dataset("openai_humaneval", split="test")

    def eval_fn(model, tokenizer):
        correct = total = 0
        for i, ex in enumerate(humaneval):
            msgs = [{"role":"system","content":"Complete the Python function."},
                    {"role":"user","content":ex['prompt']}]
            txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = tokenizer(txt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=256, temperature=0.0,
                                   do_sample=False, pad_token_id=tokenizer.eos_token_id)
            resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
            try:
                exec_globals = {}; exec(ex['prompt'] + resp, exec_globals); exec(ex['test'], exec_globals)
                correct += 1
            except: pass
            total += 1
            if (i+1) % 40 == 0:
                print(f"    Eval: {i+1}/164, correct={correct}", flush=True)
        return correct, total

    def fmt_sample(item):
        return f"<|im_start|>system\nComplete the Python function.<|im_end|>\n<|im_start|>user\n{item['prompt'][:512]}<|im_end|>\n<|im_start|>assistant\n{item['code'][:1024]}<|im_end|>"

elif DOMAIN == "math":
    ITER_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/math_iterative_v2")
    FIXED_N = 500  # Fair comparison at 500 samples

    print("Loading GSM8K pool...", flush=True)
    gsm8k_full = load_dataset("gsm8k", "main", split="train")
    random.seed(42)
    gsm8k_test = random.sample(list(load_dataset("gsm8k", "main", split="test")),
                               min(500, len(load_dataset("gsm8k", "main", split="test"))))

    pool = []
    for ex in gsm8k_full:
        q, a = ex['question'], ex['answer']
        ans = extract_answer(a)
        if ans and len(a) > 20:
            c = get_math_cell(a)
            if c:
                pool.append({'question': q, 'answer': a, 'answer_num': ans,
                             'cell': c, 'quality': min(len(a) / 300.0, 1.0)})
    print(f"Math pool: {len(pool)} GSM8K samples", flush=True)
    print(f"Math test: {len(gsm8k_test)} samples", flush=True)

    def eval_fn(model, tokenizer):
        correct = total = 0
        for i, ex in enumerate(gsm8k_test):
            msgs = [{"role":"system","content":"Solve the math problem step by step."},
                    {"role":"user","content":ex['question']}]
            txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = tokenizer(txt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=256, temperature=0.0, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
            resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
            pred_match = re.search(r'####\s*(-?[\d,]+\.?\d*)', resp)
            if not pred_match:
                pred_match = re.search(r'(\d+\.?\d*)\s*$', resp.strip())
            pred = pred_match.group(1).replace(',', '') if pred_match else None
            gold = extract_answer(ex['answer'])
            if pred and gold and pred.strip() == gold.strip():
                correct += 1
            total += 1
            if (i+1) % 100 == 0:
                print(f"    Eval: {i+1}/{len(gsm8k_test)}, correct={correct}", flush=True)
        return correct, total

    def fmt_sample(item):
        return f"<|im_start|>system\nSolve the math problem step by step.<|im_end|>\n<|im_start|>user\n{item['question'][:512]}<|im_end|>\n<|im_start|>assistant\n{item['answer'][:1024]}<|im_end|>"

# ============ Selection Strategies ============
def load_archive_cells(strategy, rnd):
    path = ITER_DIR / f"{strategy}_archive_r{rnd}.json"
    if not path.exists():
        return set()
    items = json.load(open(path))
    cells = set()
    for item in items:
        if DOMAIN == "code":
            c = get_code_cell(item.get('code', ''), item.get('prompt', ''))
        else:
            c = get_math_cell(item.get('answer', ''))
        if c:
            cells.add(c)
    return cells

def stratified_select(pool, target_cells, n_total):
    """Select n_total samples from pool, stratified by target_cells coverage."""
    # Group pool items by cell
    cell_to_items = defaultdict(list)
    for item in pool:
        if item['cell'] in target_cells:
            cell_to_items[item['cell']].append(item)

    if not cell_to_items:
        return []

    n_cells = len(cell_to_items)
    per_cell = max(1, n_total // n_cells)

    selected = []
    for cell in sorted(cell_to_items.keys()):
        items = sorted(cell_to_items[cell], key=lambda x: x['quality'], reverse=True)
        selected.extend(items[:per_cell])

    # Trim or pad to exact n_total
    if len(selected) > n_total:
        random.seed(42)
        selected = random.sample(selected, n_total)
    elif len(selected) < n_total:
        # Pad with remaining high-quality items from matched cells
        remaining = []
        for cell in sorted(cell_to_items.keys()):
            items = sorted(cell_to_items[cell], key=lambda x: x['quality'], reverse=True)
            remaining.extend(items[per_cell:])
        remaining.sort(key=lambda x: x['quality'], reverse=True)
        for item in remaining:
            if len(selected) >= n_total:
                break
            if item not in selected:
                selected.append(item)

    return selected[:n_total]

def select_all_matched(pool, target_cells):
    """Select ALL pool items matching any target cell."""
    return [item for item in pool if item['cell'] in target_cells]

# ============ Training ============
def finetune_and_eval(texts, config_name, seed=42):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    output_dir = RESULTS_DIR / f"model_{DOMAIN}_{config_name}_s{seed}"

    if len(texts) < 5:
        metric_name = "pass_at_1" if DOMAIN == "code" else "accuracy"
        return {metric_name: 0, "correct": 0, "total": 0, "n_train": len(texts)}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
                                                  device_map=DEVICE, trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj"], lora_dropout=0.05, task_type="CAUSAL_LM"))

    ds = Dataset.from_dict({"text": texts})
    trainer = SFTTrainer(model=model, args=SFTConfig(
        output_dir=str(output_dir), num_train_epochs=3,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=2e-4, logging_steps=50, save_strategy="no",
        bf16=True, report_to="none", max_length=768,
        dataset_text_field="text", packing=False),
        train_dataset=ds, processing_class=tokenizer)
    trainer.train()

    model.eval()
    correct, total = eval_fn(model, tokenizer)

    del model; torch.cuda.empty_cache()
    metric = round(correct/total, 4) if total > 0 else 0
    metric_name = "pass_at_1" if DOMAIN == "code" else "accuracy"
    print(f"  [{config_name} s{seed}] {metric_name}={metric} ({correct}/{total}), n_train={len(texts)}", flush=True)
    return {metric_name: metric, "correct": correct, "total": total, "n_train": len(texts)}

def run_config(config_name, texts, results_dict, results_file):
    if config_name in results_dict:
        print(f"  {config_name}: done ({results_dict[config_name].get('pass_at_1', results_dict[config_name].get('accuracy', '?'))}), skipping", flush=True)
        return

    seed_results = []
    for seed in [42, 123, 271, 456, 2024][:N_SEEDS]:
        result = finetune_and_eval(texts, config_name, seed)
        result['seed'] = seed
        seed_results.append(result)

    metric_name = "pass_at_1" if DOMAIN == "code" else "accuracy"
    values = [r[metric_name] for r in seed_results]
    agg = {
        "config": config_name,
        "n_train": len(texts),
        metric_name: round(np.mean(values), 4),
        "std": round(np.std(values), 4),
        "seeds": seed_results
    }
    results_dict[config_name] = agg
    print(f"  {config_name}: {metric_name}={agg[metric_name]}±{agg['std']}, n={len(texts)}", flush=True)
    with open(results_file, "w") as f:
        json.dump(results_dict, f, indent=2, default=str)

# ============ Main Experiment ============
results_file = RESULTS_DIR / f"{DOMAIN}_scaled_results.json"
if results_file.exists():
    with open(results_file) as f:
        all_results = json.load(f)
    print(f"Loaded {len(all_results)} existing results", flush=True)
else:
    all_results = {}

# --- Phase 1: Fixed-Size Fair Comparison ---
print(f"\n--- Phase 1: Fixed-Size Comparison (N={FIXED_N}) ---", flush=True)

# Random baseline
random.seed(42)
random_subset = random.sample(pool, min(FIXED_N, len(pool)))
texts_random = [fmt_sample(s) for s in random_subset]
run_config(f"random_{FIXED_N}", texts_random, all_results, results_file)

# Greedy-cell stratified selection (R7)
greedy_r7_cells = load_archive_cells("greedy", 7)
print(f"  Greedy R7: {len(greedy_r7_cells)} cells", flush=True)
greedy_selected = stratified_select(pool, greedy_r7_cells, FIXED_N)
texts_greedy = [fmt_sample(s) for s in greedy_selected]
print(f"  Greedy stratified: {len(greedy_selected)} samples from {len(greedy_r7_cells)} cells", flush=True)
run_config(f"greedy_r7_{FIXED_N}", texts_greedy, all_results, results_file)

# QD-cell stratified selection (R7)
qd_r7_cells = load_archive_cells("qd", 7)
print(f"  QD R7: {len(qd_r7_cells)} cells", flush=True)
qd_selected = stratified_select(pool, qd_r7_cells, FIXED_N)
texts_qd = [fmt_sample(s) for s in qd_selected]
print(f"  QD stratified: {len(qd_selected)} samples from {len(qd_r7_cells)} cells", flush=True)
run_config(f"qd_r7_{FIXED_N}", texts_qd, all_results, results_file)

# --- Phase 2: Per-Round ALL Matched (Collapse Evidence) ---
print(f"\n--- Phase 2: Per-Round ALL Matched Samples ---", flush=True)

for strategy in ["greedy", "qd"]:
    for rnd in [0, 3, 7]:
        cells = load_archive_cells(strategy, rnd)
        if not cells:
            continue
        matched = select_all_matched(pool, cells)
        if len(matched) < 5:
            print(f"  {strategy} R{rnd}: only {len(matched)} matched, skipping", flush=True)
            continue

        texts = [fmt_sample(s) for s in matched]
        config_name = f"{strategy}_r{rnd}_all_{len(matched)}"
        print(f"  {strategy} R{rnd}: {len(cells)} cells → {len(matched)} all matched samples", flush=True)
        run_config(config_name, texts, all_results, results_file)

# --- Phase 3: Additional Fixed Sizes for Scaling Analysis ---
print(f"\n--- Phase 3: Scaling Analysis ---", flush=True)

for n in [100, 300] if DOMAIN == "math" else [50, 100]:
    # Random baseline at this size
    random.seed(42)
    random_n = random.sample(pool, min(n, len(pool)))
    texts_rn = [fmt_sample(s) for s in random_n]
    run_config(f"random_{n}", texts_rn, all_results, results_file)

    # Greedy stratified
    greedy_n = stratified_select(pool, greedy_r7_cells, n)
    if len(greedy_n) >= 5:
        texts_gn = [fmt_sample(s) for s in greedy_n]
        run_config(f"greedy_r7_{n}", texts_gn, all_results, results_file)

    # QD stratified
    qd_n = stratified_select(pool, qd_r7_cells, n)
    if len(qd_n) >= 5:
        texts_qn = [fmt_sample(s) for s in qd_n]
        run_config(f"qd_r7_{n}", texts_qn, all_results, results_file)

# ============ Summary ============
print(f"\n{'='*60}", flush=True)
print(f"SCALED DOWNSTREAM RESULTS ({DOMAIN.upper()})", flush=True)
print(f"{'='*60}", flush=True)
metric_name = "pass_at_1" if DOMAIN == "code" else "accuracy"
for k, v in sorted(all_results.items()):
    print(f"  {k}: {metric_name}={v.get(metric_name,'?')}±{v.get('std','?')}, n={v.get('n_train','?')}", flush=True)
print(f"{'='*60}", flush=True)

with open(results_file, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"Saved: {results_file}", flush=True)
