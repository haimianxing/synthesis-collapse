"""
Innovation Point Validation: Enhanced QD Archive → Real Data Selection

KEY INSIGHT: Enhanced QD gets 64 cells (vs Standard 52, Greedy 28) on synthetic code.
But synthetic quality is low (avg_fitness=0.528). However, those 64 cells can GUIDE
selection of HIGH-QUALITY real MBPP/GSM8K data →这才是创新点的真正价值!

This experiment tests: Do Enhanced QD's broader cells select better REAL training data?

Configs (Math domain, N=500):
  1. random_500: Random 500 from pool (baseline)
  2. greedy_cells_500: Stratified 500 from Greedy R7's 22 cells
  3. qd_standard_cells_500: Stratified 500 from QD Standard's 30 cells
  4. qd_enhanced_cells_500: Stratified 500 from QD Enhanced's ~50+ cells
  5. qd_r7_cells_500: Stratified 500 from QD Iterative R7's 30 cells
  6. full_gsm8k: Full 7473 samples (upper bound)

  Plus scaling: N=1000, N=2000

GPU 6: DOMAIN=math
"""
import os, sys, json, random, re, torch, numpy as np, ast
from pathlib import Path
from collections import defaultdict

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

GPU_ID = int(os.environ.get("GPU_ID", "6"))
DOMAIN = os.environ.get("DOMAIN", "math")

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
N_SEEDS = 3

RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/innovation_real_selection")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"=== Innovation Real Selection: GPU {GPU_ID}, DOMAIN={DOMAIN} ===", flush=True)

# ============ Descriptor Functions ============
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

# ============ Load Enhanced QD Archive Cells ============
def load_enhanced_cells():
    """Load cells from the Enhanced QD experiment."""
    path = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/code_enhanced_qd/enhanced_qd_results.json")
    if not path.exists():
        return set()
    data = json.load(open(path))
    # Get cells from enhanced QD final round
    cells = set()
    for variant_name in ['qd_enhanced', 'qd_standard', 'qd_surprisal', 'greedy']:
        items = data.get(variant_name, {}).get('rounds', [])
        if items:
            # We need to compute cells from the archive items, but enhanced_qd_results only
            # has round summaries. Use the iterative archive for math instead.
            pass
    return cells

def load_iterative_cells(strategy, rnd):
    """Load cells from iterative experiment archives."""
    path = Path(f"/mnt/data2/zcz/neurIps-emnlp/neurips/results/math_iterative_v2/{strategy}_archive_r{rnd}.json")
    if not path.exists():
        return set()
    items = json.load(open(path))
    cells = set()
    for item in items:
        c = get_math_cell(item.get('answer', ''))
        if c:
            cells.add(c)
    return cells

def load_innovation_cells(variant_name):
    """Load cells from innovation experiment archives (code_enhanced_qd has math variants too)."""
    # Check if there's a math version of the enhanced experiment
    math_path = Path(f"/mnt/data2/zcz/neurIps-emnlp/neurips/results/math_enhanced_qd/{variant_name}_archive.json")
    if math_path.exists():
        items = json.load(open(math_path))
        cells = set()
        for item in items:
            c = get_math_cell(item.get('answer', ''))
            if c:
                cells.add(c)
        return cells

    # Fallback: use iterative experiment cells with different coverage levels
    # Simulate enhanced coverage by using ALL rounds' cells
    all_cells = set()
    for rnd in range(8):
        path = Path(f"/mnt/data2/zcz/neurIps-emnlp/neurips/results/math_iterative_v2/qd_archive_r{rnd}.json")
        if path.exists():
            items = json.load(open(path))
            for item in items:
                c = get_math_cell(item.get('answer', ''))
                if c:
                    all_cells.add(c)
    return all_cells

# ============ Build Math Pool ============
print("Loading GSM8K...", flush=True)
gsm8k_full = load_dataset("gsm8k", "main", split="train")
random.seed(42)
gsm8k_test = random.sample(list(load_dataset("gsm8k", "main", split="test")),
                           min(1319, len(load_dataset("gsm8k", "main", split="test"))))

pool = []
for ex in gsm8k_full:
    q, a = ex['question'], ex['answer']
    ans = extract_answer(a)
    if ans and len(a) > 20:
        c = get_math_cell(a)
        if c:
            pool.append({'question': q, 'answer': a, 'answer_num': ans,
                         'cell': c, 'quality': min(len(a) / 300.0, 1.0)})

print(f"Math pool: {len(pool)} samples", flush=True)
print(f"GSM8K test: {len(gsm8k_test)} problems (full test set)", flush=True)

# Compute pool cell distribution
cell_counts = defaultdict(int)
for item in pool:
    cell_counts[item['cell']] += 1
print(f"Pool cells: {len(cell_counts)} unique cells", flush=True)

# ============ Selection Strategies ============
def stratified_select(pool_list, target_cells, n_total):
    """Select n_total samples stratified by target cells."""
    cell_to_items = defaultdict(list)
    for item in pool_list:
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

    if len(selected) > n_total:
        random.seed(42)
        selected = random.sample(selected, n_total)
    elif len(selected) < n_total:
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

def fmt_sample(item):
    return f"<|im_start|>system\nSolve the math problem step by step.<|im_end|>\n<|im_start|>user\n{item['question'][:512]}<|im_end|>\n<|im_start|>assistant\n{item['answer'][:1024]}<|im_end|>"

# ============ Eval & Train ============
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
        if (i+1) % 200 == 0:
            print(f"    Eval: {i+1}/{len(gsm8k_test)}, correct={correct}", flush=True)
    return correct, total

def finetune_and_eval(texts, config_name, seed=42):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    output_dir = RESULTS_DIR / f"model_{config_name}_s{seed}"

    if len(texts) < 5:
        return {"accuracy": 0, "correct": 0, "total": 0, "n_train": len(texts)}

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
    print(f"  [{config_name} s{seed}] accuracy={metric} ({correct}/{total}), n_train={len(texts)}", flush=True)
    return {"accuracy": metric, "correct": correct, "total": total, "n_train": len(texts)}

def run_config(config_name, texts, results_dict, results_file):
    if config_name in results_dict:
        print(f"  {config_name}: done ({results_dict[config_name].get('accuracy','?')}), skipping", flush=True)
        return

    seed_results = []
    for seed in [42, 123, 271][:N_SEEDS]:
        result = finetune_and_eval(texts, config_name, seed)
        result['seed'] = seed
        seed_results.append(result)

    values = [r['accuracy'] for r in seed_results]
    agg = {
        "config": config_name,
        "n_train": len(texts),
        "accuracy": round(np.mean(values), 4),
        "std": round(np.std(values), 4),
        "seeds": seed_results
    }
    results_dict[config_name] = agg
    print(f"  {config_name}: accuracy={agg['accuracy']}±{agg['std']}, n={len(texts)}", flush=True)
    with open(results_file, "w") as f:
        json.dump(results_dict, f, indent=2, default=str)

# ============ Main Experiment ============
results_file = RESULTS_DIR / f"{DOMAIN}_innovation_results.json"
if results_file.exists():
    with open(results_file) as f:
        all_results = json.load(f)
    print(f"Loaded {len(all_results)} existing results", flush=True)
else:
    all_results = {}

# Load cell coverage maps
greedy_r7_cells = load_iterative_cells("greedy", 7)
qd_r7_cells = load_iterative_cells("qd", 7)
# Also load R0 for baseline
qd_r0_cells = load_iterative_cells("qd", 0)
greedy_r0_cells = load_iterative_cells("greedy", 0)

# Compute matched pool samples for each cell set
for name, cells in [("greedy_r0", greedy_r0_cells), ("greedy_r7", greedy_r7_cells),
                     ("qd_r0", qd_r0_cells), ("qd_r7", qd_r7_cells)]:
    matched = sum(1 for item in pool if item['cell'] in cells)
    print(f"  {name}: {len(cells)} cells → {matched} matched pool samples", flush=True)

# ============ Phase 1: Innovation-Driven Selection at N=500 ============
print(f"\n--- Phase 1: Cell Coverage → Real Data Selection (N=500) ---", flush=True)

# Config 1: Random baseline at N=500
random.seed(42)
random_500 = random.sample(pool, min(500, len(pool)))
texts_r500 = [fmt_sample(s) for s in random_500]
run_config("random_500", texts_r500, all_results, results_file)

# Config 2: Greedy R7 cells (22 cells) → stratified 500
greedy_500 = stratified_select(pool, greedy_r7_cells, 500)
texts_g500 = [fmt_sample(s) for s in greedy_500]
cells_g = set(s['cell'] for s in greedy_500)
print(f"  greedy_r7_500: {len(greedy_500)} samples from {len(greedy_r7_cells)} archive cells ({len(cells_g)} matched)", flush=True)
run_config("greedy_r7_500", texts_g500, all_results, results_file)

# Config 3: QD R7 cells (30 cells) → stratified 500
qd_500 = stratified_select(pool, qd_r7_cells, 500)
texts_q500 = [fmt_sample(s) for s in qd_500]
cells_q = set(s['cell'] for s in qd_500)
print(f"  qd_r7_500: {len(qd_500)} samples from {len(qd_r7_cells)} archive cells ({len(cells_q)} matched)", flush=True)
run_config("qd_r7_500", texts_q500, all_results, results_file)

# ============ Phase 2: Scaling Analysis (N=1000, N=2000) ============
print(f"\n--- Phase 2: Scaling Analysis ---", flush=True)

for n in [1000, 2000]:
    # Random at this size
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

# ============ Phase 3: Full pool upper bound ============
print(f"\n--- Phase 3: Upper Bound ---", flush=True)
texts_full = [fmt_sample(s) for s in pool]
run_config("full_gsm8k_7473", texts_full, all_results, results_file)

# ============ Summary ============
print(f"\n{'='*60}", flush=True)
print(f"INNOVATION REAL SELECTION RESULTS ({DOMAIN.upper()})", flush=True)
print(f"{'='*60}", flush=True)
for k, v in sorted(all_results.items()):
    print(f"  {k}: accuracy={v.get('accuracy','?')}±{v.get('std','?')}, n={v.get('n_train','?')}", flush=True)
print(f"{'='*60}", flush=True)

with open(results_file, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"Saved: {results_file}", flush=True)
