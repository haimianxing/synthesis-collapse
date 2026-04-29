"""
Per-Round Collapse Evidence — Top-K Per Cell Selection

Shows that greedy's frozen coverage → degraded training data selection,
while QD's growing coverage → improved training data.

Key design:
  - Code: Use ALL matched pool samples (49-384 samples per round, fast training)
  - Math: Use top-10 per cell (100-300 samples, avoids 2000+ sample training)
  - 3 seeds, 3 rounds (R0/R3/R7), both strategies

GPU allocation:
  GPU 1: Code per-round collapse evidence
  GPU 3: Math per-round collapse evidence
"""
import os, sys, json, random, re, torch, numpy as np, ast
from pathlib import Path
from collections import defaultdict

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

GPU_ID = int(os.environ.get("GPU_ID", "1"))
DOMAIN = os.environ.get("DOMAIN", "code")
N_SEEDS = 3
MATH_TOP_K = 10  # Top-10 per cell for math (avoids 2000+ training)

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

RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/collapse_evidence")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"=== Collapse Evidence: GPU {GPU_ID}, DOMAIN={DOMAIN} ===", flush=True)

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

print(f"{DOMAIN} pool: {len(pool)} samples", flush=True)

# ============ Selection: ALL matched for code, top-K for math ============
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

def select_training_data(pool, target_cells, domain):
    """Code: ALL matched. Math: top-K per cell."""
    if domain == "code":
        return [item for item in pool if item['cell'] in target_cells]
    else:
        # Top-K per cell
        cell_to_items = defaultdict(list)
        for item in pool:
            if item['cell'] in target_cells:
                cell_to_items[item['cell']].append(item)

        selected = []
        for cell, items in cell_to_items.items():
            items_sorted = sorted(items, key=lambda x: x['quality'], reverse=True)
            selected.extend(items_sorted[:MATH_TOP_K])
        return selected

# ============ Training ============
def finetune_and_eval(texts, config_name, seed=42):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    output_dir = RESULTS_DIR / f"model_{DOMAIN}_{config_name}_s{seed}"

    if len(texts) < 5:
        return {"pass_at_1": 0, "accuracy": 0, "correct": 0, "total": 0, "n_train": len(texts)}

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
        print(f"  {config_name}: done, skipping", flush=True)
        return

    seed_results = []
    for seed in [42, 123, 271][:N_SEEDS]:
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

# ============ Main ============
results_file = RESULTS_DIR / f"{DOMAIN}_collapse_evidence.json"
if results_file.exists():
    with open(results_file) as f:
        all_results = json.load(f)
else:
    all_results = {}

# Per-round evidence
ROUNDS = [0, 3, 7]
for strategy in ["greedy", "qd"]:
    print(f"\n--- {strategy.upper()} per-round ---", flush=True)
    for rnd in ROUNDS:
        cells = load_archive_cells(strategy, rnd)
        if not cells:
            continue

        selected = select_training_data(pool, cells, DOMAIN)
        if len(selected) < 5:
            print(f"  {strategy} R{rnd}: only {len(selected)} matched, skip", flush=True)
            continue

        texts = [fmt_sample(s) for s in selected]
        config_name = f"{strategy}_r{rnd}_{len(selected)}"

        # Report coverage growth
        print(f"  {strategy} R{rnd}: {len(cells)} cells → {len(selected)} training samples", flush=True)
        run_config(config_name, texts, all_results, results_file)

# ============ Summary ============
print(f"\n{'='*60}", flush=True)
print(f"COLLAPSE EVIDENCE ({DOMAIN.upper()})", flush=True)
print(f"{'='*60}", flush=True)
metric_name = "pass_at_1" if DOMAIN == "code" else "accuracy"
for k, v in sorted(all_results.items()):
    print(f"  {k}: {metric_name}={v.get(metric_name,'?')}±{v.get('std','?')}, n={v.get('n_train','?')}", flush=True)
print(f"{'='*60}", flush=True)

with open(results_file, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"Saved: {results_file}", flush=True)
