"""
Per-Round Downstream with REAL Data Selection v3

KEY FIX: Instead of training on API-generated code (low quality),
use each round's archive CELLS as selection criteria for REAL MBPP/GSM8K data.

Design:
  For each round R0-R7:
    1. Load greedy/QD archive → extract cells
    2. For each cell, find the BEST matching REAL sample from the pool
    3. Train Qwen2.5-1.5B on selected real samples
    4. Evaluate on HumanEval / GSM8K test

This tests: "Does QD's coverage advantage translate to better REAL data selection?"

GPU allocation:
  GPU_ID=1 DOMAIN=code   → Code per-round (greedy + QD, R0-R7, 5 seeds)
  GPU_ID=3 DOMAIN=math   → Math per-round (greedy + QD, R0-R7, 5 seeds)
  GPU_ID=5 CONFIG=code_ref → Code reference baselines
  GPU_ID=7 CONFIG=math_ref → Math reference baselines
"""
import os, sys, json, random, re, torch, numpy as np, ast, time
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

GPU_ID = int(os.environ.get("GPU_ID", "1"))
DOMAIN = os.environ.get("DOMAIN", "code")   # "code" or "math"
CONFIG = os.environ.get("CONFIG", "")        # "code_ref" or "math_ref" for reference baselines
N_SEEDS = 3

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

RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/per_round_v3")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"=== Per-Round v3 (Real Data Selection): GPU {GPU_ID}, DOMAIN={DOMAIN}, CONFIG={CONFIG} ===", flush=True)

# ============ Code Setup ============
if DOMAIN == "code" or CONFIG == "code_ref":
    def compute_descriptors(prompt, code, **kwargs):
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
        return {'difficulty': difficulty, 'num_APIs': min(api_count / 10.0, 1.0), 'needs_debugging': has_debug}

    def get_cell(desc):
        return (int(desc['difficulty'] * GRID_RES), int(desc['num_APIs'] * GRID_RES), int(desc['needs_debugging'] * GRID_RES))

    def compute_quality(sample):
        code = sample.get('code', '')
        return min(len(code) / 500.0, 1.0) if code else 0.1

    def fmt_code(prompt, code):
        return f"<|im_start|>system\nComplete the Python function.<|im_end|>\n<|im_start|>user\n{prompt[:512]}<|im_end|>\n<|im_start|>assistant\n{code[:1024]}<|im_end|>"

    ITER_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/code_iterative_v2")

    print("Loading MBPP (train+test as pool)...", flush=True)
    try:
        mbpp_train = load_dataset("mbpp", "sanitized", split="train")
    except:
        mbpp_train = load_dataset("mbpp", split="train")
    try:
        mbpp_test = load_dataset("mbpp", "sanitized", split="test")
    except:
        mbpp_test = load_dataset("mbpp", split="test")

    # Build pool: train + test + validation (no leakage with HumanEval)
    try:
        mbpp_val = load_dataset("mbpp", "sanitized", split="validation")
    except:
        mbpp_val = []
    pool = []
    for ex in list(mbpp_train) + list(mbpp_test) + list(mbpp_val):
        code = ex.get('code', '')
        prompt = ex.get('prompt', ex.get('text', ''))
        if code and len(code) > 20:
            desc = compute_descriptors(prompt, code)
            pool.append({'prompt': prompt, 'code': code, 'descriptors': desc,
                         'quality': compute_quality({'code': code}), 'cell': get_cell(desc)})
    print(f"Pool: {len(pool)} valid MBPP samples (train+test+val)", flush=True)

    print("Loading HumanEval...", flush=True)
    humaneval = load_dataset("openai_humaneval", split="test")
    print(f"HumanEval: {len(humaneval)}", flush=True)

    def eval_code(model, tokenizer):
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

# ============ Math Setup ============
if DOMAIN == "math" or CONFIG == "math_ref":
    def compute_math_descriptors(question, answer, **kwargs):
        steps = answer.count('<<') + 1
        difficulty = min(len(answer) / 500.0, 1.0)
        is_multi = 1 if steps >= 3 else 0
        return {'difficulty': difficulty, 'num_steps': min(steps / 10.0, 1.0), 'is_multi_step': is_multi}

    def get_math_cell(desc):
        return (int(desc['difficulty'] * GRID_RES), int(desc['num_steps'] * GRID_RES), int(desc['is_multi_step'] * GRID_RES))

    def extract_answer(text):
        match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
        return match.group(1).replace(',', '') if match else None

    def fmt_math(question, answer):
        return f"<|im_start|>system\nSolve the math problem step by step.<|im_end|>\n<|im_start|>user\n{question[:512]}<|im_end|>\n<|im_start|>assistant\n{answer[:1024]}<|im_end|>"

    ITER_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/math_iterative_v2")

    print("Loading GSM8K...", flush=True)
    gsm8k_full = load_dataset("gsm8k", "main", split="train")
    random.seed(42)
    gsm8k_test = random.sample(list(load_dataset("gsm8k", "main", split="test")), min(200, len(load_dataset("gsm8k", "main", split="test"))))

    math_pool = []
    for ex in gsm8k_full:
        q, a = ex['question'], ex['answer']
        ans = extract_answer(a)
        if ans and len(a) > 20:
            desc = compute_math_descriptors(q, a)
            math_pool.append({'question': q, 'answer': a, 'answer_num': ans,
                              'descriptors': desc, 'cell': get_math_cell(desc),
                              'quality': min(len(a) / 300.0, 1.0)})
    print(f"Math pool: {len(math_pool)} valid GSM8K train samples", flush=True)
    print(f"GSM8K test: {len(gsm8k_test)}", flush=True)

    def eval_math(model, tokenizer):
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
            if (i+1) % 50 == 0:
                print(f"    Eval: {i+1}/{len(gsm8k_test)}, correct={correct}", flush=True)
        return correct, total

# ============ Training Function ============
def finetune_and_eval(texts, config_name, eval_fn, seed=42):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    output_dir = RESULTS_DIR / f"model_{config_name}_s{seed}"

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
    print(f"  [{config_name} s{seed}] pass@1={metric} ({correct}/{total}), n_train={len(texts)}", flush=True)
    return {"pass_at_1": metric, "correct": correct, "total": total, "n_train": len(texts)}

def run_config(config_name, texts, eval_fn, results_dict, results_file):
    if config_name in results_dict:
        print(f"  {config_name}: done ({results_dict[config_name]['pass_at_1']}), skipping", flush=True)
        return

    seed_results = []
    for seed in [42, 123, 271, 456, 2024][:N_SEEDS]:
        result = finetune_and_eval(texts, config_name, eval_fn, seed)
        result['seed'] = seed
        seed_results.append(result)

    values = [r['pass_at_1'] for r in seed_results]
    agg = {
        "config": config_name,
        "n_train": len(texts),
        "pass_at_1": round(np.mean(values), 4),
        "std": round(np.std(values), 4),
        "seeds": seed_results
    }
    results_dict[config_name] = agg
    print(f"  {config_name}: pass@1={agg['pass_at_1']}±{agg['std']}, n={len(texts)}", flush=True)
    with open(results_file, "w") as f:
        json.dump(results_dict, f, indent=2, default=str)

# ============ Archive Cell → Real Data Selection ============
def select_real_by_cells(archive_path, pool_list, domain="code"):
    """Load archive, extract cells, select best real sample per cell."""
    with open(archive_path) as f:
        archive_items = json.load(f)

    # Get cells from archive
    cells = set()
    for item in archive_items:
        if domain == "code":
            desc = compute_descriptors(item.get('prompt', ''), item.get('code', ''))
        else:
            desc = compute_math_descriptors(item.get('question', ''), item.get('answer', ''))
        cells.add(get_cell(desc) if domain == "code" else get_math_cell(desc))

    # Select best real sample per cell
    cell_to_best = {}
    for item in pool_list:
        cell = item['cell']
        if cell in cells:
            if cell not in cell_to_best or item['quality'] > cell_to_best[cell]['quality']:
                cell_to_best[cell] = item

    return list(cell_to_best.values()), len(cells)

# ============ Main Experiment ============
results_file = RESULTS_DIR / f"{DOMAIN if not CONFIG else CONFIG}_results.json"
if results_file.exists():
    with open(results_file) as f:
        all_results = json.load(f)
else:
    all_results = {}

if CONFIG in ("code_ref", "math_ref"):
    # ============ Reference Baselines ============
    print(f"\n--- Reference Baselines ({CONFIG}) ---", flush=True)

    if CONFIG == "code_ref":
        # Full pool
        texts_full = [fmt_code(s['prompt'], s['code']) for s in pool]
        run_config("full_mbpp_pool", texts_full, eval_code, all_results, results_file)

        # Random baselines at key sizes
        for n in [49, 118, 200]:
            random.seed(42)
            subset = random.sample(pool, min(n, len(pool)))
            texts = [fmt_code(s['prompt'], s['code']) for s in subset]
            run_config(f"random_{n}", texts, eval_code, all_results, results_file)

        # QD cell selection from R7 archive → real data
        qd_r7_path = ITER_DIR / "qd_archive_r7.json"
        if qd_r7_path.exists():
            selected, n_cells = select_real_by_cells(qd_r7_path, pool, "code")
            texts_qd = [fmt_code(s['prompt'], s['code']) for s in selected]
            print(f"  QD R7: {n_cells} cells → {len(selected)} real samples selected", flush=True)
            run_config(f"qd_r7_real_{len(selected)}", texts_qd, eval_code, all_results, results_file)

        # Greedy cell selection from R7 archive → real data
        greedy_r7_path = ITER_DIR / "greedy_archive_r7.json"
        if greedy_r7_path.exists():
            selected, n_cells = select_real_by_cells(greedy_r7_path, pool, "code")
            texts_greedy = [fmt_code(s['prompt'], s['code']) for s in selected]
            print(f"  Greedy R7: {n_cells} cells → {len(selected)} real samples selected", flush=True)
            run_config(f"greedy_r7_real_{len(selected)}", texts_greedy, eval_code, all_results, results_file)

    elif CONFIG == "math_ref":
        # Full GSM8K train
        texts_full = [fmt_math(s['question'], s['answer']) for s in math_pool]
        run_config("full_gsm8k_train", texts_full, eval_math, all_results, results_file)

        for n in [22, 30, 200, 500]:
            random.seed(42)
            subset = random.sample(math_pool, min(n, len(math_pool)))
            texts = [fmt_math(s['question'], s['answer']) for s in subset]
            run_config(f"random_{n}", texts, eval_math, all_results, results_file)

        # QD and Greedy R7 cell selection
        for strategy in ["qd", "greedy"]:
            path = ITER_DIR / f"{strategy}_archive_r7.json"
            if path.exists():
                selected, n_cells = select_real_by_cells(path, math_pool, "math")
                texts = [fmt_math(s['question'], s['answer']) for s in selected]
                print(f"  {strategy.upper()} R7: {n_cells} cells → {len(selected)} real samples", flush=True)
                run_config(f"{strategy}_r7_real_{len(selected)}", texts, eval_math, all_results, results_file)

elif DOMAIN in ("code", "math"):
    # ============ Per-Round Experiment (Key Rounds Only) ============
    ROUNDS = [0, 3, 7]  # R0=baseline, R3=greedy freeze, R7=final
    print(f"\n--- Per-Round ({DOMAIN}): key rounds {ROUNDS} ---", flush=True)

    # Pre-select random seeds to ensure same random subset per round
    random.seed(42)

    for strategy in ["greedy", "qd"]:
        print(f"\n  Strategy: {strategy.upper()}", flush=True)
        for rnd in ROUNDS:
            archive_path = ITER_DIR / f"{strategy}_archive_r{rnd}.json"
            if not archive_path.exists():
                print(f"    R{rnd}: archive not found", flush=True)
                continue

            # Select real data by archive cells
            if DOMAIN == "code":
                selected, n_cells = select_real_by_cells(archive_path, pool, "code")
                texts = [fmt_code(s['prompt'], s['code']) for s in selected]
            else:
                selected, n_cells = select_real_by_cells(archive_path, math_pool, "math")
                texts = [fmt_math(s['question'], s['answer']) for s in selected]

            config_name = f"{DOMAIN}_{strategy}_r{rnd}"
            print(f"    R{rnd}: {n_cells} cells → {len(selected)} real samples", flush=True)

            run_config(config_name, texts, eval_code if DOMAIN == "code" else eval_math,
                      all_results, results_file)

# ============ Summary ============
print(f"\n{'='*60}", flush=True)
print(f"RESULTS SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
for k, v in sorted(all_results.items()):
    print(f"  {k}: pass@1={v.get('pass_at_1','?')}±{v.get('std','?')}, n={v.get('n_train','?')}", flush=True)
print(f"{'='*60}", flush=True)

with open(results_file, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"Saved: {results_file}", flush=True)
