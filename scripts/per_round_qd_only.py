"""
Per-Round QD-only downstream (parallel version)
Runs only the QD rounds, skipping already-completed greedy rounds.
Each round uses a separate GPU for parallelism.
"""
import os, sys, json, random, re, torch, numpy as np, ast, time
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

GPU_ID = int(os.environ.get("GPU_ID", "3"))
DOMAIN = os.environ.get("DOMAIN", "code")  # "code" or "math"
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-1___5B-Instruct"
GRID_RES = 10
DEVICE = "cuda:0"
N_SEEDS = 3

RESULTS_DIR = Path(f"/mnt/data2/zcz/neurIps-emnlp/neurips/results/{DOMAIN}_per_round_real")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load existing results
results_file = RESULTS_DIR / f"{DOMAIN}_per_round_real_results.json"
if results_file.exists():
    with open(results_file) as f:
        all_results = json.load(f)
    print(f"Loaded {len(all_results)} existing results", flush=True)
else:
    all_results = {}

print(f"=== Per-Round QD-only: {DOMAIN.upper()} (GPU {GPU_ID}) ===", flush=True)

# ============ Domain-specific setup (same as original) ============
if DOMAIN == "code":
    ITER_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/code_iterative_v2")
    ROUNDS_TO_TEST = [0, 1, 2, 3]  # QD rounds

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
        return min(len(sample.get('code', '')) / 500.0, 1.0) if sample.get('code') else 0.1

    def fmt_sample(sample):
        return f"<|im_start|>system\nComplete the Python function.<|im_end|>\n<|im_start|>user\n{sample.get('prompt', '')[:512]}<|im_end|>\n<|im_start|>assistant\n{sample.get('code', '')[:768]}<|im_end|>"

    print("Loading MBPP + HumanEval...", flush=True)
    try:
        mbpp = load_dataset("mbpp", "sanitized", split="test")
    except:
        mbpp = load_dataset("mbpp", split="test")
    humaneval = load_dataset("openai_humaneval", split="test")
    print(f"MBPP: {len(mbpp)}, HumanEval: {len(humaneval)}", flush=True)

    pool = []
    for ex in mbpp:
        code = ex.get('code', '')
        desc = compute_descriptors(ex.get('prompt', ''), code)
        pool.append({'prompt': ex.get('prompt', ''), 'code': code, 'text': ex.get('text', ''),
                     'descriptors': desc, 'quality': compute_quality({'code': code})})

    def eval_model(model, tokenizer):
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

elif DOMAIN == "math":
    ITER_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/math_iterative_v2")
    ROUNDS_TO_TEST = [0, 1, 2, 3, 5]

    def compute_descriptors(question, answer, **kwargs):
        steps = answer.count('<<') + 1
        difficulty = min(len(answer) / 500.0, 1.0)
        is_multi = 1 if steps >= 3 else 0
        return {'difficulty': difficulty, 'num_steps': min(steps / 10.0, 1.0), 'is_multi_step': is_multi}

    def get_cell(desc):
        return (int(desc['difficulty'] * GRID_RES), int(desc['num_steps'] * GRID_RES), int(desc['is_multi_step'] * GRID_RES))

    def compute_quality(sample):
        return min(len(sample.get('answer', '')) / 300.0, 1.0)

    def fmt_sample(sample):
        return f"<|im_start|>system\nSolve the math problem step by step.<|im_end|>\n<|im_start|>user\n{sample.get('question','')[:512]}<|im_end|>\n<|im_start|>assistant\n{sample.get('answer','')[:768]}<|im_end|>"

    def extract_answer(text):
        match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
        return match.group(1).replace(',', '') if match else None

    print("Loading GSM8K...", flush=True)
    gsm8k_pool = load_dataset("gsm8k", "main", split="test")
    gsm8k_test_full = load_dataset("gsm8k", "main", split="train")
    random.seed(42)
    gsm8k_test = random.sample(list(gsm8k_test_full), min(200, len(gsm8k_test_full)))
    print(f"GSM8K pool: {len(gsm8k_pool)}, eval: {len(gsm8k_test)}", flush=True)

    pool = []
    for ex in gsm8k_pool:
        q, a = ex['question'], ex['answer']
        ans = extract_answer(a)
        if ans is None: continue
        desc = compute_descriptors(q, a)
        pool.append({'question': q, 'answer': a, 'answer_num': ans,
                     'descriptors': desc, 'quality': compute_quality({'answer': a})})

    def eval_model(model, tokenizer):
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

# ============ Archive-guided selection ============
def load_archive_cells(archive_path):
    with open(archive_path) as f:
        items = json.load(f)
    cells = set()
    for item in items:
        if DOMAIN == "code":
            desc = compute_descriptors(item.get('prompt', ''), item.get('code', ''))
        else:
            desc = compute_descriptors(item.get('question', ''), item.get('answer', ''))
        cells.add(get_cell(desc))
    return cells, len(items)

def select_real_data_for_cells(cells):
    cell_to_best = {}
    for item in pool:
        cell = get_cell(item['descriptors'])
        if cell in cells:
            if cell not in cell_to_best or item['quality'] > cell_to_best[cell]['quality']:
                cell_to_best[cell] = item
    return list(cell_to_best.values())

# ============ Fine-tuning and evaluation ============
def finetune_and_eval(train_data, config_name, seed=42):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    output_dir = RESULTS_DIR / f"model_{config_name}_s{seed}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
                                                  device_map=DEVICE, trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj"], lora_dropout=0.05, task_type="CAUSAL_LM"))

    valid = [s for s in train_data if (DOMAIN == "code" and s.get('code') and len(s['code']) > 20)
             or (DOMAIN == "math" and s.get('answer') and len(s['answer']) > 20)]
    texts = [fmt_sample(s) for s in valid]
    print(f"  [{config_name} s{seed}] Training on {len(texts)} real samples", flush=True)

    if len(texts) < 5:
        del model; torch.cuda.empty_cache()
        metric = 0
        metric_name = "pass_at_1" if DOMAIN == "code" else "accuracy"
        return {metric_name: 0, "correct": 0, "total": 0, "n_train": len(texts)}

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
    correct, total = eval_model(model, tokenizer)

    del model; torch.cuda.empty_cache()
    metric = round(correct/total, 4) if total > 0 else 0
    metric_name = "pass_at_1" if DOMAIN == "code" else "accuracy"
    print(f"  [{config_name} s{seed}] {metric_name}={metric} ({correct}/{total})", flush=True)
    return {metric_name: metric, "correct": correct, "total": total, "n_train": len(texts)}

# ============ Main: QD rounds only ============
print(f"\nRunning QD rounds: {ROUNDS_TO_TEST}", flush=True)

for rnd in ROUNDS_TO_TEST:
    key = f"qd_r{rnd}"
    if key in all_results:
        print(f"  {key}: already done, skipping", flush=True)
        continue

    archive_path = ITER_DIR / f"qd_archive_r{rnd}.json"
    if not archive_path.exists():
        print(f"  R{rnd}: archive not found, skipping", flush=True)
        continue

    cells, n_archive = load_archive_cells(archive_path)
    real_data = select_real_data_for_cells(cells)
    n_matched = len(real_data)

    print(f"\n  R{rnd}: archive={n_archive} items, {len(cells)} cells → {n_matched} real samples selected", flush=True)

    if n_matched < 5:
        print(f"  R{rnd}: too few real samples ({n_matched}), skipping", flush=True)
        continue

    seed_results = []
    for seed in [42, 123, 271][:N_SEEDS]:
        result = finetune_and_eval(real_data, f"qd_r{rnd}", seed)
        result['seed'] = seed
        seed_results.append(result)

    metric_name = "pass_at_1" if DOMAIN == "code" else "accuracy"
    values = [r[metric_name] for r in seed_results]
    agg = {
        "round": rnd, "strategy": "qd",
        "n_cells": len(cells), "n_archive": n_archive,
        "n_real_selected": n_matched,
        metric_name: round(np.mean(values), 4),
        "std": round(np.std(values), 4),
        "seeds": seed_results
    }
    all_results[key] = agg
    print(f"  R{rnd}: {metric_name}={agg[metric_name]}±{agg['std']}, cells={len(cells)}, real={n_matched}", flush=True)

    # Save after each round
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

# ============ Summary ============
print(f"\n{'='*60}", flush=True)
print(f"PER-ROUND RESULTS ({DOMAIN.upper()})", flush=True)
print(f"{'='*60}", flush=True)
metric_name = "pass_at_1" if DOMAIN == "code" else "accuracy"
for strategy in ["greedy", "qd"]:
    print(f"\n  {strategy.upper()}:", flush=True)
    for key in sorted(all_results.keys()):
        if key.startswith(strategy):
            r = all_results[key]
            print(f"    R{r['round']}: {metric_name}={r[metric_name]}±{r['std']}, cells={r['n_cells']}, real={r['n_real_selected']}", flush=True)
print(f"{'='*60}", flush=True)

with open(results_file, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"Results saved: {results_file}", flush=True)
