"""
Reference Baselines for Per-Round Downstream
Provides context for per-round numbers with upper/lower bounds.

GPU 5: Code baselines (Full MBPP, Random-118, QD-200 from real MBPP)
GPU 6: Math baselines (Full GSM8K-500, Random-30)
GPU 7: Scaled experiments (Cumulative archives)
"""
import os, sys, json, random, re, torch, numpy as np, ast
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

GPU_ID = int(os.environ.get("GPU_ID", "5"))
CONFIG = os.environ.get("CONFIG", "code_baselines")

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-1___5B-Instruct"
DEVICE = "cuda:0"
N_SEEDS = 3
GRID_RES = 10

RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/per_round_v2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

results_file = RESULTS_DIR / f"ref_{CONFIG}_results.json"
if results_file.exists():
    with open(results_file) as f:
        all_results = json.load(f)
    print(f"Loaded {len(all_results)} existing results", flush=True)
else:
    all_results = {}

print(f"=== Reference Baselines: {CONFIG} (GPU {GPU_ID}) ===", flush=True)

# ============ Shared training function ============
def finetune_and_eval(texts, config_name, eval_fn, seed=42):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    output_dir = RESULTS_DIR / f"model_ref_{config_name}_s{seed}"

    print(f"  [{config_name} s{seed}] Training on {len(texts)} samples", flush=True)

    if len(texts) < 5:
        return {"pass_at_1": 0, "accuracy": 0, "n_train": len(texts)}

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
        bf16=True, report_to="none", max_length=1024,
        dataset_text_field="text", packing=False),
        train_dataset=ds, processing_class=tokenizer)
    trainer.train()

    model.eval()
    correct, total = eval_fn(model, tokenizer)

    del model; torch.cuda.empty_cache()
    metric = round(correct/total, 4) if total > 0 else 0
    print(f"  [{config_name} s{seed}] {correct}/{total} = {metric}", flush=True)
    return {"pass_at_1": metric, "accuracy": metric, "correct": correct, "total": total, "n_train": len(texts)}

def run_config(config_name, texts, eval_fn):
    if config_name in all_results:
        print(f"  {config_name}: already done, skipping", flush=True)
        return

    seed_results = []
    for seed in [42, 123, 271][:N_SEEDS]:
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
    all_results[config_name] = agg
    print(f"  {config_name}: pass@1={agg['pass_at_1']}±{agg['std']}, n={len(texts)}", flush=True)

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

# ============ Code Baselines ============
if CONFIG == "code_baselines":
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

    def fmt_code(prompt, code):
        return f"<|im_start|>system\nComplete the Python function.<|im_end|>\n<|im_start|>user\n{prompt[:512]}<|im_end|>\n<|im_start|>assistant\n{code[:1024]}<|im_end|>"

    print("Loading MBPP + HumanEval...", flush=True)
    try:
        mbpp = load_dataset("mbpp", "sanitized", split="test")
    except:
        mbpp = load_dataset("mbpp", split="test")
    humaneval = load_dataset("openai_humaneval", split="test")
    print(f"MBPP: {len(mbpp)}, HumanEval: {len(humaneval)}", flush=True)

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

    # Build pool
    pool = []
    for ex in mbpp:
        code = ex.get('code', '')
        prompt = ex.get('prompt', '')
        if code and len(code) > 20:
            pool.append({'prompt': prompt, 'code': code})

    print(f"\nPool: {len(pool)} valid MBPP samples", flush=True)

    # Config 1: Full MBPP (all ~257 samples) - upper bound
    texts_full = [fmt_code(s['prompt'], s['code']) for s in pool]
    run_config("full_mbpp", texts_full, eval_code)

    # Config 2: Random 118 from MBPP - random baseline for QD R7 size
    random.seed(42)
    random_118 = random.sample(pool, min(118, len(pool)))
    texts_r118 = [fmt_code(s['prompt'], s['code']) for s in random_118]
    run_config("random_118", texts_r118, eval_code)

    # Config 3: Random 49 from MBPP - random baseline for Greedy R7 size
    random.seed(42)
    random_49 = random.sample(pool, min(49, len(pool)))
    texts_r49 = [fmt_code(s['prompt'], s['code']) for s in random_49]
    run_config("random_49", texts_r49, eval_code)

    # Config 4: QD-cell selected 200 from MBPP - reproduce main result
    # Load QD archive to get cells, then select matching MBPP samples
    qd_archive_path = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/code_iterative_v2/qd_archive_r7.json")
    if qd_archive_path.exists():
        with open(qd_archive_path) as f:
            qd_archive = json.load(f)
        qd_cells = set()
        for item in qd_archive:
            desc = compute_descriptors(item.get('prompt', ''), item.get('code', ''))
            qd_cells.add(get_cell(desc))

        # Select best item per cell from pool
        cell_to_best = {}
        for item in pool:
            desc = compute_descriptors(item['prompt'], item['code'])
            cell = get_cell(desc)
            if cell in qd_cells:
                quality = min(len(item['code']) / 500.0, 1.0)
                if cell not in cell_to_best or quality > cell_to_best[cell][1]:
                    cell_to_best[cell] = (item, quality)

        qd_selected = [v[0] for v in cell_to_best.values()]
        texts_qd200 = [fmt_code(s['prompt'], s['code']) for s in qd_selected]
        print(f"\n  QD-cell matched: {len(qd_selected)} samples from {len(qd_cells)} cells", flush=True)
        run_config("qd_cell_matched", texts_qd200, eval_code)

# ============ Math Baselines ============
elif CONFIG == "math_baselines":
    def extract_answer(text):
        match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
        return match.group(1).replace(',', '') if match else None

    def fmt_math(question, answer):
        return f"<|im_start|>system\nSolve the math problem step by step.<|im_end|>\n<|im_start|>user\n{question[:512]}<|im_end|>\n<|im_start|>assistant\n{answer[:1024]}<|im_end|>"

    print("Loading GSM8K...", flush=True)
    gsm8k_pool = load_dataset("gsm8k", "main", split="test")
    gsm8k_test_full = load_dataset("gsm8k", "main", split="train")
    random.seed(42)
    gsm8k_test = random.sample(list(gsm8k_test_full), min(200, len(gsm8k_test_full)))

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

    # Build pool
    pool = []
    for ex in gsm8k_pool:
        q, a = ex['question'], ex['answer']
        ans = extract_answer(a)
        if ans and len(a) > 20:
            pool.append({'question': q, 'answer': a})

    print(f"\nPool: {len(pool)} valid GSM8K samples", flush=True)

    # Config 1: Full GSM8K test (all ~1319 samples)
    texts_full = [fmt_math(s['question'], s['answer']) for s in pool]
    run_config("full_gsm8k", texts_full, eval_math)

    # Config 2: Random 30 - baseline for QD R7 size
    random.seed(42)
    random_30 = random.sample(pool, min(30, len(pool)))
    texts_r30 = [fmt_math(s['question'], s['answer']) for s in random_30]
    run_config("random_30", texts_r30, eval_math)

    # Config 3: Random 22 - baseline for Greedy R7 size
    random.seed(42)
    random_22 = random.sample(pool, min(22, len(pool)))
    texts_r22 = [fmt_math(s['question'], s['answer']) for s in random_22]
    run_config("random_22", texts_r22, eval_math)

# ============ Scaled Cumulative Experiments ============
elif CONFIG == "scaled":
    # Train on cumulative archives: R0, R0+R1, ..., R0+R1+...+R7
    ITER_DIR_CODE = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/code_iterative_v2")
    ITER_DIR_MATH = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/math_iterative_v2")

    def extract_answer(text):
        match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
        return match.group(1).replace(',', '') if match else None

    # Code scaled
    print("\n--- Code Scaled (Cumulative) ---", flush=True)
    humaneval = load_dataset("openai_humaneval", split="test")

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

    for strategy in ["greedy", "qd"]:
        # Load all rounds' archives
        all_items = []
        for rnd in range(8):
            path = ITER_DIR_CODE / f"{strategy}_archive_r{rnd}.json"
            if path.exists():
                with open(path) as f:
                    items = json.load(f)
                # Use the final archive (which is cumulative)
                all_items = items  # Each archive IS the cumulative state

                valid = [i for i in items if i.get('code') and len(i.get('code', '')) > 20]
                texts = [f"<|im_start|>system\nComplete the Python function.<|im_end|>\n<|im_start|>user\n{i.get('prompt','')[:512]}<|im_end|>\n<|im_start|>assistant\n{i.get('code','')[:1024]}<|im_end|>" for i in valid]

                config_name = f"scaled_code_{strategy}_r{rnd}"
                run_config(config_name, texts, eval_code)

# ============ Summary ============
print(f"\n{'='*60}", flush=True)
print(f"REFERENCE BASELINES ({CONFIG})", flush=True)
print(f"{'='*60}", flush=True)
for key, r in sorted(all_results.items()):
    print(f"  {key}: pass@1={r.get('pass_at_1', r.get('accuracy', '?'))}±{r.get('std', '?')}, n={r.get('n_train', '?')}", flush=True)

with open(results_file, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"Results saved: {results_file}", flush=True)
