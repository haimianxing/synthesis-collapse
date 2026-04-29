"""
Code Domain MATCHED Comparison: QD-40 vs Greedy-40 vs QD-200 vs Greedy-200
Addresses SAC C2: Is QD's advantage from diversity or sample size?

Key comparison: QD-40 (40 unique cells) vs Greedy-40 (top-40 by quality, per-cell best)
If QD-40 > Greedy-40, the advantage is from diversity, not sample size.

5 seeds, GPU auto-assigned via CUDA_VISIBLE_DEVICES
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys, json, random, re, torch, numpy as np, time, ast
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-1___5B-Instruct"
SEEDS = [42, 123, 271, 456, 2024]
GRID_RES = 10
GPU_ID = int(os.environ.get("GPU_ID", "2"))
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
DEVICE = "cuda:0"

RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/code_matched")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"=== Code Domain MATCHED Comparison (GPU {GPU_ID}) ===", flush=True)
print(f"Config: 5 seeds, 4 methods", flush=True)

# Load MBPP
print("Loading MBPP...", flush=True)
try:
    mbpp_train = load_dataset("mbpp", "sanitized", split="test")
except:
    mbpp_train = load_dataset("mbpp", split="test")
print(f"MBPP pool: {len(mbpp_train)}", flush=True)

# Load HumanEval
print("Loading HumanEval...", flush=True)
humaneval = load_dataset("openai_humaneval", split="test")
print(f"HumanEval: {len(humaneval)}", flush=True)

def compute_code_descriptors(prompt, code, test_list):
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

# Build pool
pool = []
for ex in mbpp_train:
    prompt = ex.get('prompt', '')
    code = ex.get('code', '')
    text = ex.get('text', '')
    desc = compute_code_descriptors(prompt, code, ex.get('test_list', []))
    quality = min(len(code) / 500.0, 1.0) if code else 0.1
    pool.append({'prompt': prompt, 'code': code, 'text': text, 'descriptors': desc, 'quality': quality})
print(f"Pool: {len(pool)}", flush=True)

# Selection strategies
def select_qd(items, k, seed=42):
    """QD: best per cell, then top-k by quality"""
    grid = {}
    for item in items:
        cell = get_cell(item['descriptors'])
        if cell not in grid or item['quality'] > grid[cell]['quality']:
            grid[cell] = item
    return sorted(grid.values(), key=lambda x: x['quality'], reverse=True)[:k]

def select_greedy_per_cell(items, n_cells):
    """Greedy per-cell: from the greedy top selection, pick best-1 per cell → ~n_cells samples.
    This simulates what happens if Greedy used cell-aware dedup."""
    # First get top-quality items
    sorted_items = sorted(items, key=lambda x: x['quality'], reverse=True)
    # Pick best-1 per cell
    seen_cells = set()
    selected = []
    for item in sorted_items:
        cell = get_cell(item['descriptors'])
        if cell not in seen_cells:
            seen_cells.add(cell)
            selected.append(item)
            if len(selected) >= n_cells:
                break
    return selected

def select_greedy(items, k):
    return sorted(items, key=lambda x: x['quality'], reverse=True)[:k]

def select_random(items, k, seed=42):
    return random.Random(seed).sample(items, k)

# Fine-tuning
def finetune(train_samples, config_name, seed):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    output_dir = RESULTS_DIR / f"model_{config_name}_seed{seed}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"], lora_dropout=0.05, task_type="CAUSAL_LM"))

    def fmt(s):
        return f"<|im_start|>system\nComplete the Python function.<|im_end|>\n<|im_start|>user\n{s['prompt'][:512]}<|im_end|>\n<|im_start|>assistant\n{s['code'][:768]}<|im_end|>"

    ds = Dataset.from_dict({"text": [fmt(s) for s in train_samples]})
    trainer = SFTTrainer(model=model, args=SFTConfig(output_dir=str(output_dir), num_train_epochs=3, per_device_train_batch_size=4, gradient_accumulation_steps=4, learning_rate=2e-4, logging_steps=50, save_strategy="no", bf16=True, report_to="none", max_length=768, dataset_text_field="text", packing=False), train_dataset=ds, processing_class=tokenizer)
    trainer.train()
    model.save_pretrained(output_dir / "lora"); tokenizer.save_pretrained(output_dir / "lora")
    del model, trainer; torch.cuda.empty_cache()
    return output_dir / "lora"

def evaluate_humaneval(model_path, seed):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, model_path); model.eval()

    correct = total = 0
    for i, ex in enumerate(humaneval):
        msgs = [{"role":"system","content":"Complete the Python function. Return only the function body."},{"role":"user","content":ex['prompt']}]
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(txt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=256, temperature=0.0, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        full_code = ex['prompt'] + resp
        try:
            exec_globals = {}; exec(full_code, exec_globals); exec(ex['test'], exec_globals)
            correct += 1
        except: pass
        total += 1
        if (i+1) % 50 == 0: print(f"    [{model_path.parent.name}] {i+1}/{total}, pass@1={correct/total:.4f}", flush=True)

    del model, base; torch.cuda.empty_cache()
    return {"pass_at_1": round(correct/total, 4), "correct": correct, "total": total}

# Determine QD cell count for matching
qd_full = select_qd(pool, 200, seed=42)
n_qd_cells = len(set(get_cell(x['descriptors']) for x in qd_full))
print(f"\nQD-200 unique cells: {n_qd_cells}", flush=True)

# Run experiments
all_results = {}
for si, seed in enumerate(SEEDS):
    print(f"\n{'='*60}", flush=True)
    print(f"SEED {seed} ({si+1}/{len(SEEDS)})", flush=True)
    print(f"{'='*60}", flush=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # 4 methods
    qd_200 = select_qd(pool, 200, seed)
    qd_40 = qd_200[:n_qd_cells]  # QD's actual output (all unique cells)
    greedy_200 = select_greedy(pool, 200)
    greedy_40 = select_greedy_per_cell(pool, n_qd_cells)  # Matched cell count from greedy

    methods = {
        f"qd_{len(qd_40)}": qd_40,
        f"greedy_per_cell_{len(greedy_40)}": greedy_40,
        "qd_200": qd_200,
        "greedy_200": greedy_200,
    }

    for name, sel in methods.items():
        cells = len(set(get_cell(x['descriptors']) for x in sel))
        print(f"  [{name}] n={len(sel)}, cells={cells}, avg_q={np.mean([x['quality'] for x in sel]):.3f}", flush=True)

    results = {}
    for name, sel in methods.items():
        cn = f"{name}_s{seed}"
        print(f"\n--- {cn} ---", flush=True)
        t0 = time.time()
        mp = finetune(sel, cn, seed)
        print(f"  Train: {time.time()-t0:.0f}s", flush=True)
        t0 = time.time()
        ev = evaluate_humaneval(mp, seed)
        ev["seed"] = seed; ev["n_train"] = len(sel)
        ev["n_cells"] = len(set(get_cell(x['descriptors']) for x in sel))
        results[name] = ev
        print(f"  pass@1: {ev['pass_at_1']:.4f} ({ev['correct']}/{ev['total']}), n={ev['n_train']}, cells={ev['n_cells']}, {time.time()-t0:.0f}s", flush=True)

    all_results[seed] = results
    with open(RESULTS_DIR / "code_matched_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

# Aggregate
print(f"\n{'='*60}", flush=True)
print("AGGREGATE RESULTS", flush=True)
print(f"{'='*60}", flush=True)

method_names = list(all_results[SEEDS[0]].keys())
for m in method_names:
    accs = [all_results[s][m]["pass_at_1"] for s in SEEDS]
    ns = [all_results[s][m]["n_train"] for s in SEEDS]
    cs = [all_results[s][m]["n_cells"] for s in SEEDS]
    print(f"{m}: {np.mean(accs):.4f} ± {np.std(accs):.4f} (n={ns[0]}, cells={cs[0]})", flush=True)

# Key comparison: QD-40 vs Greedy-per-cell-40
try:
    from scipy.stats import wilcoxon
    qd_key = [k for k in method_names if k.startswith("qd_") and not k.endswith("200")][0]
    greedy_key = [k for k in method_names if k.startswith("greedy_per_cell")][0]
    qa = [all_results[s][qd_key]["pass_at_1"] for s in SEEDS]
    ga = [all_results[s][greedy_key]["pass_at_1"] for s in SEEDS]
    print(f"\nKEY COMPARISON: {qd_key} vs {greedy_key}", flush=True)
    print(f"  QD: {qa}", flush=True)
    print(f"  Greedy-per-cell: {ga}", flush=True)
    wins = sum(1 for a, b in zip(qa, ga) if a > b)
    print(f"  QD wins {wins}/{len(SEEDS)} seeds", flush=True)
    if len(set(qa)) > 1 and len(set(ga)) > 1:
        s, p = wilcoxon(qa, ga)
        print(f"  Wilcoxon p={p:.4f}", flush=True)
    else:
        print(f"  Wilcoxon: all equal or constant, no test", flush=True)
except Exception as e:
    print(f"Stats error: {e}", flush=True)

print(f"\nSaved to {RESULTS_DIR / 'code_matched_results.json'}", flush=True)
