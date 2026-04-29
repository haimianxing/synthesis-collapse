"""
Code Domain 8-Seed Downstream Fine-tuning (Enhanced Statistical Power)
Addresses SAC: Wilcoxon p=0.0625 with n=5, need n=8 for p<0.05

Pool: 500 MBPP sanitized → HumanEval evaluation
Methods: QD-200, Greedy-200, Random-200, Greedy-per-cell-40
8 seeds for proper statistical testing
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys, json, random, re, torch, numpy as np, time, ast
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-1___5B-Instruct"
SEEDS = [42, 123, 271, 456, 789, 2024, 314, 159]
GRID_RES = 10
GPU_ID = int(os.environ.get("GPU_ID", "4"))
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
DEVICE = "cuda:0"

RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/code_8seed")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"=== Code Domain 8-Seed (GPU {GPU_ID}) ===", flush=True)

# Load datasets
print("Loading datasets...", flush=True)
try:
    mbpp_train = load_dataset("mbpp", "sanitized", split="test")
except:
    mbpp_train = load_dataset("mbpp", split="test")
humaneval = load_dataset("openai_humaneval", split="test")
print(f"MBPP: {len(mbpp_train)}, HumanEval: {len(humaneval)}", flush=True)

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

pool = []
for ex in mbpp_train:
    code = ex.get('code', '')
    desc = compute_code_descriptors(ex.get('prompt', ''), code, ex.get('test_list', []))
    quality = min(len(code) / 500.0, 1.0) if code else 0.1
    pool.append({'prompt': ex.get('prompt', ''), 'code': code, 'text': ex.get('text', ''),
                 'descriptors': desc, 'quality': quality})
print(f"Pool: {len(pool)}", flush=True)

def select_qd(items, k, seed=42):
    grid = {}
    for item in items:
        cell = get_cell(item['descriptors'])
        if cell not in grid or item['quality'] > grid[cell]['quality']:
            grid[cell] = item
    return sorted(grid.values(), key=lambda x: x['quality'], reverse=True)[:k]

def select_greedy(items, k):
    return sorted(items, key=lambda x: x['quality'], reverse=True)[:k]

def select_random(items, k, seed=42):
    return random.Random(seed).sample(items, k)

def select_greedy_per_cell(items, n_cells):
    sorted_items = sorted(items, key=lambda x: x['quality'], reverse=True)
    seen = set(); sel = []
    for item in sorted_items:
        cell = get_cell(item['descriptors'])
        if cell not in seen:
            seen.add(cell); sel.append(item)
            if len(sel) >= n_cells: break
    return sel

def finetune(train_samples, config_name, seed):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    output_dir = RESULTS_DIR / f"model_{config_name}"
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
    for ex in humaneval:
        msgs = [{"role":"system","content":"Complete the Python function. Return only the function body."},{"role":"user","content":ex['prompt']}]
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(txt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=256, temperature=0.0, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        try:
            exec_globals = {}; exec(ex['prompt'] + resp, exec_globals); exec(ex['test'], exec_globals)
            correct += 1
        except: pass
        total += 1

    del model, base; torch.cuda.empty_cache()
    return {"pass_at_1": round(correct/total, 4), "correct": correct, "total": total}

# Determine QD cell count
qd_test = select_qd(pool, 200, 42)
n_qd_cells = len(set(get_cell(x['descriptors']) for x in qd_test))
print(f"QD unique cells: {n_qd_cells}", flush=True)

all_results = {}
for si, seed in enumerate(SEEDS):
    print(f"\n{'='*60}\nSEED {seed} ({si+1}/{len(SEEDS)})\n{'='*60}", flush=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    methods = {
        "qd_200": select_qd(pool, 200, seed),
        "greedy_200": select_greedy(pool, 200),
        "random_200": select_random(pool, 200, seed),
        "greedy_per_cell": select_greedy_per_cell(pool, n_qd_cells),
    }

    results = {}
    for name, sel in methods.items():
        cn = f"{name}_s{seed}"
        cells = len(set(get_cell(x['descriptors']) for x in sel))
        print(f"--- {cn}: n={len(sel)}, cells={cells} ---", flush=True)
        t0 = time.time()
        mp = finetune(sel, cn, seed)
        ev = evaluate_humaneval(mp, seed)
        ev["seed"] = seed; ev["n_train"] = len(sel); ev["n_cells"] = cells
        results[name] = ev
        print(f"  pass@1: {ev['pass_at_1']:.4f} ({ev['correct']}/{ev['total']}), {time.time()-t0:.0f}s", flush=True)

    all_results[seed] = results
    with open(RESULTS_DIR / "code_8seed_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

# Aggregate
print(f"\n{'='*60}\nAGGREGATE\n{'='*60}", flush=True)
for m in ["qd_200", "greedy_200", "random_200", "greedy_per_cell"]:
    accs = [all_results[s][m]["pass_at_1"] for s in SEEDS]
    ns = [all_results[s][m]["n_train"] for s in SEEDS]
    print(f"{m}: {np.mean(accs):.4f} ± {np.std(accs):.4f} (n={ns[0]})", flush=True)

try:
    from scipy.stats import wilcoxon
    qa = [all_results[s]["qd_200"]["pass_at_1"] for s in SEEDS]
    ga = [all_results[s]["greedy_200"]["pass_at_1"] for s in SEEDS]
    ra = [all_results[s]["random_200"]["pass_at_1"] for s in SEEDS]
    gca = [all_results[s]["greedy_per_cell"]["pass_at_1"] for s in SEEDS]

    print(f"\nQD vs Greedy-200:", flush=True)
    s1, p1 = wilcoxon(qa, ga); d1 = abs(np.mean(qa)-np.mean(ga))/np.std([a-b for a,b in zip(qa,ga)])
    print(f"  p={p1:.4f}, d={d1:.2f}, QD wins {sum(1 for a,b in zip(qa,ga) if a>b)}/{len(SEEDS)}", flush=True)

    print(f"QD vs Greedy-per-cell:", flush=True)
    s2, p2 = wilcoxon(qa, gca); d2 = abs(np.mean(qa)-np.mean(gca))/np.std([a-b for a,b in zip(qa,gca)])
    print(f"  p={p2:.4f}, d={d2:.2f}, QD wins {sum(1 for a,b in zip(qa,gca) if a>b)}/{len(SEEDS)}", flush=True)

    print(f"QD vs Random:", flush=True)
    s3, p3 = wilcoxon(qa, ra)
    print(f"  p={p3:.4f}, QD wins {sum(1 for a,b in zip(qa,ra) if a>b)}/{len(SEEDS)}", flush=True)
except Exception as e:
    print(f"Stats error: {e}", flush=True)

print(f"\nSaved to {RESULTS_DIR / 'code_8seed_results.json'}", flush=True)
