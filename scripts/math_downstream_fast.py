"""
Math Domain Downstream Fine-tuning - FAST version (200 test problems, 4 seeds)
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys, json, random, re, torch, numpy as np, time
from pathlib import Path
from collections import Counter, defaultdict
from datasets import load_dataset

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-1___5B-Instruct"
K = 500
N_TEST = 200  # Use 200 test problems for speed
SEEDS = [42, 123, 271, 456]
GRID_RES = 10

GPU_ID = int(os.environ.get("GPU_ID", "3"))
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
DEVICE = "cuda:0"

RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/math_downstream_fast")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"=== Math Downstream FAST (GPU {GPU_ID}) ===", flush=True)
print(f"Config: K={K}, N_TEST={N_TEST}, {len(SEEDS)} seeds", flush=True)

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
    return {'difficulty': difficulty, 'num_steps': min(steps / 10.0, 1.0), 'is_multi_step': is_multi_step}

def get_cell(desc):
    return (int(desc['difficulty'] * GRID_RES), int(desc['num_steps'] * GRID_RES), int(desc['is_multi_step'] * GRID_RES))

print(f"Computing descriptors...", flush=True)
pool = []
for ex in train_ds:
    q, a = ex['question'], ex['answer']
    ans = extract_answer(a)
    if ans is None: continue
    desc = compute_descriptors(q, a)
    quality = min(len(a) / 300.0, 1.0)
    pool.append({'question': q, 'answer': a, 'answer_num': ans, 'descriptors': desc, 'quality': quality})
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
        return f"<|im_start|>system\nSolve the math problem step by step.<|im_end|>\n<|im_start|>user\n{s['question']}<|im_end|>\n<|im_start|>assistant\n{s['answer'][:768]}<|im_end|>"

    ds = Dataset.from_dict({"text": [fmt(s) for s in train_samples]})
    trainer = SFTTrainer(model=model, args=SFTConfig(output_dir=str(output_dir), num_train_epochs=3, per_device_train_batch_size=4, gradient_accumulation_steps=4, learning_rate=2e-4, logging_steps=50, save_strategy="no", bf16=True, report_to="none", max_length=768, dataset_text_field="text", packing=False), train_dataset=ds, processing_class=tokenizer)
    trainer.train()
    model.save_pretrained(output_dir / "lora"); tokenizer.save_pretrained(output_dir / "lora")
    del model, trainer; torch.cuda.empty_cache()
    return output_dir / "lora"

def evaluate(model_path, seed):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, model_path); model.eval()

    correct = total = 0
    per_diff = defaultdict(lambda: {"c": 0, "t": 0})

    for i, ex in enumerate(test_ds):
        if i >= N_TEST: break
        test_ans = extract_answer(ex['answer'])
        desc = compute_descriptors(ex['question'], ex['answer'])
        db = "easy" if desc['difficulty'] < 0.33 else ("medium" if desc['difficulty'] < 0.66 else "hard")

        msgs = [{"role":"system","content":"Solve the math problem step by step."},{"role":"user","content":ex['question']}]
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(txt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=256, temperature=0.0, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)

        nums = re.findall(r'-?[\d,]+\.?\d*', resp.replace(',', ''))
        pred = nums[-1] if nums else None
        ok = False
        if pred and test_ans:
            try:
                if abs(float(pred) - float(test_ans)) < 0.01: ok = True; correct += 1
            except: pass
        total += 1; per_diff[db]["t"] += 1
        if ok: per_diff[db]["c"] += 1
        if (i+1) % 50 == 0: print(f"    {i+1}/{total}, acc={correct/total:.4f}", flush=True)

    del model, base; torch.cuda.empty_cache()
    return {"accuracy": round(correct/total, 4), "correct": correct, "total": total,
            "per_difficulty": {k: v["c"]/max(v["t"],1) for k,v in per_diff.items()}}

# Run
all_results = {}
for si, seed in enumerate(SEEDS):
    print(f"\nSEED {seed} ({si+1}/{len(SEEDS)})", flush=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    methods = {"qd_500": select_qd(pool, K, seed), "greedy_500": select_greedy(pool, K), "random_500": select_random(pool, K, seed)}
    for n, s in methods.items():
        cells = len(set(get_cell(x['descriptors']) for x in s))
        print(f"  [{n}] n={len(s)}, cells={cells}", flush=True)

    results = {}
    for name, sel in methods.items():
        cn = f"{name}_s{seed}"
        print(f"--- {cn} ---", flush=True)
        t0 = time.time()
        mp = finetune(sel, cn, seed)
        print(f"  Train: {time.time()-t0:.0f}s", flush=True)
        t0 = time.time()
        ev = evaluate(mp, seed)
        ev["seed"] = seed; ev["n_train"] = len(sel)
        results[name] = ev
        print(f"  Acc: {ev['accuracy']:.4f} ({ev['correct']}/{ev['total']}), {time.time()-t0:.0f}s", flush=True)
        print(f"  Per-diff: {ev['per_difficulty']}", flush=True)

    all_results[seed] = results
    with open(RESULTS_DIR / "math_downstream_fast_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

# Aggregate
print(f"\nAGGREGATE", flush=True)
for m in ["qd_500", "greedy_500", "random_500"]:
    accs = [all_results[s][m]["accuracy"] for s in SEEDS]
    print(f"{m}: {np.mean(accs):.4f} +/- {np.std(accs):.4f}", flush=True)

try:
    from scipy.stats import wilcoxon
    qa = [all_results[s]["qd_500"]["accuracy"] for s in SEEDS]
    ga = [all_results[s]["greedy_500"]["accuracy"] for s in SEEDS]
    ra = [all_results[s]["random_500"]["accuracy"] for s in SEEDS]
    if len(set(qa)) > 1:
        s1, p1 = wilcoxon(qa, ga); d1 = abs(np.mean(qa)-np.mean(ga))/np.std([a-b for a,b in zip(qa,ga)])
        print(f"QD vs Greedy: p={p1:.4f}, d={d1:.2f}", flush=True)
        s2, p2 = wilcoxon(qa, ra)
        print(f"QD vs Random: p={p2:.4f}", flush=True)
except Exception as e:
    print(f"Stats: {e}", flush=True)

print(f"\nSaved to {RESULTS_DIR / 'math_downstream_fast_results.json'}", flush=True)
