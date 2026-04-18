"""
Self-Synthesis v3: Base-Reset Experiment
=========================================
CRITICAL FIX for cumulative LoRA drift confound.

Key difference from v2:
- v2 (drift path): R0 trains from base, R1 trains from merged_R0, R2 from merged_R1...
  → Confounds data quality with parameter drift
- v3 (base-reset): EVERY round trains from BASE model with ALL accumulated data
  → Isolates data quality/selection effect from parameter drift

Design:
  R0: base model → generate → select → save data
  R1: base model → train on [R0 data + R1 generated data] → eval
  R2: base model → train on [R0+R1 data + R2 generated data] → eval
  ...

This reveals: how much of v2's convergence/collapse is due to drift vs data quality?

GPU Allocation (3 parallel runs after v2 completes):
  GPU 1: STRATEGY=greedy  SEED=42
  GPU 2: STRATEGY=qd      SEED=42
  GPU 3: STRATEGY=random  SEED=42

Usage:
  CUDA_VISIBLE_DEVICES=1 GPU_ID=0 STRATEGY=greedy SEED=42 PYTHONUNBUFFERED=1 python -u self_synthesis_base_reset.py &
"""
import os, sys, json, random, re, torch, numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import time, math

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

GPU_ID = int(os.environ.get("GPU_ID", "0"))
STRATEGY = os.environ.get("STRATEGY", "greedy")
SEED = int(os.environ.get("SEED", "42"))

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-7B-Instruct"
DEVICE = "cuda:0"
GRID_RES = 10
N_ROUNDS = 5
N_GENERATE = 1000
N_SELECT = 300
N_EVAL = 500

RESULTS_DIR = Path(f"/mnt/data2/zcz/neurIps-emnlp/neurips/results/self_synthesis_v3_base_reset/{STRATEGY}_s{SEED}")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"=== Self-Synthesis v3 (Base-Reset): {STRATEGY.upper()} seed={SEED} (GPU {GPU_ID}) ===", flush=True)
print(f"  Key: Each round trains from BASE model with accumulated data", flush=True)
print(f"  N_GENERATE={N_GENERATE}, N_SELECT={N_SELECT}, N_ROUNDS={N_ROUNDS}", flush=True)

# ============ Reuse descriptor/quality functions from v2 ============
def get_math_cell(answer):
    if not answer or len(answer) < 20: return None
    steps = answer.count('<<') + answer.count('\n') + 1
    difficulty = min(len(answer) / 800.0, 1.0)
    struct = 0 if steps <= 2 else (1 if steps <= 6 else 2)
    return (int(difficulty * GRID_RES), int(min(steps / 15.0, 1.0) * GRID_RES), struct * (GRID_RES // 2))

def extract_answer(text):
    m = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    return m.group(1).replace(',', '') if m else None

def extract_flexible(text):
    for pattern in [
        r'####\s*(-?[\d,]+\.?\d*)',
        r'(?:answer is|answer:)\s*\$?(-?[\d,]+\.?\d*)',
        r'\\boxed\{(-?[\d,]+\.?\d*)\}',
        r'(?:=|is)\s*(-?[\d,]+\.?\d*)',
    ]:
        m = re.search(pattern, text, re.IGNORECASE if 'answer' in pattern else 0)
        if m: return m.group(1).replace(',', '')
    nums = re.findall(r'-?[\d,]+\.?\d*', text.strip())
    return nums[-1].replace(',', '') if nums else None

def quality_score(answer, gold_answer=None):
    if not answer or len(answer) < 20: return 0
    score = 0
    if '####' in answer: score += 0.2
    score += min(answer.count('<<') / 5.0, 0.2)
    score += min(len(answer) / 800.0, 0.15)
    pred = extract_flexible(answer)
    if pred: score += 0.15
    if gold_answer and pred:
        try:
            if abs(float(pred.strip()) - float(gold_answer.strip())) < 1e-6: score += 0.3
        except:
            if pred.strip() == gold_answer.strip(): score += 0.3
    return min(score, 1.0)

def check_correct(pred, gold):
    if not pred or not gold: return False
    try:
        return abs(float(pred.strip()) - float(gold.strip())) < 1e-6
    except:
        return pred.strip() == gold.strip()

# ============ Diversity Metrics ============
def compute_cell_entropy(solutions):
    cells = [s['cell'] for s in solutions if s['cell']]
    if not cells: return 0
    counts = Counter(cells)
    total = len(cells)
    return -sum((c/total) * math.log2(c/total) for c in counts.values() if c > 0)

def compute_unique_strategies(solutions):
    structures = set()
    for s in solutions:
        ans = s.get('answer', '')
        if not ans or len(ans) < 20: continue
        steps = ans.count('<<') + ans.count('\n') + 1
        has_calc = '<<' in ans or '=' in ans
        has_final = '####' in ans
        length_bin = min(int(len(ans) / 200), 9)
        structures.add((min(steps, 20), has_calc, has_final, length_bin))
    return len(structures)

def compute_vocab_diversity(solutions):
    ratios = []
    for s in solutions:
        ans = s.get('answer', '')
        if not ans: continue
        tokens = ans.lower().split()
        if tokens:
            ratios.append(len(set(tokens)) / len(tokens))
    return np.mean(ratios) if ratios else 0

# ============ Selection Functions ============
def select_greedy(solutions, n):
    return sorted([s for s in solutions if s['quality'] > 0.1], key=lambda x: x['quality'], reverse=True)[:n]

def select_qd(solutions, n, archive_cells=None):
    if archive_cells is None: archive_cells = set()
    cell_to_items = defaultdict(list)
    for sol in solutions:
        if sol['cell'] and sol['quality'] > 0.1:
            cell_to_items[sol['cell']].append(sol)
    if not cell_to_items: return select_greedy(solutions, n)
    selected = []
    for cell in [c for c in cell_to_items if c not in archive_cells]:
        selected.append(max(cell_to_items[cell], key=lambda x: x['quality']))
    if len(selected) < n:
        for cell in sorted(cell_to_items.keys()):
            best = max(cell_to_items[cell], key=lambda x: x['quality'])
            if best not in selected:
                selected.append(best)
                if len(selected) >= n: break
    if len(selected) < n:
        for item in sorted([s for s in solutions if s not in selected and s['quality'] > 0.1],
                          key=lambda x: x['quality'], reverse=True):
            if len(selected) >= n: break
            selected.append(item)
    return selected[:n]

def select_qd_no_surprisal(solutions, n, archive_cells=None):
    """QD without surprisal: pick best quality per cell, no empty-cell priority."""
    cell_to_items = defaultdict(list)
    for sol in solutions:
        if sol['cell'] and sol['quality'] > 0.1:
            cell_to_items[sol['cell']].append(sol)
    if not cell_to_items: return select_greedy(solutions, n)
    selected = []
    for cell in sorted(cell_to_items.keys()):
        selected.append(max(cell_to_items[cell], key=lambda x: x['quality']))
    if len(selected) < n:
        for item in sorted([s for s in solutions if s not in selected and s['quality'] > 0.1],
                          key=lambda x: x['quality'], reverse=True):
            if len(selected) >= n: break
            selected.append(item)
    return selected[:n]

def select_random(solutions, n, rng=None):
    valid = [s for s in solutions if s['quality'] > 0.1]
    r = rng if rng else random
    return r.sample(valid, min(n, len(valid)))

# ============ Generation & Eval ============
def generate_solutions(model, tokenizer, prompts, gold_answers):
    solutions = []
    SYS = "You are a mathematics expert. Solve the given math problem step by step.\n\nInstructions:\n1. Read the problem carefully.\n2. Break it down into steps.\n3. Show all calculations.\n4. Write your final numerical answer after ####\n\nExample format:\nJohn has 5 apples and buys 3 more.\nStep 1: Total = 5 + 3 = 8\n#### 8"
    for i, prompt in enumerate(prompts):
        gold = gold_answers[i]
        msgs = [{"role":"system","content":SYS},{"role":"user","content":prompt}]
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(txt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=1024, temperature=0.8,
                               do_sample=True, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        pred = extract_flexible(resp)
        solutions.append({
            'question': prompt, 'answer': resp,
            'cell': get_math_cell(resp),
            'quality': quality_score(resp, gold),
            'correct': check_correct(pred, gold)
        })
        if (i+1) % 100 == 0:
            nc = sum(1 for s in solutions if s['correct'])
            nv = sum(1 for s in solutions if s['quality'] > 0.1)
            print(f"    Gen {i+1}/{len(prompts)} ({nv}v, {nc}c)", flush=True)
    return solutions

def evaluate_gsm8k(model, tokenizer, test_data, n=None, seed=42):
    if n:
        rng = random.Random(seed)
        test_data = rng.sample(test_data, min(n, len(test_data)))
    correct = total = 0
    for i, ex in enumerate(test_data):
        msgs = [{"role":"system","content":"Solve the math problem step by step. Put your final answer after ####."},
                {"role":"user","content":ex['question']}]
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(txt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=1024, temperature=0.0,
                               do_sample=False, pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        pred = extract_flexible(resp)
        gold = extract_answer(ex['answer'])
        if check_correct(pred, gold): correct += 1
        total += 1
        if (i+1) % 100 == 0:
            print(f"    Eval {i+1}/{len(test_data)}, acc={correct/total:.4f}", flush=True)
    return correct, total

def fmt_sample(item):
    return f"<|im_start|>system\nSolve the math problem step by step.<|im_end|>\n<|im_start|>user\n{item['question'][:512]}<|im_end|>\n<|im_start|>assistant\n{item['answer'][:2048]}<|im_end|>"

# ============ Main ============
print("Loading GSM8K...", flush=True)
gsm8k_train = list(load_dataset("gsm8k", "main", split="train"))
gsm8k_test = list(load_dataset("gsm8k", "main", split="test"))
prompt_pool = [(ex['question'], extract_answer(ex['answer'])) for ex in gsm8k_train]

results_file = RESULTS_DIR / f"{STRATEGY}_s{SEED}_v3.json"
all_results = json.load(open(results_file)) if results_file.exists() else {}
archive_cells = set()

# Restore archive from previous results
for k in sorted(all_results.keys()):
    r = all_results[k]
    if r.get("status") == "completed":
        for c in r.get("archive_cells", []):
            archive_cells.add(tuple(c))

# Track ALL accumulated selected data across rounds
accumulated_data = []  # list of selected solution dicts
# Restore accumulated data from previous results
for k in sorted(all_results.keys()):
    r = all_results[k]
    if r.get("status") == "completed" and "selected_data" in r:
        accumulated_data.extend(r["selected_data"])

for rnd in range(N_ROUNDS):
    rnd_key = f"{STRATEGY}_s{SEED}_r{rnd}"
    t0 = time.time()

    if rnd_key in all_results and all_results[rnd_key].get("status") == "completed":
        print(f"  {rnd_key}: done (acc={all_results[rnd_key].get('accuracy','?')}), skipping", flush=True)
        for c in all_results[rnd_key].get("archive_cells", []):
            archive_cells.add(tuple(c))
        continue

    print(f"\n  ROUND {rnd} ({STRATEGY} seed={SEED}) [BASE-RESET]", flush=True)
    torch.manual_seed(SEED + rnd); random.seed(SEED + rnd); np.random.seed(SEED + rnd)

    # ===== KEY DIFFERENCE: Always load BASE model for generation =====
    print(f"  Loading BASE model for generation...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    gen_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
                                                      device_map=DEVICE, trust_remote_code=True)
    gen_model.eval()

    # Step 1: Generate from BASE model
    rng = random.Random(SEED + rnd)
    sampled = rng.sample(prompt_pool, min(N_GENERATE, len(prompt_pool)))
    prompts = [q for q, a in sampled]
    golds = [a for q, a in sampled]

    solutions = generate_solutions(gen_model, tokenizer, prompts, golds)

    n_valid = sum(1 for s in solutions if s['quality'] > 0.1)
    n_correct = sum(1 for s in solutions if s['correct'])
    n_cells = len(set(s['cell'] for s in solutions if s['cell']))
    gen_entropy = compute_cell_entropy(solutions)
    gen_strategies = compute_unique_strategies(solutions)
    gen_vocab_div = compute_vocab_diversity(solutions)
    avg_quality = np.mean([s['quality'] for s in solutions if s['quality'] > 0.1]) if n_valid > 0 else 0

    print(f"  Gen (base): {n_valid}v/{N_GENERATE}, {n_correct}c, {n_cells} cells, H={gen_entropy:.2f}", flush=True)

    # Step 2: Select from THIS round's generated data
    if STRATEGY == "greedy":
        selected = select_greedy(solutions, N_SELECT)
    elif STRATEGY == "qd":
        selected = select_qd(solutions, N_SELECT, archive_cells)
    elif STRATEGY == "qd_no_surprisal":
        selected = select_qd_no_surprisal(solutions, N_SELECT, archive_cells)
    else:  # random
        selected = select_random(solutions, N_SELECT, rng)

    sel_cells = set(s['cell'] for s in selected if s['cell'])
    sel_correct = sum(1 for s in selected if s['correct'])
    sel_entropy = compute_cell_entropy(selected)
    sel_avg_quality = np.mean([s['quality'] for s in selected]) if selected else 0
    archive_cells.update(sel_cells)

    # Add this round's selected data to accumulated pool
    accumulated_data.extend(selected)

    print(f"  Sel: {len(selected)}, {len(sel_cells)} cells, {sel_correct}c, H={sel_entropy:.2f}", flush=True)
    print(f"  Accumulated training data: {len(accumulated_data)} samples from {rnd+1} rounds", flush=True)

    # Free generation model
    del gen_model; torch.cuda.empty_cache()

    # Step 3: Train from BASE model on ALL accumulated data
    print(f"  Loading BASE model for training on {len(accumulated_data)} accumulated samples...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    train_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
                                                        device_map=DEVICE, trust_remote_code=True)

    # Format all accumulated data
    texts = [fmt_sample(s) for s in accumulated_data if s['quality'] > 0.1]
    if len(texts) < 10:
        print(f"  Too few ({len(texts)}), skipping", flush=True)
        del train_model; torch.cuda.empty_cache(); continue

    train_model = get_peft_model(train_model, LoraConfig(r=16, lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0.05, task_type="CAUSAL_LM"))

    ds = Dataset.from_dict({"text": texts})
    trainer = SFTTrainer(model=train_model, args=SFTConfig(
        output_dir=str(RESULTS_DIR / f"ckpt_{rnd_key}"),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=50,
        save_strategy="no",
        bf16=True,
        report_to="none",
        max_length=1024,
        dataset_text_field="text",
        packing=False),
        train_dataset=ds,
        processing_class=tokenizer)
    trainer.train()

    # Merge and save
    train_model.eval()
    merged = train_model.merge_and_unload()
    merged_path = str(RESULTS_DIR / f"merged_{rnd_key}")
    merged.save_pretrained(merged_path); tokenizer.save_pretrained(merged_path)
    del train_model, merged; torch.cuda.empty_cache()

    # Step 4: Evaluate
    print(f"  Evaluating GSM8K ({N_EVAL})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    eval_model = AutoModelForCausalLM.from_pretrained(merged_path, torch_dtype=torch.bfloat16,
                                                       device_map=DEVICE, trust_remote_code=True)
    eval_model.eval()

    correct, total = evaluate_gsm8k(eval_model, tokenizer, gsm8k_test, N_EVAL, seed=SEED)
    accuracy = round(correct/total, 4) if total > 0 else 0
    del eval_model; torch.cuda.empty_cache()

    elapsed = round(time.time() - t0, 1)

    # Save serializable selected data (for accumulation across restarts)
    serializable_selected = [
        {'question': s['question'][:512], 'answer': s['answer'][:2048],
         'cell': list(s['cell']) if s['cell'] else None,
         'quality': s['quality'], 'correct': s['correct']}
        for s in selected
    ]

    all_results[rnd_key] = {
        "round": rnd, "seed": SEED, "strategy": STRATEGY,
        "experiment": "base_reset_v3",
        "n_accumulated_samples": len(accumulated_data),
        "n_generated": n_valid, "n_correct_gen": n_correct,
        "n_cells_generated": n_cells, "gen_entropy": round(gen_entropy, 4),
        "gen_strategies": gen_strategies, "gen_vocab_diversity": round(gen_vocab_div, 4),
        "avg_quality_gen": round(avg_quality, 4),
        "n_selected": len(selected), "n_cells_selected": len(sel_cells),
        "n_correct_selected": sel_correct, "sel_entropy": round(sel_entropy, 4),
        "sel_avg_quality": round(sel_avg_quality, 4),
        "archive_cells": [list(c) for c in archive_cells],
        "archive_size": len(archive_cells),
        "accuracy": accuracy, "correct": correct, "total": total,
        "elapsed_sec": elapsed, "status": "completed",
        "selected_data": serializable_selected,
    }
    with open(results_file, "w") as f: json.dump(all_results, f, indent=2)
    print(f"  R{rnd}: acc={accuracy} ({correct}/{total}), accumulated={len(accumulated_data)}, {elapsed}s", flush=True)

# ============ Summary ============
print(f"\n=== {STRATEGY} seed={SEED} (BASE-RESET) DONE ===", flush=True)
for k,v in sorted(all_results.items()):
    print(f"  {k}: acc={v['accuracy']}, cells={v.get('n_cells_generated','?')}, "
          f"H={v.get('gen_entropy','?')}, accumulated={v.get('n_accumulated_samples','?')}", flush=True)
