"""
Self-Synthesis v5: Noise-Controlled Multi-Domain Experiment
============================================================

Addresses ALL critical issues identified in SAC analysis:

1. NOISE FLOOR (3.2%): Shared generation pool + 3 training seeds
2. SELECTION OVERLAP (99%): N_GENERATE=3000 → more room for selection to differ
3. SINGLE METRIC (GSM8K only): Multi-domain evaluation (GSM8K+MATH+SVAMP+ASDiv)
4. LORA DRIFT: Base-reset every round (train from BASE on accumulated data)
5. STATISTICAL RIGOR: 3 training seeds, report mean±std, Cohen's d

Design:
  Phase 1: Generate 3000 samples from base model → shared pool
  Phase 2: Run 4 selection strategies on shared pool
  Phase 3: Train with 3 LoRA seeds per selection → 12 models per round
  Phase 4: Multi-domain evaluation on all models

GPU Allocation:
  Generation: 1 GPU (~3h for 3000 samples)
  Training: 4 GPUs × 1 strategy × 3 seeds = 12 runs (~30min total)
  Evaluation: Can overlap with next round generation

Strategies: greedy, qd, qd_no_surprisal, random
  + qd_enhanced (S3+S2+S9 TRIZ innovations)
  + qd_s3_s9 (S3+S9 ablation, no anti-archive)

Usage:
  # Single round (all phases):
  CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python -u self_synthesis_v5_improved.py \
      --round 0 --strategy all --seed 42

  # Generate only:
  CUDA_VISIBLE_DEVICES=1 python -u self_synthesis_v5_improved.py --phase generate --round 0

  # Train specific strategy:
  CUDA_VISIBLE_DEVICES=1 python -u self_synthesis_v5_improved.py --phase train \
      --strategy qd --round 0 --train-seed 42
"""

import os, sys, json, random, re, torch, numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import time, math, argparse, hashlib

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-7B-Instruct"
DEVICE = "cuda:0"
GRID_RES = 10
N_ROUNDS = 5
N_GENERATE = 3000
N_SELECT = 500
N_EVAL = 500
N_TRAIN_SEEDS = 3
TRAIN_SEEDS = [42, 123, 456]

STRATEGIES = ['greedy', 'qd', 'qd_no_surprisal', 'random', 'qd_enhanced', 'qd_s3_s9']

RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/self_synthesis_v5_improved")
SHARED_POOL_DIR = RESULTS_DIR / "shared_pool"
SELECTIONS_DIR = RESULTS_DIR / "selections"
TRAIN_DIR = RESULTS_DIR / "trained"
EVAL_DIR = RESULTS_DIR / "eval"

# Multi-domain evaluation benchmarks
EVAL_DATASETS = {
    'gsm8k': {'path': 'gsm8k', 'subset': 'main', 'split': 'test', 'n': 500},
    'math': {'path': 'lighteval/MATH', 'subset': 'all', 'split': 'test', 'n': 500},
    'svamp': {'path': 'ChilleD/SVAMP', 'subset': None, 'split': 'test', 'n': 300},
    'asdiv': {'path': 'EleutherAI/asdiv', 'subset': None, 'split': 'test', 'n': 300},
}

# ============ Descriptor/Quality Functions ============
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

# ============ Selection Functions ============
def select_greedy(solutions, n):
    return sorted([s for s in solutions if s['quality'] > 0.1],
                  key=lambda x: x['quality'], reverse=True)[:n]

def select_qd(solutions, n, archive_cells=None):
    if archive_cells is None: archive_cells = set()
    cell_to_items = defaultdict(list)
    for sol in solutions:
        if sol['cell'] and sol['quality'] > 0.1:
            cell_to_items[sol['cell']].append(sol)
    if not cell_to_items: return select_greedy(solutions, n)
    selected = []
    # Priority 1: Fill empty cells (surprisal)
    for cell in [c for c in cell_to_items if c not in archive_cells]:
        selected.append(max(cell_to_items[cell], key=lambda x: x['quality']))
    # Priority 2: Fill remaining with best quality per cell
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

def select_qd_enhanced(solutions, n, archive_cells=None, golden_ids=None, anti_archive=None):
    """QD with S3 Golden Ratio + S2 Anti-Archive + score extrapolation"""
    if archive_cells is None: archive_cells = set()
    if golden_ids is None: golden_ids = set()
    if anti_archive is None: anti_archive = []

    cell_to_items = defaultdict(list)
    for sol in solutions:
        if sol['cell'] and sol['quality'] > 0.1:
            cell_to_items[sol['cell']].append(sol)
    if not cell_to_items: return select_greedy(solutions, n)

    # Score extrapolation using anti-archive
    if anti_archive:
        anti_scores = [s['quality'] for s in anti_archive if s.get('quality', 0) > 0.1]
        anti_avg = np.mean(anti_scores) if anti_scores else 0
    else:
        anti_avg = 0

    selected = []
    # Priority 1: Golden seeds (S3) - always included
    golden_items = [s for s in solutions if s.get('id', '') in golden_ids]
    selected.extend(golden_items[:int(n * 0.618)])

    # Priority 2: Surprisal (empty cells)
    remaining = n - len(selected)
    for cell in [c for c in cell_to_items if c not in archive_cells]:
        if remaining <= 0: break
        best = max(cell_to_items[cell], key=lambda x: x['quality'])
        if best not in selected:
            selected.append(best)
            remaining -= 1

    # Priority 3: Best per cell with score extrapolation (S2)
    if remaining > 0:
        scored = []
        for cell in sorted(cell_to_items.keys()):
            best = max(cell_to_items[cell], key=lambda x: x['quality'])
            if best not in selected:
                # Score extrapolation
                extrapolated = best['quality'] + 0.3 * (best['quality'] - anti_avg)
                scored.append((extrapolated, best))
        scored.sort(key=lambda x: x[0], reverse=True)
        for _, item in scored:
            if remaining <= 0: break
            selected.append(item)
            remaining -= 1

    if len(selected) < n:
        for item in sorted([s for s in solutions if s not in selected and s['quality'] > 0.1],
                          key=lambda x: x['quality'], reverse=True):
            if len(selected) >= n: break
            selected.append(item)

    return selected[:n]

def select_qd_s3_s9(solutions, n, archive_cells=None, golden_ids=None, stage='explore'):
    """QD with S3 Golden Ratio + S9 Curriculum (no S2 anti-archive)"""
    if archive_cells is None: archive_cells = set()
    if golden_ids is None: golden_ids = set()

    cell_to_items = defaultdict(list)
    for sol in solutions:
        if sol['cell'] and sol['quality'] > 0.1:
            cell_to_items[sol['cell']].append(sol)
    if not cell_to_items: return select_greedy(solutions, n)

    selected = []
    # S3: Golden seeds always included
    golden_items = [s for s in solutions if s.get('id', '') in golden_ids]
    selected.extend(golden_items[:int(n * 0.618)])

    # Fill remaining with per-cell elitism
    remaining = n - len(selected)
    for cell in sorted(cell_to_items.keys()):
        if remaining <= 0: break
        best = max(cell_to_items[cell], key=lambda x: x['quality'])
        if best not in selected:
            selected.append(best)
            remaining -= 1

    if len(selected) < n:
        for item in sorted([s for s in solutions if s not in selected and s['quality'] > 0.1],
                          key=lambda x: x['quality'], reverse=True):
            if len(selected) >= n: break
            selected.append(item)

    return selected[:n]


# ============ Phase 1: Generate Shared Pool ============
def phase_generate(round_num, seed=42, temperature=0.8):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pool_file = SHARED_POOL_DIR / f"pool_r{round_num}_s{seed}.json"
    if pool_file.exists():
        print(f"Pool already exists: {pool_file}", flush=True)
        return json.load(open(pool_file))

    print(f"\n{'='*60}", flush=True)
    print(f"Phase 1: Generating {N_GENERATE} solutions for Round {round_num}", flush=True)
    print(f"{'='*60}", flush=True)
    SHARED_POOL_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
                                                  device_map=DEVICE, trust_remote_code=True)
    model.eval()

    gsm8k_train = list(load_dataset("gsm8k", "main", split="train"))
    prompt_pool = [(ex['question'], extract_answer(ex['answer'])) for ex in gsm8k_train]

    rng = random.Random(seed + round_num)
    sampled = rng.sample(prompt_pool, min(N_GENERATE, len(prompt_pool)))

    SYS = "You are a mathematics expert. Solve the given math problem step by step.\n\nInstructions:\n1. Read the problem carefully.\n2. Break it down into steps.\n3. Show all calculations.\n4. Write your final numerical answer after ####\n\nExample format:\nJohn has 5 apples and buys 3 more.\nStep 1: Total = 5 + 3 = 8\n#### 8"

    solutions = []
    for i, (prompt, gold) in enumerate(sampled):
        msgs = [{"role":"system","content":SYS},{"role":"user","content":prompt}]
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(txt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=1024, temperature=temperature,
                               do_sample=True, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        pred = extract_flexible(resp)
        sol_id = hashlib.md5((prompt[:100]).encode()).hexdigest()[:12]
        solutions.append({
            'id': sol_id,
            'question': prompt, 'answer': resp, 'gold': gold,
            'cell': get_math_cell(resp),
            'quality': quality_score(resp, gold),
            'correct': check_correct(pred, gold)
        })
        if (i+1) % 200 == 0:
            nc = sum(1 for s in solutions if s['correct'])
            nv = sum(1 for s in solutions if s['quality'] > 0.1)
            ncells = len(set(s['cell'] for s in solutions if s['cell']))
            print(f"  Gen {i+1}/{N_GENERATE} ({nv}v, {nc}c, {ncells} cells)", flush=True)

    with open(pool_file, 'w') as f:
        json.dump(solutions, f, indent=2)
    nc = sum(1 for s in solutions if s['correct'])
    ncells = len(set(s['cell'] for s in solutions if s['cell']))
    print(f"Saved {len(solutions)} solutions: {nc} correct, {ncells} cells", flush=True)
    del model; torch.cuda.empty_cache()
    return solutions


# ============ Phase 2: Selection on Shared Pool ============
def phase_select(round_num, seed=42, golden_ids=None, anti_archive=None, archive_cells=None):
    SELECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    pool_file = SHARED_POOL_DIR / f"pool_r{round_num}_s{seed}.json"
    solutions = json.load(open(pool_file))

    # Convert cell tuples
    for s in solutions:
        if s['cell'] and isinstance(s['cell'], list):
            s['cell'] = tuple(s['cell'])

    n_valid = sum(1 for s in solutions if s['quality'] > 0.1)
    n_correct = sum(1 for s in solutions if s['correct'])
    n_cells = len(set(s['cell'] for s in solutions if s['cell']))
    print(f"\nPool R{round_num}: {len(solutions)} total, {n_valid} valid, {n_correct} correct, {n_cells} cells")

    if golden_ids is None: golden_ids = set()
    if anti_archive is None: anti_archive = []
    if archive_cells is None: archive_cells = set()

    for strategy in STRATEGIES:
        sel_file = SELECTIONS_DIR / f"sel_{strategy}_r{round_num}_s{seed}.json"
        if sel_file.exists():
            print(f"  {strategy}: exists, skipping")
            continue

        rng = random.Random(seed + round_num + hash(strategy) % 1000)

        if strategy == 'greedy':
            selected = select_greedy(solutions, N_SELECT)
        elif strategy == 'qd':
            selected = select_qd(solutions, N_SELECT, archive_cells)
        elif strategy == 'qd_no_surprisal':
            selected = select_qd_no_surprisal(solutions, N_SELECT, archive_cells)
        elif strategy == 'random':
            selected = select_random(solutions, N_SELECT, rng)
        elif strategy == 'qd_enhanced':
            selected = select_qd_enhanced(solutions, N_SELECT, archive_cells, golden_ids, anti_archive)
        elif strategy == 'qd_s3_s9':
            selected = select_qd_s3_s9(solutions, N_SELECT, archive_cells, golden_ids, 'explore')
        else:
            continue

        sel_data = {
            'strategy': strategy, 'round': round_num, 'seed': seed,
            'n_selected': len(selected),
            'n_cells': len(set(s['cell'] for s in selected if s['cell'])),
            'n_correct': sum(1 for s in selected if s['correct']),
            'avg_quality': float(np.mean([s['quality'] for s in selected])),
            'entropy': float(-sum(
                (c/len(selected)) * math.log2(c/len(selected))
                for c in Counter(s['cell'] for s in selected if s['cell']).values()
                if c > 0
            )),
            'selection_ids': [s.get('id', hashlib.md5(s['question'][:100].encode()).hexdigest()[:12])
                             for s in selected],
            'samples': [{'id': s.get('id',''), 'question': s['question'][:512],
                         'answer': s['answer'][:2048],
                         'cell': list(s['cell']) if s['cell'] else None,
                         'quality': s['quality'], 'correct': s['correct']}
                        for s in selected]
        }
        with open(sel_file, 'w') as f:
            json.dump(sel_data, f, indent=2)
        print(f"  {strategy}: {len(selected)} sel, {sel_data['n_cells']} cells, "
              f"{sel_data['n_correct']} correct, H={sel_data['entropy']:.3f}")

    # Compute pairwise overlap for this round
    print(f"\n  Selection Overlap Matrix (R{round_num}):")
    sels = {}
    for strategy in STRATEGIES:
        sel_file = SELECTIONS_DIR / f"sel_{strategy}_r{round_num}_s{seed}.json"
        if sel_file.exists():
            sd = json.load(open(sel_file))
            sels[strategy] = set(sd['selection_ids'])

    for i, s1 in enumerate(STRATEGIES):
        for s2 in STRATEGIES[i+1:]:
            if s1 in sels and s2 in sels:
                overlap = len(sels[s1] & sels[s2])
                total = min(len(sels[s1]), len(sels[s2]))
                print(f"    {s1} vs {s2}: {overlap}/{total} ({overlap/total*100:.0f}%)")


# ============ Phase 3: Train + Evaluate ============
def phase_train(strategy, round_num, seed=42, train_seed=42):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset, load_dataset

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    result_file = TRAIN_DIR / f"{strategy}_r{round_num}_s{seed}_ts{train_seed}.json"
    if result_file.exists():
        print(f"Results exist: {result_file}", flush=True)
        return json.load(open(result_file))

    sel_file = SELECTIONS_DIR / f"sel_{strategy}_r{round_num}_s{seed}.json"
    if not sel_file.exists():
        print(f"Selection not found: {sel_file}", flush=True)
        return None

    sel_data = json.load(open(sel_file))
    samples = sel_data['samples']

    print(f"\n=== Training {strategy} R{round_num} seed={train_seed} ({len(samples)} samples) ===", flush=True)
    t0 = time.time()

    torch.manual_seed(train_seed)
    torch.cuda.manual_seed_all(train_seed)
    random.seed(train_seed)
    np.random.seed(train_seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
                                                   device_map=DEVICE, trust_remote_code=True)

    texts = [f"<|im_start|>system\nSolve the math problem step by step. Put final answer after ####.<|im_end|>\n"
             f"<|im_start|>user\n{s['question'][:512]}<|im_end|>\n"
             f"<|im_start|>assistant\n{s['answer'][:2048]}<|im_end|>"
             for s in samples if s['quality'] > 0.1]

    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0.05, task_type="CAUSAL_LM"))

    ds = Dataset.from_dict({"text": texts})
    trainer = SFTTrainer(model=model, args=SFTConfig(
        output_dir=str(TRAIN_DIR / f"ckpt_{strategy}_r{round_num}_ts{train_seed}"),
        num_train_epochs=3, per_device_train_batch_size=2,
        gradient_accumulation_steps=8, learning_rate=2e-4,
        logging_steps=50, save_strategy="no", bf16=True,
        report_to="none", max_length=1024,
        dataset_text_field="text", packing=False,
        seed=train_seed,
    ), train_dataset=ds, processing_class=tokenizer)
    trainer.train()

    model.eval()
    merged = model.merge_and_unload()
    merged_path = str(TRAIN_DIR / f"merged_{strategy}_r{round_num}_ts{train_seed}")
    merged.save_pretrained(merged_path); tokenizer.save_pretrained(merged_path)
    del model, merged; torch.cuda.empty_cache()

    # Multi-domain evaluation
    print(f"  Multi-domain evaluation...", flush=True)
    eval_results = {}
    tokenizer = AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    eval_model = AutoModelForCausalLM.from_pretrained(merged_path, torch_dtype=torch.bfloat16,
                                                       device_map=DEVICE, trust_remote_code=True)
    eval_model.eval()

    for bname, bcfg in EVAL_DATASETS.items():
        try:
            if bcfg['subset']:
                ds_eval = list(load_dataset(bcfg['path'], bcfg['subset'], split=bcfg['split']))
            else:
                ds_eval = list(load_dataset(bcfg['path'], split=bcfg['split']))

            correct = total = 0
            for i, ex in enumerate(ds_eval[:bcfg['n']]):
                q = ex.get('question', ex.get('problem', ''))
                a = ex.get('answer', ex.get('solution', ''))
                gold = extract_answer(a) if '####' in a else extract_flexible(a)

                msgs = [{"role":"system","content":"Solve the math problem step by step. Put final answer after ####."},
                        {"role":"user","content":q}]
                txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inp = tokenizer(txt, return_tensors="pt").to(eval_model.device)
                with torch.no_grad():
                    out = eval_model.generate(**inp, max_new_tokens=1024, temperature=0.0,
                                           do_sample=False, pad_token_id=tokenizer.eos_token_id)
                resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
                pred = extract_flexible(resp)
                if check_correct(pred, gold): correct += 1
                total += 1

            acc = round(correct/total, 4) if total > 0 else 0
            eval_results[bname] = {'accuracy': acc, 'correct': correct, 'total': total}
            print(f"    {bname}: {acc:.4f} ({correct}/{total})", flush=True)
        except Exception as e:
            print(f"    {bname}: Error - {e}", flush=True)
            eval_results[bname] = {'accuracy': 0, 'error': str(e)}

    # Also evaluate on GSM8K
    gsm8k_test = list(load_dataset("gsm8k", "main", split="test"))
    correct = total = 0
    for ex in gsm8k_test[:N_EVAL]:
        msgs = [{"role":"system","content":"Solve the math problem step by step. Put final answer after ####."},
                {"role":"user","content":ex['question']}]
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(txt, return_tensors="pt").to(eval_model.device)
        with torch.no_grad():
            out = eval_model.generate(**inp, max_new_tokens=1024, temperature=0.0,
                                   do_sample=False, pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        pred = extract_flexible(resp)
        gold = extract_answer(ex['answer'])
        if check_correct(pred, gold): correct += 1
        total += 1

    eval_results['gsm8k'] = {'accuracy': round(correct/total, 4), 'correct': correct, 'total': total}
    print(f"    gsm8k: {eval_results['gsm8k']['accuracy']:.4f} ({correct}/{total})", flush=True)

    del eval_model; torch.cuda.empty_cache()
    elapsed = round(time.time() - t0, 1)

    result = {
        'strategy': strategy, 'round': round_num, 'seed': seed,
        'train_seed': train_seed, 'elapsed_sec': elapsed,
        'n_train_samples': len(texts),
        'eval': eval_results,
        'selection_info': {
            'n_cells': sel_data['n_cells'],
            'n_correct': sel_data['n_correct'],
            'avg_quality': sel_data['avg_quality'],
            'entropy': sel_data['entropy'],
        }
    }
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Done: {elapsed}s", flush=True)
    return result


# ============ Aggregate Results ============
def aggregate_results(round_num, seed=42):
    """Aggregate multi-seed results for all strategies."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    agg_file = EVAL_DIR / f"aggregate_r{round_num}_s{seed}.json"

    agg = {}
    for strategy in STRATEGIES:
        results = []
        for ts in TRAIN_SEEDS:
            rf = TRAIN_DIR / f"{strategy}_r{round_num}_s{seed}_ts{ts}.json"
            if rf.exists():
                results.append(json.load(open(rf)))

        if not results:
            continue

        # Aggregate per-benchmark
        benchmarks = defaultdict(list)
        for r in results:
            for bname, bdata in r.get('eval', {}).items():
                if 'accuracy' in bdata:
                    benchmarks[bname].append(bdata['accuracy'])

        agg[strategy] = {
            'n_seeds': len(results),
            'benchmarks': {},
            'selection_info': results[0].get('selection_info', {}) if results else {},
        }
        for bname, accs in benchmarks.items():
            agg[strategy]['benchmarks'][bname] = {
                'mean': round(float(np.mean(accs)), 4),
                'std': round(float(np.std(accs)), 4),
                'min': round(float(min(accs)), 4),
                'max': round(float(max(accs)), 4),
                'values': accs,
            }

    with open(agg_file, 'w') as f:
        json.dump(agg, f, indent=2)

    # Print summary
    print(f"\n{'='*80}", flush=True)
    print(f"Round {round_num} Summary (n={TRAIN_SEEDS} seeds)", flush=True)
    print(f"{'='*80}", flush=True)
    header = f"{'Strategy':20s} | {'GSM8K':>12s} | {'MATH':>12s} | {'SVAMP':>12s} | {'ASDiv':>12s} | {'Cells':>5s} | {'Sel_H':>6s}"
    print(header)
    print("-" * len(header))
    for strategy in STRATEGIES:
        if strategy not in agg: continue
        s = agg[strategy]
        bm = s['benchmarks']
        cells = s.get('selection_info', {}).get('n_cells', '?')
        ent = s.get('selection_info', {}).get('entropy', '?')
        row = f"{strategy:20s} |"
        for bn in ['gsm8k', 'math', 'svamp', 'asdiv']:
            if bn in bm:
                row += f" {bm[bn]['mean']*100:.1f}±{bm[bn]['std']*100:.1f}% |"
            else:
                row += f" {'N/A':>12s} |"
        row += f" {cells:>5} | {ent:.3f}" if isinstance(ent, float) else f" {cells:>5} | {ent}"
        print(row)

    # Effect sizes (Cohen's d) vs Greedy
    if 'greedy' in agg:
        print(f"\nEffect sizes (Cohen's d) vs Greedy on GSM8K:")
        greedy_accs = agg['greedy']['benchmarks'].get('gsm8k', {}).get('values', [])
        for strategy in STRATEGIES:
            if strategy == 'greedy' or strategy not in agg: continue
            strat_accs = agg[strategy]['benchmarks'].get('gsm8k', {}).get('values', [])
            if len(greedy_accs) >= 2 and len(strat_accs) >= 2:
                d = (np.mean(strat_accs) - np.mean(greedy_accs)) / np.sqrt(
                    (np.std(greedy_accs)**2 + np.std(strat_accs)**2) / 2
                )
                print(f"  {strategy}: d={d:.2f} ({'large' if abs(d)>0.8 else 'medium' if abs(d)>0.5 else 'small'})")

    return agg


# ============ Main ============
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True,
                       choices=['generate', 'select', 'train', 'aggregate', 'all', 'round'])
    parser.add_argument('--strategy', default='all')
    parser.add_argument('--round', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-seed', type=int, default=42)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=0.8)
    args = parser.parse_args()

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    DEVICE = "cuda:0"

    if args.phase == 'generate':
        phase_generate(args.round, args.seed, args.temperature)

    elif args.phase == 'select':
        phase_select(args.round, args.seed)

    elif args.phase == 'train':
        if args.strategy == 'all':
            for ts in TRAIN_SEEDS:
                for strat in STRATEGIES:
                    phase_train(strat, args.round, args.seed, ts)
        else:
            phase_train(args.strategy, args.round, args.seed, args.train_seed)

    elif args.phase == 'aggregate':
        aggregate_results(args.round, args.seed)

    elif args.phase == 'round':
        # Full round: generate → select → train all → aggregate
        phase_generate(args.round, args.seed, args.temperature)
        phase_select(args.round, args.seed)
        for ts in TRAIN_SEEDS:
            for strat in STRATEGIES:
                phase_train(strat, args.round, args.seed, ts)
        aggregate_results(args.round, args.seed)

    elif args.phase == 'all':
        # All rounds
        for r in range(N_ROUNDS):
            phase_generate(r, args.seed, args.temperature)
            phase_select(r, args.seed)
            for ts in TRAIN_SEEDS:
                for strat in STRATEGIES:
                    phase_train(strat, args.round, args.seed, ts)
            aggregate_results(r, args.seed)
