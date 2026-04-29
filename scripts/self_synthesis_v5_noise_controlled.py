"""
Self-Synthesis v5: Noise-Controlled Experiment
===============================================

KEY FIX for v3 noise floor issue (3.2% training noise drowns out selection effects):

1. SHARED GENERATION POOL: Generate ONCE, all strategies select from SAME pool
2. MULTI-TRAINING SEEDS: Each selection result trained 5 times with different LoRA seeds
3. DETERMINISTIC TRAINING: torch.use_deterministic_mode, CUBLAS_WORKSPACE_CONFIG
4. LARGER SCALE: N_GENERATE=3000, N_SELECT=500, 5 rounds

This eliminates:
- Generation noise (shared pool)
- Training noise (multiple seeds + deterministic mode)
- Confounds from different generation outputs

Design:
  Phase 1 (CPU/GPU 1): Generate 3000 solutions from base model → save to shared pool
  Phase 2 (CPU):        Run all selection strategies on shared pool → save selected sets
  Phase 3 (GPU 1-7):   Train + evaluate each selection set with 5 LoRA seeds → mean±std

GPU Allocation (after v3/v4 complete):
  Phase 1: 1 GPU for generation (~2h for 3000 samples)
  Phase 3: 7 GPUs parallel, each running 1 strategy × 5 seeds = ~5 runs per GPU

Usage:
  # Phase 1: Generate shared pool (run first, takes ~2h)
  CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python -u self_synthesis_v5_noise_controlled.py --phase generate

  # Phase 2: Run selections (CPU, fast)
  python self_synthesis_v5_noise_controlled.py --phase select

  # Phase 3: Train + evaluate (7 GPUs parallel)
  CUDA_VISIBLE_DEVICES=1 python -u self_synthesis_v5_noise_controlled.py --phase train --strategy greedy --round 0 &
  CUDA_VISIBLE_DEVICES=2 python -u self_synthesis_v5_noise_controlled.py --phase train --strategy qd --round 0 &
  ...
"""
import os, sys, json, random, re, torch, numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import time, math, argparse
import hashlib

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# Deterministic CUDA
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-7B-Instruct"
DEVICE = "cuda:0"
GRID_RES = 10
N_ROUNDS = 5
N_GENERATE = 3000
N_SELECT = 500
N_EVAL = 500
N_TRAIN_SEEDS = 5  # Multiple LoRA training seeds for each selection
TRAIN_SEEDS = [42, 123, 456, 789, 2024]
STRATEGIES = ['greedy', 'qd', 'random', 'qd_no_surprisal']

RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/self_synthesis_v5_noise_controlled")
SHARED_POOL_DIR = RESULTS_DIR / "shared_pool"
SELECTIONS_DIR = RESULTS_DIR / "selections"
TRAIN_DIR = RESULTS_DIR / "trained"

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
    return sorted([s for s in solutions if s['quality'] > 0.1], key=lambda x: x['quality'], reverse=True)[:n]

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
    # Priority 2: Fill remaining with best quality from unselected
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

# ============ Phase 1: Generate Shared Pool ============
def phase_generate(round_num, seed=42):
    """Generate N_GENERATE solutions from base model. Save to shared pool."""
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pool_file = SHARED_POOL_DIR / f"pool_r{round_num}_s{seed}.json"
    if pool_file.exists():
        print(f"Pool already exists: {pool_file}", flush=True)
        return

    print(f"=== Phase 1: Generating {N_GENERATE} solutions for Round {round_num} ===", flush=True)
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
            out = model.generate(**inp, max_new_tokens=1024, temperature=0.8,
                               do_sample=True, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        pred = extract_flexible(resp)
        solutions.append({
            'question': prompt, 'answer': resp, 'gold': gold,
            'cell': get_math_cell(resp),
            'quality': quality_score(resp, gold),
            'correct': check_correct(pred, gold)
        })
        if (i+1) % 200 == 0:
            nc = sum(1 for s in solutions if s['correct'])
            nv = sum(1 for s in solutions if s['quality'] > 0.1)
            print(f"  Gen {i+1}/{N_GENERATE} ({nv}v, {nc}c)", flush=True)

    # Save
    with open(pool_file, 'w') as f:
        json.dump(solutions, f, indent=2)
    print(f"Saved {len(solutions)} solutions to {pool_file}", flush=True)
    del model; torch.cuda.empty_cache()
    return solutions

# ============ Phase 2: Selection on Shared Pool ============
def phase_select(round_num, seed=42):
    """Run all selection strategies on shared pool. Save selection sets."""
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
    print(f"Pool R{round_num}: {len(solutions)} total, {n_valid} valid, {n_correct} correct, {n_cells} cells")

    archive_cells = set()  # Empty for R0

    for strategy in STRATEGIES:
        sel_file = SELECTIONS_DIR / f"sel_{strategy}_r{round_num}_s{seed}.json"
        if sel_file.exists():
            print(f"  {strategy}: selection exists, skipping")
            continue

        rng = random.Random(seed + round_num + hash(strategy) % 1000)
        if strategy == 'greedy':
            selected = select_greedy(solutions, N_SELECT)
        elif strategy == 'qd':
            selected = select_qd(solutions, N_SELECT, archive_cells)
        elif strategy == 'qd_no_surprisal':
            selected = select_qd_no_surprisal(solutions, N_SELECT, archive_cells)
        else:
            selected = select_random(solutions, N_SELECT, rng)

        # Save selection (serializable)
        sel_data = {
            'strategy': strategy, 'round': round_num, 'seed': seed,
            'n_selected': len(selected),
            'n_cells': len(set(s['cell'] for s in selected if s['cell'])),
            'n_correct': sum(1 for s in selected if s['correct']),
            'avg_quality': np.mean([s['quality'] for s in selected]),
            'selection_ids': [hashlib.md5((s['question'][:100]).encode()).hexdigest()[:12] for s in selected],
            'samples': [{'question': s['question'][:512], 'answer': s['answer'][:2048],
                         'cell': list(s['cell']) if s['cell'] else None,
                         'quality': s['quality'], 'correct': s['correct']}
                        for s in selected]
        }
        with open(sel_file, 'w') as f:
            json.dump(sel_data, f, indent=2)
        print(f"  {strategy}: {len(selected)} selected, {sel_data['n_cells']} cells, {sel_data['n_correct']} correct")

# ============ Phase 3: Train + Evaluate with Multiple Seeds ============
def phase_train(strategy, round_num, seed=42, gpu_id=0):
    """Train LoRA with multiple training seeds on a selection set."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    results_file = TRAIN_DIR / f"train_{strategy}_r{round_num}_s{seed}_results.json"
    if results_file.exists():
        print(f"Results exist: {results_file}", flush=True)
        return

    sel_file = SELECTIONS_DIR / f"sel_{strategy}_r{round_num}_s{seed}.json"
    sel_data = json.load(open(sel_file))
    samples = sel_data['samples']

    gsm8k_test = list(load_dataset("gsm8k", "main", split="test"))

    all_results = {}
    for train_seed in TRAIN_SEEDS:
        seed_key = f"train_seed_{train_seed}"
        result_file = TRAIN_DIR / f"{strategy}_r{round_num}_s{seed}_ts{train_seed}.json"
        if result_file.exists():
            all_results[seed_key] = json.load(open(result_file))
            print(f"  Seed {train_seed}: exists (acc={all_results[seed_key].get('accuracy','?')})", flush=True)
            continue

        print(f"\n=== Training {strategy} R{round_num} with LoRA seed={train_seed} ===", flush=True)
        t0 = time.time()

        # Set ALL seeds for determinism
        torch.manual_seed(train_seed)
        torch.cuda.manual_seed_all(train_seed)
        random.seed(train_seed)
        np.random.seed(train_seed)
        try:
            torch.use_deterministic_mode(True)
        except:
            pass

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
                                                       device_map=DEVICE, trust_remote_code=True)

        # Format training data
        texts = [f"<|im_start|>system\nSolve the math problem step by step.<|im_end|>\n"
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
            seed=train_seed,  # Explicit seed for SFT
        ), train_dataset=ds, processing_class=tokenizer)
        trainer.train()

        # Merge
        model.eval()
        merged = model.merge_and_unload()
        merged_path = str(TRAIN_DIR / f"merged_{strategy}_r{round_num}_ts{train_seed}")
        merged.save_pretrained(merged_path); tokenizer.save_pretrained(merged_path)
        del model, merged; torch.cuda.empty_cache()

        # Evaluate
        print(f"  Evaluating with seed {train_seed}...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        eval_model = AutoModelForCausalLM.from_pretrained(merged_path, torch_dtype=torch.bfloat16,
                                                           device_map=DEVICE, trust_remote_code=True)
        eval_model.eval()

        correct = total = 0
        # Use deterministic eval
        torch.manual_seed(99999)
        for i, ex in enumerate(gsm8k_test[:N_EVAL]):
            msgs = [{"role":"system","content":"Solve the math problem step by step. Put your final answer after ####."},
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
            if (i+1) % 100 == 0:
                print(f"    Eval {i+1}/{N_EVAL}, acc={correct/total:.4f}", flush=True)

        accuracy = round(correct/total, 4)
        elapsed = round(time.time() - t0, 1)
        del eval_model; torch.cuda.empty_cache()

        result = {'strategy': strategy, 'round': round_num, 'seed': seed,
                  'train_seed': train_seed, 'accuracy': accuracy,
                  'correct': correct, 'total': total, 'elapsed_sec': elapsed}
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

        all_results[seed_key] = result
        print(f"  Seed {train_seed}: acc={accuracy} ({correct}/{total}), {elapsed}s", flush=True)

    # Summary
    accs = [r['accuracy'] for r in all_results.values()]
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    summary = {
        'strategy': strategy, 'round': round_num, 'seed': seed,
        'n_train_seeds': len(accs),
        'mean_accuracy': round(float(mean_acc), 4),
        'std_accuracy': round(float(std_acc), 4),
        'min_accuracy': round(float(min(accs)), 4),
        'max_accuracy': round(float(max(accs)), 4),
        'individual_results': all_results
    }
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== {strategy} R{round_num}: {mean_acc:.4f} ± {std_acc:.4f} (n={len(accs)}) ===", flush=True)

# ============ Main ============
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['generate', 'select', 'train', 'all'])
    parser.add_argument('--strategy', default='all')
    parser.add_argument('--round', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    DEVICE = "cuda:0"

    if args.phase == 'generate':
        phase_generate(args.round, args.seed)
    elif args.phase == 'select':
        phase_select(args.round, args.seed)
    elif args.phase == 'train':
        if args.strategy == 'all':
            for strat in STRATEGIES:
                phase_train(strat, args.round, args.seed, args.gpu_id)
        else:
            phase_train(args.strategy, args.round, args.seed, args.gpu_id)
    elif args.phase == 'all':
        phase_generate(args.round, args.seed)
        phase_select(args.round, args.seed)
        for strat in STRATEGIES:
            phase_train(strat, args.round, args.seed, args.gpu_id)
