"""
Self-Synthesis Collapse Experiment
===================================
Core: Model generates its own training data → selects → trains → repeats

R0: Base model generates → select → train → Model_R0
R1: Model_R0 generates → select → train → Model_R1
...
R7: Model_R6 generates → select → train → Model_R7

After each round: evaluate on GSM8K test set
Hypothesis: Greedy collapses (acc drops), QD prevents (acc maintains/improves)

Usage:
  GPU_ID=2 STRATEGY=greedy python self_synthesis_loop.py &
  GPU_ID=3 STRATEGY=qd python self_synthesis_loop.py &
  GPU_ID=4 STRATEGY=random python self_synthesis_loop.py &
"""
import os, sys, json, random, re, torch, numpy as np
from pathlib import Path
from collections import defaultdict
import time

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

GPU_ID = int(os.environ.get("GPU_ID", "2"))
STRATEGY = os.environ.get("STRATEGY", "greedy")

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
N_ROUNDS = 7
N_GENERATE = 200    # generate per round
N_SELECT = 100      # select for training
N_EVAL = 500        # eval subset (speed)

RESULTS_DIR = Path(f"/mnt/data2/zcz/neurIps-emnlp/neurips/results/self_synthesis/{STRATEGY}")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"=== Self-Synthesis: {STRATEGY.upper()} (GPU {GPU_ID}) ===", flush=True)
print(f"  N_GENERATE={N_GENERATE}, N_SELECT={N_SELECT}, N_ROUNDS={N_ROUNDS}", flush=True)

# ============ Descriptor Functions ============
def get_math_cell(answer):
    if not answer or len(answer) < 20: return None
    steps = answer.count('<<') + 1
    difficulty = min(len(answer) / 500.0, 1.0)
    is_multi = 1 if steps >= 3 else 0
    return (int(difficulty * GRID_RES), int(min(steps / 10.0, 1.0) * GRID_RES), int(is_multi * GRID_RES))

def extract_answer(text):
    m = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    return m.group(1).replace(',', '') if m else None

def quality_score(answer, gold_answer=None):
    """Quality: format + steps + length + correctness."""
    if not answer or len(answer) < 20: return 0
    score = 0
    if '####' in answer: score += 0.2
    score += min(answer.count('<<') / 5.0, 0.2)
    score += min(len(answer) / 500.0, 0.15)
    pred = extract_answer(answer)
    if pred: score += 0.15
    if gold_answer and pred:
        if pred.strip() == gold_answer.strip(): score += 0.3
    return min(score, 1.0)

# ============ Selection Functions ============
def select_greedy(solutions, n):
    sorted_sols = sorted([s for s in solutions if s['quality'] > 0.1],
                         key=lambda x: x['quality'], reverse=True)
    return sorted_sols[:n]

def select_qd(solutions, n, archive_cells=None):
    if archive_cells is None:
        archive_cells = set()

    cell_to_items = defaultdict(list)
    for sol in solutions:
        if sol['cell'] and sol['quality'] > 0.1:
            cell_to_items[sol['cell']].append(sol)

    if not cell_to_items:
        return select_greedy(solutions, n)

    selected = []
    # Priority 1: Empty cells (surprisal/novelty)
    empty_cells = [c for c in cell_to_items if c not in archive_cells]
    for cell in empty_cells:
        best = max(cell_to_items[cell], key=lambda x: x['quality'])
        selected.append(best)

    # Priority 2: Improve existing cells
    if len(selected) < n:
        for cell in sorted(cell_to_items.keys()):
            best = max(cell_to_items[cell], key=lambda x: x['quality'])
            if best not in selected:
                selected.append(best)
                if len(selected) >= n: break

    # Priority 3: Top quality fill
    if len(selected) < n:
        remaining = [s for s in solutions if s not in selected and s['quality'] > 0.1]
        remaining.sort(key=lambda x: x['quality'], reverse=True)
        for item in remaining:
            if len(selected) >= n: break
            selected.append(item)

    return selected[:n]

def select_random(solutions, n):
    valid = [s for s in solutions if s['quality'] > 0.1]
    return random.sample(valid, min(n, len(valid)))

# ============ Generation Function ============
def generate_solutions(model, tokenizer, prompts, gold_answers=None):
    solutions = []
    for i, prompt in enumerate(prompts):
        gold = gold_answers[i] if gold_answers else None
        msgs = [
            {"role": "system", "content": "Solve the math problem step by step. Show your work using <<>> notation and put the final answer after ####."},
            {"role": "user", "content": prompt}
        ]
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(txt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=512, temperature=0.7,
                               do_sample=True, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)

        solutions.append({
            'question': prompt,
            'answer': resp,
            'cell': get_math_cell(resp),
            'quality': quality_score(resp, gold),
            'correct': (extract_answer(resp) == gold) if gold and extract_answer(resp) else False
        })

        if (i+1) % 50 == 0:
            n_valid = sum(1 for s in solutions if s['quality'] > 0.1)
            print(f"    Generated {i+1}/{len(prompts)} ({n_valid} valid)", flush=True)

    return solutions

# ============ Evaluation Function ============
def evaluate_gsm8k(model, tokenizer, test_data, n=None):
    if n:
        random.seed(42)
        test_data = random.sample(test_data, min(n, len(test_data)))

    correct = total = 0
    for i, ex in enumerate(test_data):
        msgs = [{"role":"system","content":"Solve the math problem step by step. Put your final answer after ####."},
                {"role":"user","content":ex['question']}]
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(txt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=512, temperature=0.0,
                               do_sample=False, pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        # Try #### format first
        pred_match = re.search(r'####\s*(-?[\d,]+\.?\d*)', resp)
        if not pred_match:
            # Try last number in response
            pred_match = re.search(r'(-?[\d,]+\.?\d*)\s*$', resp.strip())
        if not pred_match:
            # Try any number preceded by "is" or "=" near end
            pred_match = re.search(r'(?:is|=)\s*(-?[\d,]+\.?\d*)', resp)
        pred = pred_match.group(1).replace(',', '') if pred_match else None
        gold = extract_answer(ex['answer'])
        if pred and gold and pred.strip() == gold.strip():
            correct += 1
        total += 1
        if (i+1) % 100 == 0:
            print(f"    Eval {i+1}/{len(test_data)}, acc={correct/total:.4f}", flush=True)
    return correct, total

def fmt_sample(item):
    return f"<|im_start|>system\nSolve the math problem step by step.<|im_end|>\n<|im_start|>user\n{item['question'][:512]}<|im_end|>\n<|im_start|>assistant\n{item['answer'][:1024]}<|im_end|>"

# ============ Load Data ============
print("Loading GSM8K...", flush=True)
gsm8k_train = list(load_dataset("gsm8k", "main", split="train"))
gsm8k_test = list(load_dataset("gsm8k", "main", split="test"))

# Prepare prompt pool with gold answers for quality scoring
prompt_pool = [(ex['question'], extract_answer(ex['answer'])) for ex in gsm8k_train]

print(f"Pool: {len(prompt_pool)} prompts, Test: {len(gsm8k_test)} problems", flush=True)

# ============ Main Loop ============
results_file = RESULTS_DIR / f"{STRATEGY}_self_synthesis.json"
all_results = json.load(open(results_file)) if results_file.exists() else {}
archive_cells = set()

# Restore archive from previous results
for k in sorted(all_results.keys()):
    r = all_results[k]
    if r.get("status") == "completed":
        for c in r.get("archive_cells", []):
            archive_cells.add(tuple(c))

for rnd in range(N_ROUNDS):
    rnd_key = f"{STRATEGY}_r{rnd}"
    t0 = time.time()

    if rnd_key in all_results and all_results[rnd_key].get("status") == "completed":
        print(f"\n  {rnd_key}: done (acc={all_results[rnd_key].get('accuracy','?')}), skipping", flush=True)
        continue

    print(f"\n{'='*50}", flush=True)
    print(f"  ROUND {rnd} ({STRATEGY.upper()})", flush=True)
    print(f"{'='*50}", flush=True)

    # Load current model
    if rnd == 0:
        model_path = MODEL_PATH
    else:
        prev_key = f"{STRATEGY}_r{rnd-1}"
        model_path = str(RESULTS_DIR / f"merged_{prev_key}")
        if not Path(model_path).exists():
            print(f"  ERROR: {model_path} not found!", flush=True)
            break

    print(f"  Loading model from {model_path}", flush=True)
    torch.manual_seed(42 + rnd); random.seed(42 + rnd); np.random.seed(42 + rnd)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                  device_map=DEVICE, trust_remote_code=True)
    model.eval()

    # Step 1: Generate
    print(f"  Generating {N_GENERATE} solutions...", flush=True)
    random.seed(42 + rnd)
    sampled = random.sample(prompt_pool, min(N_GENERATE, len(prompt_pool)))
    prompts = [q for q, a in sampled]
    golds = [a for q, a in sampled]

    solutions = generate_solutions(model, tokenizer, prompts, golds)

    n_valid = sum(1 for s in solutions if s['quality'] > 0.1)
    n_correct_gen = sum(1 for s in solutions if s['correct'])
    n_cells_gen = len(set(s['cell'] for s in solutions if s['cell']))
    print(f"  Generated: {n_valid}/{N_GENERATE} valid, {n_correct_gen}/{N_GENERATE} correct, {n_cells_gen} cells", flush=True)

    # Step 2: Select
    if STRATEGY == "greedy":
        selected = select_greedy(solutions, N_SELECT)
    elif STRATEGY == "qd":
        selected = select_qd(solutions, N_SELECT, archive_cells)
    elif STRATEGY == "random":
        selected = select_random(solutions, N_SELECT)

    sel_cells = set(s['cell'] for s in selected if s['cell'])
    sel_correct = sum(1 for s in selected if s['correct'])
    archive_cells.update(sel_cells)

    print(f"  Selected: {len(selected)} samples, {len(sel_cells)} cells, {sel_correct} correct", flush=True)
    print(f"  Archive: {len(archive_cells)} total cells", flush=True)

    # Step 3: Train
    texts = [fmt_sample(s) for s in selected if s['quality'] > 0.1]
    if len(texts) < 5:
        print(f"  Too few valid ({len(texts)}), skipping", flush=True)
        del model; torch.cuda.empty_cache()
        continue

    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0.05, task_type="CAUSAL_LM"))

    ds = Dataset.from_dict({"text": texts})
    trainer = SFTTrainer(model=model, args=SFTConfig(
        output_dir=str(RESULTS_DIR / f"ckpt_{rnd_key}"),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=50,
        save_strategy="no",
        bf16=True,
        report_to="none",
        max_length=768,
        dataset_text_field="text",
        packing=False),
        train_dataset=ds,
        processing_class=tokenizer)
    trainer.train()

    # Merge and save
    model.eval()
    merged = model.merge_and_unload()
    merged_path = str(RESULTS_DIR / f"merged_{rnd_key}")
    merged.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    del model, merged; torch.cuda.empty_cache()
    print(f"  Saved merged model to {merged_path}", flush=True)

    # Step 4: Evaluate
    print(f"  Evaluating on GSM8K ({N_EVAL} samples)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(merged_path, torch_dtype=torch.bfloat16,
                                                  device_map=DEVICE, trust_remote_code=True)
    model.eval()

    correct, total = evaluate_gsm8k(model, tokenizer, gsm8k_test, N_EVAL)
    accuracy = round(correct/total, 4) if total > 0 else 0

    del model; torch.cuda.empty_cache()

    elapsed = round(time.time() - t0, 1)
    rnd_results = {
        "round": rnd,
        "model_path": model_path if rnd == 0 else merged_path,
        "n_generated": n_valid,
        "n_correct_gen": n_correct_gen,
        "n_cells_generated": n_cells_gen,
        "n_selected": len(selected),
        "n_cells_selected": len(sel_cells),
        "n_correct_selected": sel_correct,
        "archive_cells": [list(c) for c in archive_cells],
        "archive_size": len(archive_cells),
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "elapsed_sec": elapsed,
        "status": "completed"
    }

    all_results[rnd_key] = rnd_results
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"  R{rnd}: acc={accuracy} ({correct}/{total}), archive={len(archive_cells)} cells, {elapsed}s", flush=True)

# ============ Summary ============
print(f"\n{'='*60}", flush=True)
print(f"{STRATEGY.upper()} SELF-SYNTHESIS RESULTS", flush=True)
print(f"{'='*60}", flush=True)
for k, v in sorted(all_results.items()):
    if v.get("status") == "completed":
        print(f"  R{v['round']}: acc={v['accuracy']}, cells_gen={v['n_cells_generated']}, "
              f"cells_sel={v['n_cells_selected']}, archive={v['archive_size']}, "
              f"correct_gen={v['n_correct_gen']}/{N_GENERATE}", flush=True)
print(f"{'='*60}", flush=True)
