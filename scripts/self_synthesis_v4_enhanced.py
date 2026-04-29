"""
Self-Synthesis v4: Enhanced QD-Synth with TRIZ Innovations
============================================================

S3: Golden Ratio Archive Management (>=61.8% seed protection)
    - R0 selected data is marked as "golden seeds"
    - In subsequent rounds, seed data is never removed from accumulated pool
    - Ensures original high-quality data always forms majority of training set
    - Reference: Accumulation Strategy (2404.01413, 116 cites)

S2: Negative-Guided Mutation (anti-archive + score extrapolation)
    - Track worst rejected samples as "anti-patterns"
    - Include anti-patterns in generation prompt to avoid known failures
    - Score extrapolation: s_real = s_base + alpha * (s_base - s_anti_avg)
    - Reference: SIMS negative guidance (2408.16333, 31 cites)

S9: Three-Stage Curriculum (explore -> improve -> consolidate)
    - Stage 1 (R0-R1): Coverage exploration, temp=1.0, q_min=0.05
    - Stage 2 (R2-R3): Quality improvement, temp=0.8, q_min=0.2
    - Stage 3 (R4):    Consolidation, temp=0.7, q_min=0.3, force seed mixing
    - Reference: Accumulation Strategy + Collapse or Thrive (2410.16713)

Base-reset framework: EVERY round trains from BASE model with accumulated data.
This eliminates the LoRA drift confound found in v2 experiments.

GPU Allocation (3 parallel runs):
    GPU 5: STRATEGY=qd_enhanced    SEED=42  (Full S3+S2+S9)
    GPU 6: STRATEGY=qd_enhanced    SEED=123 (Full S3+S2+S9, 2nd seed)
    GPU 7: STRATEGY=qd_s3_s9      SEED=42  (S3+S9 only, ablation without S2)

Usage:
    CUDA_VISIBLE_DEVICES=5 GPU_ID=0 STRATEGY=qd_enhanced SEED=42 \
        PYTHONUNBUFFERED=1 python -u self_synthesis_v4_enhanced.py
"""
import os, sys, json, random, re, torch, numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import time, math

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

GPU_ID = int(os.environ.get("GPU_ID", "0"))
STRATEGY = os.environ.get("STRATEGY", "qd_enhanced")
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
N_SELECT = 400  # Larger than v2's 300
N_EVAL = 500
GOLDEN_RATIO = 0.618  # S3: >=61.8% seed protection
ANTI_ARCHIVE_SIZE = 10  # S2: keep top-10 worst samples
SCORE_EXTRAPOLATION_ALPHA = 0.3  # S2: extrapolation strength

RESULTS_DIR = Path(f"/mnt/data2/zcz/neurIps-emnlp/neurips/results/self_synthesis_v4_enhanced/{STRATEGY}_s{SEED}")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Determine active innovations
S3_ACTIVE = True  # Golden ratio (always active in enhanced strategies)
S2_ACTIVE = "s2" in STRATEGY or STRATEGY == "qd_enhanced"  # Negative guidance
S9_ACTIVE = True  # Curriculum (always active in enhanced strategies)

print(f"=== Self-Synthesis v4 (Enhanced): {STRATEGY.upper()} seed={SEED} (GPU {GPU_ID}) ===", flush=True)
print(f"  S3 Golden Ratio: {S3_ACTIVE} (ratio={GOLDEN_RATIO})", flush=True)
print(f"  S2 Negative Guidance: {S2_ACTIVE} (anti_size={ANTI_ARCHIVE_SIZE}, alpha={SCORE_EXTRAPOLATION_ALPHA})", flush=True)
print(f"  S9 Curriculum: {S9_ACTIVE} (3 stages: explore/improve/consolidate)", flush=True)
print(f"  N_GENERATE={N_GENERATE}, N_SELECT={N_SELECT}, N_ROUNDS={N_ROUNDS}", flush=True)
print(f"  Base-reset: YES (each round trains from BASE with accumulated data)", flush=True)

# ============ S9: Three-Stage Curriculum ============

def get_stage_params(round_num, total_rounds=N_ROUNDS):
    """S9: Return stage-specific parameters for curriculum learning."""
    if not S9_ACTIVE:
        # Default params (no curriculum)
        return {'temperature': 0.8, 'q_min': 0.1, 'stage': 'default', 'seed_mix': False}

    if round_num < total_rounds * 2 // 5:  # R0-R1: Explore (2/5 of rounds)
        return {
            'temperature': 1.0,      # Higher temp for diversity
            'q_min': 0.05,           # Very low quality gate
            'stage': 'explore',
            'seed_mix': False,
            'description': 'Coverage maximization: high temp, low quality gate'
        }
    elif round_num < total_rounds * 4 // 5:  # R2-R3: Improve (2/5 of rounds)
        return {
            'temperature': 0.8,      # Normal temp
            'q_min': 0.2,            # Normal quality gate
            'stage': 'improve',
            'seed_mix': False,
            'description': 'Quality improvement: normal temp, standard quality gate'
        }
    else:  # R4: Consolidate (1/5 of rounds)
        return {
            'temperature': 0.7,      # Lower temp for precision
            'q_min': 0.3,            # Higher quality gate
            'stage': 'consolidate',
            'seed_mix': True,        # Force seed data mixing
            'description': 'Consolidation: lower temp, force golden seed mixing'
        }

# ============ S2: Anti-Archive for Negative Guidance ============

class AntiArchive:
    """S2: Track worst rejected samples as anti-patterns for negative guidance."""
    def __init__(self, max_size=ANTI_ARCHIVE_SIZE):
        self.patterns = []
        self.max_size = max_size

    def update(self, solutions):
        """Update anti-archive with worst rejected samples from this round."""
        # Focus on samples with very low quality but some content (not empty)
        bad_solutions = [s for s in solutions
                        if s['quality'] < 0.15 and s['answer'] and len(s['answer']) > 30]
        if bad_solutions:
            # Keep the worst (most representative failures)
            worst = sorted(bad_solutions, key=lambda x: x['quality'])[:self.max_size]
            self.patterns = worst

    def get_anti_prompt_section(self):
        """Generate anti-pattern prompt section for negative guidance."""
        if not S2_ACTIVE or not self.patterns:
            return ""

        section = "\n\nIMPORTANT - Avoid these common mistakes:\n"
        for i, anti in enumerate(self.patterns[:3]):  # Use top 3 worst examples
            section += f"\n--- Bad Example {i+1} (DO NOT imitate) ---\n"
            section += f"Problem: {anti['question'][:200]}\n"
            section += f"Bad approach: {anti['answer'][:300]}\n"
            if anti.get('correct') is False:
                section += "This approach is WRONG - it gives an incorrect answer.\n"
            section += "---\n"
        section += "\nNow solve the following problem CORRECTLY, avoiding the mistakes above:\n"
        return section

    def extrapolate_score(self, base_score):
        """S2: Score extrapolation away from anti-patterns."""
        if not S2_ACTIVE or not self.patterns:
            return base_score
        avg_anti = np.mean([p['quality'] for p in self.patterns])
        extrapolated = base_score + SCORE_EXTRAPOLATION_ALPHA * (base_score - avg_anti)
        return max(0, min(1, extrapolated))

# ============ S3: Golden Ratio Archive ============

class GoldenRatioArchive:
    """S3: Manage accumulated data with golden ratio seed protection."""
    def __init__(self, golden_ratio=GOLDEN_RATIO):
        self.golden_ratio = golden_ratio
        self.seed_data = []        # R0 selected data (PROTECTED)
        self.round_data = []       # R1+ selected data
        self.all_data = []         # Combined training data
        self.seed_count_history = []  # Track seed ratio over rounds

    def add_round(self, round_num, selected):
        """Add selected data from a round, maintaining golden ratio."""
        if round_num == 0:
            self.seed_data = list(selected)
            print(f"  [S3] Marked {len(self.seed_data)} samples as golden seeds", flush=True)
        else:
            self.round_data.extend(selected)

        # Rebuild combined data with golden ratio enforcement
        total = len(self.seed_data) + len(self.round_data)
        if total == 0:
            self.all_data = []
            return

        seed_ratio = len(self.seed_data) / total
        self.seed_count_history.append((round_num, len(self.seed_data),
                                         len(self.round_data), seed_ratio))

        if seed_ratio >= self.golden_ratio:
            # Seed data already >= 61.8%, use all data
            self.all_data = self.seed_data + self.round_data
            print(f"  [S3] Seed ratio={seed_ratio:.3f} >= {self.golden_ratio}, "
                  f"using all {len(self.all_data)} samples", flush=True)
        else:
            # Need to trim round_data to maintain golden ratio
            max_round = int(len(self.seed_data) * (1 - self.golden_ratio) / self.golden_ratio)
            if len(self.round_data) > max_round:
                # Keep most recent round data (highest quality from latest model)
                self.round_data = self.round_data[-max_round:]
            self.all_data = self.seed_data + self.round_data
            actual_ratio = len(self.seed_data) / len(self.all_data)
            print(f"  [S3] TRIMMED: seed={len(self.seed_data)}, round={len(self.round_data)}, "
                  f"total={len(self.all_data)}, ratio={actual_ratio:.3f}", flush=True)

    def get_consolidation_data(self):
        """S9 Stage 3: Get data with forced seed mixing."""
        if not self.seed_data:
            return self.all_data
        # In consolidation stage, ensure seeds are emphasized
        # Duplicate seed data to increase its weight
        enhanced = list(self.seed_data) + list(self.all_data)
        print(f"  [S9] Consolidation: {len(self.seed_data)} seeds + {len(self.all_data)} all "
              f"= {len(enhanced)} total (seeds counted 2x)", flush=True)
        return enhanced

# ============ Reuse descriptor/quality functions ============

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

# ============ Enhanced Selection Functions ============

def select_qd_enhanced(solutions, n, archive_cells, anti_archive=None):
    """Enhanced QD selection with score extrapolation (S2)."""
    cell_to_items = defaultdict(list)
    for sol in solutions:
        if sol['cell'] and sol['quality'] > 0.05:  # S9 Stage 1: lower quality gate
            # S2: Apply score extrapolation
            if anti_archive and S2_ACTIVE:
                sol['enhanced_quality'] = anti_archive.extrapolate_score(sol['quality'])
            else:
                sol['enhanced_quality'] = sol['quality']
            cell_to_items[sol['cell']].append(sol)

    if not cell_to_items: return select_greedy(solutions, n)

    selected = []
    # Priority 1: Fill empty cells (surprisal) — but with enhanced quality
    for cell in [c for c in cell_to_items if c not in archive_cells]:
        selected.append(max(cell_to_items[cell], key=lambda x: x.get('enhanced_quality', x['quality'])))

    # Priority 2: Best from existing cells (per-cell elitism)
    if len(selected) < n:
        for cell in sorted(cell_to_items.keys()):
            best = max(cell_to_items[cell], key=lambda x: x.get('enhanced_quality', x['quality']))
            if best not in selected:
                selected.append(best)
                if len(selected) >= n: break

    # Priority 3: Fill remaining by quality
    if len(selected) < n:
        for item in sorted([s for s in solutions if s not in selected and s['quality'] > 0.05],
                          key=lambda x: x.get('enhanced_quality', x['quality']), reverse=True):
            if len(selected) >= n: break
            selected.append(item)

    return selected[:n]

def select_greedy(solutions, n):
    return sorted([s for s in solutions if s['quality'] > 0.1],
                  key=lambda x: x['quality'], reverse=True)[:n]

# ============ Generation with S2 Negative Guidance ============

BASE_SYSTEM = """You are a mathematics expert. Solve the given math problem step by step.

Instructions:
1. Read the problem carefully.
2. Break it down into steps.
3. Show all calculations.
4. Write your final numerical answer after ####

Example format:
John has 5 apples and buys 3 more.
Step 1: Total = 5 + 3 = 8
#### 8"""

def generate_solutions(model, tokenizer, prompts, gold_answers, stage_params, anti_archive):
    """Generate solutions with S2 negative guidance and S9 stage-aware temperature."""
    solutions = []
    temperature = stage_params['temperature']
    q_min = stage_params['q_min']

    for i, prompt in enumerate(prompts):
        gold = gold_answers[i]
        # S2: Add negative guidance to system prompt
        sys_prompt = BASE_SYSTEM
        if S2_ACTIVE and anti_archive:
            sys_prompt += anti_archive.get_anti_prompt_section()

        msgs = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}]
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(txt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=1024, temperature=temperature,
                               do_sample=True, top_p=0.9, pad_token_id=tokenizer.eos_token_id)

        resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        pred = extract_flexible(resp)
        q = quality_score(resp, gold)

        solutions.append({
            'question': prompt, 'answer': resp,
            'cell': get_math_cell(resp),
            'quality': q,
            'enhanced_quality': anti_archive.extrapolate_score(q) if (S2_ACTIVE and anti_archive) else q,
            'correct': check_correct(pred, gold)
        })

        if (i+1) % 100 == 0:
            nc = sum(1 for s in solutions if s['correct'])
            nv = sum(1 for s in solutions if s['quality'] > q_min)
            print(f"    Gen {i+1}/{len(prompts)} (stage={stage_params['stage']}, "
                  f"T={temperature:.1f}, {nv}v, {nc}c)", flush=True)
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

# ============ Main Experiment Loop ============

print("Loading GSM8K...", flush=True)
gsm8k_train = list(load_dataset("gsm8k", "main", split="train"))
gsm8k_test = list(load_dataset("gsm8k", "main", split="test"))
prompt_pool = [(ex['question'], extract_answer(ex['answer'])) for ex in gsm8k_train]

results_file = RESULTS_DIR / f"{STRATEGY}_s{SEED}_v4.json"
all_results = json.load(open(results_file)) if results_file.exists() else {}
archive_cells = set()

# Restore state from previous results
for k in sorted(all_results.keys()):
    r = all_results[k]
    if r.get("status") == "completed":
        for c in r.get("archive_cells", []):
            archive_cells.add(tuple(c))

# Initialize S2 anti-archive and S3 golden ratio archive
anti_archive = AntiArchive(max_size=ANTI_ARCHIVE_SIZE)
golden_archive = GoldenRatioArchive(golden_ratio=GOLDEN_RATIO)

# Restore golden archive from previous results
for k in sorted(all_results.keys()):
    r = all_results[k]
    if r.get("status") == "completed" and "selected_data" in r:
        selected_data = r["selected_data"]
        # Reconstruct cell info
        for s in selected_data:
            if s.get('cell'):
                s['cell'] = tuple(s['cell']) if isinstance(s['cell'], list) else s['cell']
        rnd = r['round']
        golden_archive.add_round(rnd, selected_data)

# Main round loop
for rnd in range(N_ROUNDS):
    rnd_key = f"{STRATEGY}_s{SEED}_r{rnd}"
    t0 = time.time()

    if rnd_key in all_results and all_results[rnd_key].get("status") == "completed":
        print(f"  {rnd_key}: done (acc={all_results[rnd_key].get('accuracy','?')}), skipping", flush=True)
        for c in all_results[rnd_key].get("archive_cells", []):
            archive_cells.add(tuple(c))
        # Update anti_archive from this round's rejected samples
        if all_results[rnd_key].get("anti_patterns"):
            anti_archive.patterns = all_results[rnd_key]["anti_patterns"]
        continue

    # S9: Get stage-specific parameters
    stage = get_stage_params(rnd)
    print(f"\n  ROUND {rnd} ({STRATEGY} seed={SEED}) [BASE-RESET + S3+S2+S9]", flush=True)
    print(f"  Stage: {stage['stage']} — {stage.get('description', '')}", flush=True)

    torch.manual_seed(SEED + rnd); random.seed(SEED + rnd); np.random.seed(SEED + rnd)

    # Step 1: Generate from BASE model with S2 negative guidance
    print(f"  Loading BASE model for generation...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    gen_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
                                                      device_map=DEVICE, trust_remote_code=True)
    gen_model.eval()

    rng = random.Random(SEED + rnd)
    sampled = rng.sample(prompt_pool, min(N_GENERATE, len(prompt_pool)))
    prompts = [q for q, a in sampled]
    golds = [a for q, a in sampled]

    solutions = generate_solutions(gen_model, tokenizer, prompts, golds, stage, anti_archive)

    n_valid = sum(1 for s in solutions if s['quality'] > stage['q_min'])
    n_correct = sum(1 for s in solutions if s['correct'])
    n_cells = len(set(s['cell'] for s in solutions if s['cell']))
    gen_entropy = compute_cell_entropy(solutions)
    gen_strategies = compute_unique_strategies(solutions)
    avg_quality = np.mean([s['quality'] for s in solutions if s['quality'] > 0.05]) if n_valid > 0 else 0

    print(f"  Gen (base, T={stage['temperature']:.1f}): {n_valid}v/{N_GENERATE}, "
          f"{n_correct}c, {n_cells} cells, H={gen_entropy:.2f}, q̄={avg_quality:.3f}", flush=True)

    # S2: Update anti-archive with worst rejected samples
    anti_archive.update(solutions)
    if S2_ACTIVE:
        print(f"  [S2] Anti-archive updated: {len(anti_archive.patterns)} anti-patterns", flush=True)

    # Step 2: Select with enhanced QD
    selected = select_qd_enhanced(solutions, N_SELECT, archive_cells, anti_archive)

    sel_cells = set(s['cell'] for s in selected if s['cell'])
    sel_correct = sum(1 for s in selected if s['correct'])
    sel_entropy = compute_cell_entropy(selected)
    sel_avg_quality = np.mean([s['quality'] for s in selected]) if selected else 0
    archive_cells.update(sel_cells)

    # S3: Add to golden ratio archive
    serializable_selected = [
        {'question': s['question'][:512], 'answer': s['answer'][:2048],
         'cell': list(s['cell']) if s['cell'] else None,
         'quality': s['quality'], 'correct': s['correct']}
        for s in selected
    ]
    golden_archive.add_round(rnd, selected)

    print(f"  Sel: {len(selected)}, {len(sel_cells)} cells, {sel_correct}c, "
          f"H={sel_entropy:.2f}, q̄={sel_avg_quality:.3f}", flush=True)
    print(f"  [S3] Accumulated: seeds={len(golden_archive.seed_data)}, "
          f"rounds={len(golden_archive.round_data)}, total={len(golden_archive.all_data)}", flush=True)

    # Free generation model
    del gen_model; torch.cuda.empty_cache()

    # Step 3: Train from BASE model on accumulated data
    # S9: In consolidation stage, use enhanced data with seed mixing
    if S9_ACTIVE and stage.get('seed_mix'):
        training_data = golden_archive.get_consolidation_data()
    else:
        training_data = golden_archive.all_data

    print(f"  Loading BASE model for training on {len(training_data)} samples...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    train_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
                                                        device_map=DEVICE, trust_remote_code=True)

    q_min_threshold = stage['q_min']
    texts = [fmt_sample(s) for s in training_data if s['quality'] > q_min_threshold]
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

    # Save results
    all_results[rnd_key] = {
        "round": rnd, "seed": SEED, "strategy": STRATEGY,
        "experiment": "enhanced_v4",
        "stage": stage['stage'],
        "stage_temperature": stage['temperature'],
        "stage_q_min": stage['q_min'],
        "stage_seed_mix": stage.get('seed_mix', False),
        # S3 metrics
        "golden_seeds": len(golden_archive.seed_data),
        "round_data_count": len(golden_archive.round_data),
        "accumulated_total": len(golden_archive.all_data),
        "seed_ratio": round(len(golden_archive.seed_data) / max(1, len(golden_archive.all_data)), 3),
        "n_accumulated_samples": len(training_data),
        # Generation metrics
        "n_generated": n_valid, "n_correct_gen": n_correct,
        "n_cells_generated": n_cells, "gen_entropy": round(gen_entropy, 4),
        "gen_strategies": gen_strategies,
        "avg_quality_gen": round(avg_quality, 4),
        # Selection metrics
        "n_selected": len(selected), "n_cells_selected": len(sel_cells),
        "n_correct_selected": sel_correct, "sel_entropy": round(sel_entropy, 4),
        "sel_avg_quality": round(sel_avg_quality, 4),
        # S2 metrics
        "anti_archive_size": len(anti_archive.patterns),
        "s2_active": S2_ACTIVE,
        # Archive
        "archive_cells": [list(c) for c in archive_cells],
        "archive_size": len(archive_cells),
        # Eval
        "accuracy": accuracy, "correct": correct, "total": total,
        "elapsed_sec": elapsed, "status": "completed",
        # Data for restart
        "selected_data": serializable_selected,
    }

    # Save anti-patterns for restart (simplified)
    if anti_archive.patterns:
        all_results[rnd_key]["anti_patterns"] = [
            {'question': p['question'][:200], 'answer': p['answer'][:300],
             'quality': p['quality'], 'correct': p['correct']}
            for p in anti_archive.patterns
        ]

    with open(results_file, "w") as f: json.dump(all_results, f, indent=2)
    print(f"  R{rnd}: acc={accuracy} ({correct}/{total}), "
          f"stage={stage['stage']}, T={stage['temperature']:.1f}, "
          f"accumulated={len(training_data)}, "
          f"seed_ratio={len(golden_archive.seed_data)/max(1,len(golden_archive.all_data)):.3f}, "
          f"{elapsed}s", flush=True)

# ============ Summary ============
print(f"\n=== {STRATEGY} seed={SEED} (Enhanced v4) DONE ===", flush=True)
for k, v in sorted(all_results.items()):
    print(f"  {k}: acc={v['accuracy']}, stage={v.get('stage','?')}, "
          f"T={v.get('stage_temperature','?')}, "
          f"cells={v.get('n_cells_generated','?')}, H={v.get('gen_entropy','?')}, "
          f"seed_ratio={v.get('seed_ratio','?')}, "
          f"accumulated={v.get('n_accumulated_samples','?')}", flush=True)
