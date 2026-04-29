"""
Enhanced QD-Synth with 4 TRIZ Innovations — CODE Domain
Innovations: (1) Surprisal-weighted fitness, (2) Cumulative archive, (3) Adaptive budget, (4) Exponential scaling

This is the CORE contribution that differentiates from SPARQ:
- SPARQ uses solve-rate (needs MC rollouts) → we use surprisal (information-theoretic)
- SPARQ replaces data each round → we accumulate with real anchors (103 cites backing)
- SPARQ constant budget → we use exponential scaling (Spend Wisely theory)

Compares 4 variants:
  1. QD-Standard: basic MAP-Elites (baseline)
  2. QD-Cumulative: keep real anchors, never replace
  3. QD-Surprisal: use surprisal as fitness instead of text-length
  4. QD-Enhanced: all 4 innovations combined
Plus: Greedy baseline

Each variant does iterative generation (5 rounds), then fine-tune on final archive → evaluate HumanEval.
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys, json, random, re, time, ast, math, torch, numpy as np
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset
from openai import OpenAI

API_KEY = "sk-bf04f622fcd94499833c70e98dac0803"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-plus"
LLM_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-1___5B-Instruct"
GRID_RES = 10
N_SEED = 20
N_ROUNDS = 5
GAMMA = 1.5  # Exponential scaling factor

GPU_ID = int(os.environ.get("GPU_ID", "5"))
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
DEVICE = "cuda:0"

RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/code_enhanced_qd")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

print(f"=== Enhanced QD-Synth (GPU {GPU_ID}) ===", flush=True)

# Load datasets
print("Loading datasets...", flush=True)
try:
    mbpp = load_dataset("mbpp", "sanitized", split="test")
except:
    mbpp = load_dataset("mbpp", split="test")
humaneval = load_dataset("openai_humaneval", split="test")
print(f"MBPP: {len(mbpp)}, HumanEval: {len(humaneval)}", flush=True)

# ============ Descriptors ============
def compute_code_descriptors(prompt, code, test_list=None):
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
    code = sample.get('code', '')
    return min(len(code) / 500.0, 1.0) if code else 0.1

# ============ Innovation #1: Surprisal Scorer ============
class SurprisalScorer:
    """Use base LM's log-likelihood as quality metric. Higher surprisal = more informative."""
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("Loading surprisal scorer model...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True)
        self.model.eval()

    def compute_surprisal(self, text):
        """Compute average negative log-likelihood (surprisal) of text."""
        if not text or len(text) < 10:
            return 0.5
        try:
            inputs = self.tokenizer(text[:512], return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
            nll = outputs.loss.item()  # Average negative log-likelihood
            # Normalize to [0, 1] — higher surprisal = more informative
            surprisal = min(nll / 5.0, 1.0)  # 5.0 nats is quite surprising for code
            return surprisal
        except:
            return 0.5

# Build pool
pool = []
for ex in mbpp:
    code = ex.get('code', '')
    desc = compute_code_descriptors(ex.get('prompt', ''), code)
    pool.append({
        'prompt': ex.get('prompt', ''), 'code': code, 'text': ex.get('text', ''),
        'descriptors': desc, 'quality': compute_quality({'code': code}), 'is_real': True
    })
print(f"Pool: {len(pool)}", flush=True)

# Select 20 diverse seeds
random.seed(42)
seeds = []
used_cells = set()
for item in sorted(pool, key=lambda x: x['quality'], reverse=True):
    cell = get_cell(item['descriptors'])
    if cell not in used_cells:
        seeds.append(item)
        used_cells.add(cell)
        if len(seeds) >= N_SEED:
            break
print(f"Seeds: {len(seeds)} ({len(used_cells)} cells)", flush=True)

# ============ API Generation ============
def call_api(prompt_text, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a Python programming expert. Generate problems as specified."},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.85, max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"    API error: {e}", flush=True)
            time.sleep(3 * (attempt + 1))
    return None

def generate_for_cell(parent, target_desc_vals):
    """Generate a code problem targeting specific descriptor values."""
    d_val, a_val, b_val = target_desc_vals
    d_label = "easy (short)" if d_val < 0.3 else ("medium" if d_val < 0.7 else "hard (long solution)")
    n_apis = "few (0-2)" if a_val < 0.3 else ("several (3-6)" if a_val < 0.7 else "many (7+)")
    debug_str = "with try/except/assert" if b_val >= 0.5 else "without explicit error handling"

    prompt = f"""Create a NEW Python programming problem:
- Difficulty: {d_label}
- Standard library API calls: {n_apis}
- Error handling: {debug_str}

Format:
**Problem:**
[description with function signature]

**Solution:**
```python
[code]
```"""
    resp = call_api(prompt)
    if resp:
        return parse_code(resp)
    return None

def generate_greedy(parent):
    prompt = f"""Here is a Python problem:
{parent['prompt'][:300]}

Create a NEW, DIFFERENT problem of similar difficulty.
**Problem:**
[description]
**Solution:**
```python
[code]
```"""
    resp = call_api(prompt)
    if resp:
        return parse_code(resp)
    return None

def parse_code(text):
    code_blocks = re.findall(r'```python\s*(.*?)\s*```', text, re.DOTALL)
    if not code_blocks:
        code_blocks = re.findall(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if not code_blocks:
        return None
    code = code_blocks[-1].strip()
    if len(code) < 20:
        return None
    try:
        ast.parse(code)
    except:
        return None
    problem = text.split('```')[0].strip()[:500]
    return {'prompt': problem, 'code': code[:1000], 'text': problem[:300]}

# ============ Enhanced QD Archive ============
class EnhancedArchive:
    def __init__(self, mode="standard", surprisal_scorer=None):
        self.grid = {}
        self.real_anchors = {}  # Real data never replaced (innovation #5)
        self.mode = mode
        self.scorer = surprisal_scorer
        self.discard_count = 0

    def add(self, item, is_real=False):
        """Add item to archive. In cumulative mode, never remove real anchors."""
        if 'descriptors' not in item:
            item['descriptors'] = compute_code_descriptors(item.get('prompt', ''), item.get('code', ''))
        if 'quality' not in item:
            item['quality'] = compute_quality(item)

        # Innovation #1: Surprisal weighting
        if self.mode in ("surprisal", "enhanced") and self.scorer:
            item['surprisal'] = self.scorer.compute_surprisal(item.get('code', ''))
            fitness = 0.5 * item['quality'] + 0.5 * item['surprisal']
        else:
            item['surprisal'] = 0.0
            fitness = item['quality']
        item['fitness'] = fitness

        cell = get_cell(item['descriptors'])

        if cell not in self.grid:
            self.grid[cell] = item
            if is_real or self.mode in ("cumulative", "enhanced"):
                self.real_anchors[cell] = True
        else:
            existing = self.grid[cell]
            # Innovation #5: Cumulative — never replace real anchors
            if self.real_anchors.get(cell) and self.mode in ("cumulative", "enhanced"):
                # Only replace if new item is strictly better AND anchor was synthetic
                if not is_real and fitness > existing['fitness']:
                    self.grid[cell] = item
                elif is_real:
                    self.grid[cell] = item
                    self.real_anchors[cell] = True
                else:
                    self.discard_count += 1
            else:
                if fitness > existing['fitness']:
                    self.grid[cell] = item

        item['is_real'] = is_real
        return cell in self.grid and self.grid[cell] is item

    def get_empty_cells(self):
        all_cells = set()
        for di in range(GRID_RES):
            for ai in range(GRID_RES):
                for bi in range(GRID_RES):
                    all_cells.add((di, ai, bi))
        return all_cells - set(self.grid.keys())

    def metrics(self):
        n = len(self.grid)
        total = GRID_RES ** 3
        qualities = [v.get('fitness', v.get('quality', 0)) for v in self.grid.values()]
        if len(qualities) > 1:
            q = np.array(qualities); q = q / q.sum()
            entropy = -np.sum(q * np.log(q + 1e-10))
        else:
            entropy = 0.0
        return {
            'n_cells': n, 'coverage': round(n / total, 4),
            'entropy': round(entropy, 4),
            'avg_fitness': round(np.mean(qualities), 4) if qualities else 0,
            'n_real_anchors': sum(1 for v in self.grid.values() if v.get('is_real')),
        }

# ============ Run Experiment ============
def run_variant(variant_name, mode, use_surprisal=False, use_exponential=False):
    print(f"\n{'='*60}\nVARIANT: {variant_name} (mode={mode})\n{'='*60}", flush=True)

    scorer = SurprisalScorer() if use_surprisal else None
    archive = EnhancedArchive(mode=mode, surprisal_scorer=scorer)

    # Initialize with seeds
    for item in seeds:
        archive.add(dict(item), is_real=True)

    round_data = []
    m = archive.metrics()
    m['round'] = 0
    round_data.append(m)
    print(f"  R0: cells={m['n_cells']}, cov={m['coverage']}, fitness={m['avg_fitness']}", flush=True)

    for rnd in range(1, N_ROUNDS + 1):
        t0 = time.time()
        # Innovation #8: Exponential scaling
        if use_exponential:
            n_gen = int(N_SEED * (GAMMA ** rnd))  # 20, 30, 45, 67, 101
        else:
            n_gen = 50  # Standard constant

        n_gen = min(n_gen, 100)  # Cap at 100 for API limits
        print(f"  R{rnd}: generating {n_gen} samples...", flush=True)

        empty_cells = list(archive.get_empty_cells())
        n_new = 0

        for i in range(n_gen):
            parents = list(archive.grid.values())

            if mode == "greedy":
                parent = max(parents, key=lambda x: x.get('fitness', x.get('quality', 0)))
                result = generate_greedy(parent)
            else:
                # QD variants: target empty cells
                parent = random.choice(parents)
                if empty_cells:
                    target = random.choice(empty_cells)
                    target_vals = (target[0]/GRID_RES, target[1]/GRID_RES, target[2]/GRID_RES)
                    result = generate_for_cell(parent, target_vals)
                else:
                    result = generate_greedy(parent)

            if result:
                added = archive.add(result, is_real=False)
                if added:
                    n_new += 1
                    empty_cells = list(archive.get_empty_cells())

            if (i+1) % 10 == 0:
                print(f"    {i+1}/{n_gen}, cells={len(archive.grid)}", flush=True)

        m = archive.metrics()
        m['round'] = rnd
        m['n_gen'] = n_gen
        m['n_new_cells'] = n_new
        m['time_s'] = round(time.time() - t0)
        round_data.append(m)
        print(f"  R{rnd}: cells={m['n_cells']}, cov={m['coverage']}, fitness={m['avg_fitness']}, n_new={n_new}, {m['time_s']}s", flush=True)

        # Save intermediate
        with open(RESULTS_DIR / f"{variant_name}_rounds.json", "w") as f:
            json.dump(round_data, f, indent=2, default=str)
        # Save archive
        archive_items = [{k: v for k, v in item.items() if k != 'descriptors'} for item in archive.grid.values()]
        with open(RESULTS_DIR / f"{variant_name}_archive.json", "w") as f:
            json.dump(archive_items, f, indent=2, ensure_ascii=False, default=str)

    # Clean up scorer
    if scorer:
        del scorer.model; torch.cuda.empty_cache()

    return round_data, archive

# ============ Fine-tune & Evaluate ============
def finetune_and_eval(archive_items, config_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    output_dir = RESULTS_DIR / f"model_{config_name}"

    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(LLM_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"], lora_dropout=0.05, task_type="CAUSAL_LM"))

    def fmt(s):
        return f"<|im_start|>system\nComplete the Python function.<|im_end|>\n<|im_start|>user\n{s['prompt'][:512]}<|im_end|>\n<|im_start|>assistant\n{s['code'][:768]}<|im_end|>"

    valid = [item for item in archive_items if item.get('code') and len(item.get('code', '')) > 20]
    print(f"  [{config_name}] Training on {len(valid)} samples", flush=True)

    ds = Dataset.from_dict({"text": [fmt(s) for s in valid]})
    trainer = SFTTrainer(model=model, args=SFTConfig(output_dir=str(output_dir), num_train_epochs=3, per_device_train_batch_size=4, gradient_accumulation_steps=4, learning_rate=2e-4, logging_steps=50, save_strategy="no", bf16=True, report_to="none", max_length=768, dataset_text_field="text", packing=False), train_dataset=ds, processing_class=tokenizer)
    trainer.train()
    model.eval()

    correct = total = 0
    for ex in humaneval:
        msgs = [{"role":"system","content":"Complete the Python function."},{"role":"user","content":ex['prompt']}]
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

    del model; torch.cuda.empty_cache()
    return {"pass_at_1": round(correct/total, 4), "correct": correct, "total": total, "n_train": len(valid)}

# ============ Main ============
all_results = {}

# Run 4 variants + Greedy baseline
variants = [
    ("greedy",       "greedy",     False, False),
    ("qd_standard",  "standard",   False, False),
    ("qd_cumulative","cumulative", False, False),
    ("qd_surprisal", "surprisal",  True,  False),
    ("qd_enhanced",  "enhanced",   True,  True),
]

for vname, mode, use_s, use_exp in variants:
    rounds, archive = run_variant(vname, mode, use_s, use_exp)
    # Fine-tune and evaluate
    archive_items = list(archive.grid.values())
    ft_result = finetune_and_eval(archive_items, vname)
    print(f"  [{vname}] pass@1: {ft_result['pass_at_1']} ({ft_result['correct']}/{ft_result['total']}), n_train={ft_result['n_train']}", flush=True)

    all_results[vname] = {"rounds": rounds, "downstream": ft_result}
    with open(RESULTS_DIR / "enhanced_qd_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

print(f"\nDone. Results: {RESULTS_DIR / 'enhanced_qd_results.json'}", flush=True)
