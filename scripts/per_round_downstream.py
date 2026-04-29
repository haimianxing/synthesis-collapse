"""
Per-Round Downstream Fine-tuning: Show Collapse → Performance Degradation

Key experiment: Fine-tune on data from EACH round of iterative generation.
Expected:
- Greedy: performance DEGRADES from R0 to R5 (data becomes redundant)
- QD: performance STABLE or IMPROVES from R0 to R5 (data stays diverse)

Uses existing Math iterative archives (greedy_archive.json, qd_archive.json)
which have per-round data saved within the final archive.

Since we only have final archives, we use a different approach:
1. Re-run a SHORT iterative experiment (50/round × 5 rounds) saving per-round
2. Fine-tune on each round's accumulated data
3. Evaluate on GSM8K

Actually, we'll use the v2 experiments which save per-round archives.
But those are running. So we'll use the EXISTING archives for now and
run a quick iterative experiment with per-round saving.

Strategy: Quick 50/round × 5 rounds × 2 strategies, save per-round,
then fine-tune on each round's data.
"""
import os, sys, json, random, re, time, torch, numpy as np
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
GPU_ID = int(os.environ.get("GPU_ID", "6"))
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

from datasets import load_dataset
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

API_KEY = "sk-bf04f622fcd94499833c70e98dac0803"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-plus"
MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-1___5B-Instruct"

GRID_RES = 10
N_SEED = 20
N_PER_ROUND = 50
N_ROUNDS = 5
DEVICE = "cuda:0"

RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/per_round_downstream")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

print(f"=== Per-Round Downstream (GPU {GPU_ID}) ===", flush=True)

# Load GSM8K
print("Loading GSM8K...", flush=True)
train_pool = load_dataset("gsm8k", "main", split="test")  # Use test as pool
test_ds = load_dataset("gsm8k", "main", split="train")     # Use train as test (avoid contamination)
print(f"Pool: {len(train_pool)}, Test: {len(test_ds)}", flush=True)

def extract_answer(text):
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    return match.group(1).replace(',', '') if match else None

def compute_descriptors(question, answer):
    steps = answer.count('<<') + 1
    difficulty = min(len(answer) / 500.0, 1.0)
    is_multi = 1 if steps >= 3 else 0
    return {'difficulty': difficulty, 'num_steps': min(steps / 10.0, 1.0), 'is_multi_step': is_multi}

def get_cell(desc):
    return (int(desc['difficulty'] * GRID_RES), int(desc['num_steps'] * GRID_RES), int(desc['is_multi_step'] * GRID_RES))

def compute_quality(sample):
    return min(len(sample.get('answer', '')) / 300.0, 1.0)

# Build pool
pool = []
for ex in train_pool:
    q, a = ex['question'], ex['answer']
    ans = extract_answer(a)
    if ans is None: continue
    desc = compute_descriptors(q, a)
    pool.append({'question': q, 'answer': a, 'answer_num': ans,
                 'descriptors': desc, 'quality': compute_quality({'answer': a})})

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
print(f"Seeds: {len(seeds)}", flush=True)

# API functions
def call_api(prompt_text, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": "You are a math education expert."},
                          {"role": "user", "content": prompt_text}],
                temperature=0.85, max_tokens=1024)
            return response.choices[0].message.content
        except Exception as e:
            time.sleep(3 * (attempt + 1))
    return None

def generate_greedy(parent):
    prompt = f"""Create a NEW math word problem similar to:
{parent['question'][:300]}

Include step-by-step solution ending with #### [answer].
**Problem:** [text]
**Solution:** [steps]
#### [answer]"""
    resp = call_api(prompt)
    return parse_math(resp) if resp else None

def generate_qd(parent, grid_state):
    empty_cells = [(di, si, mi) for di in range(GRID_RES) for si in range(GRID_RES) for mi in range(2)
                   if (di, si, mi) not in grid_state]
    if empty_cells:
        t = random.choice(empty_cells)
        d_label = "easy" if t[0]/GRID_RES < 0.3 else ("medium" if t[0]/GRID_RES < 0.7 else "hard")
        s_label = f"{max(1, int(t[1]/GRID_RES*10))} steps"
        m_label = "multi-step" if t[2] else "single-step"
    else:
        d_label, s_label, m_label = "medium", "3 steps", "multi-step"

    prompt = f"""Create a math problem: {d_label}, {s_label}, {m_label}.
**Problem:** [text]
**Solution:** [steps]
#### [answer]"""
    resp = call_api(prompt)
    return parse_math(resp) if resp else None

def parse_math(text):
    if not text: return None
    ans_match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if not ans_match: return None
    answer_num = ans_match.group(1).replace(',', '')
    parts = text.split('**Problem:**')
    if len(parts) >= 2:
        rest = parts[1]
        sol = rest.split('**Solution:**')
        question = sol[0].strip()[:500] if sol else rest[:300]
        answer = sol[1].strip() if len(sol) > 1 else rest
    else:
        question = text[:300]
        answer = text
    if len(question) < 20: return None
    return {'question': question, 'answer': answer, 'answer_num': answer_num}

# Fine-tuning
def fmt_math(sample):
    return f"<|im_start|>system\nSolve the math problem step by step.<|im_end|>\n<|im_start|>user\n{sample.get('question','')[:512]}<|im_end|>\n<|im_start|>assistant\n{sample.get('answer','')[:768]}<|im_end|>"

def finetune_and_eval(train_data, config_name):
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    output_dir = RESULTS_DIR / f"model_{config_name}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"], lora_dropout=0.05, task_type="CAUSAL_LM"))

    texts = [fmt_math(s) for s in train_data if len(s.get('answer', '')) > 20]
    print(f"  [{config_name}] Training on {len(texts)} samples", flush=True)

    if len(texts) < 5:
        print(f"  [{config_name}] Too few samples, skipping", flush=True)
        del model; torch.cuda.empty_cache()
        return {"accuracy": 0, "n_train": len(texts)}

    ds = Dataset.from_dict({"text": texts})
    trainer = SFTTrainer(model=model, args=SFTConfig(
        output_dir=str(output_dir), num_train_epochs=5,
        per_device_train_batch_size=2, gradient_accumulation_steps=8,
        learning_rate=2e-4, logging_steps=10, save_strategy="no",
        bf16=True, report_to="none", max_length=768,
        dataset_text_field="text", packing=False), train_dataset=ds, processing_class=tokenizer)
    trainer.train()

    model.eval()
    correct = total = 0
    for ex in test_ds:
        msgs = [{"role":"system","content":"Solve the math problem step by step."},
                {"role":"user","content":ex['question']}]
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(txt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=256, temperature=0.0, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        pred_match = re.search(r'####\s*(-?[\d,]+\.?\d*)', resp)
        if not pred_match:
            pred_match = re.search(r'(\d+\.?\d*)\s*$', resp.strip())
        pred = pred_match.group(1).replace(',', '') if pred_match else None
        gold = extract_answer(ex['answer'])
        if pred and gold and pred.strip() == gold.strip():
            correct += 1
        total += 1

    del model; torch.cuda.empty_cache()
    result = {"accuracy": round(correct/total, 4), "correct": correct, "total": total, "n_train": len(texts)}
    print(f"  [{config_name}] acc={result['accuracy']} ({correct}/{total})", flush=True)
    return result

# ============ Main: Run iterative generation + per-round evaluation ============
all_results = {}

for strategy in ["greedy", "qd"]:
    print(f"\n{'='*50}\nSTRATEGY: {strategy.upper()}\n{'='*50}", flush=True)

    archive = {}
    round_archives = {}  # Save per-round archives

    for item in seeds:
        cell = get_cell(item['descriptors'])
        if cell not in archive or item['quality'] > archive[cell]['quality']:
            archive[cell] = dict(item)

    # Evaluate R0 (seeds only)
    round_archives[0] = list(archive.values())
    r0_result = finetune_and_eval(list(archive.values()), f"{strategy}_r0")
    all_results[f"{strategy}_r0"] = r0_result

    for rnd in range(1, N_ROUNDS + 1):
        print(f"\n  Round {rnd}/{N_ROUNDS} ({strategy})", flush=True)
        t0 = time.time()
        grid_cells = set(archive.keys())

        for i in range(N_PER_ROUND):
            parents = list(archive.values())
            if strategy == "greedy":
                parent = max(parents, key=lambda x: x['quality'])
                result = generate_greedy(parent)
            else:
                parent = random.choice(parents)
                result = generate_qd(parent, grid_cells)

            if result:
                result['descriptors'] = compute_descriptors(result['question'], result['answer'])
                result['quality'] = compute_quality(result)
                cell = get_cell(result['descriptors'])
                if cell not in archive or result['quality'] > archive[cell]['quality']:
                    archive[cell] = result
                    grid_cells.add(cell)

            if (i+1) % 10 == 0:
                print(f"    {i+1}/{N_PER_ROUND}, cells={len(archive)}", flush=True)

        # Save and evaluate this round
        round_archives[rnd] = list(archive.values())
        n_cells = len(archive)
        print(f"  R{rnd}: {n_cells} cells, evaluating...", flush=True)

        round_result = finetune_and_eval(list(archive.values()), f"{strategy}_r{rnd}")
        round_result['round'] = rnd
        round_result['n_cells'] = n_cells
        round_result['time_s'] = round(time.time() - t0)
        all_results[f"{strategy}_r{rnd}"] = round_result

        # Save intermediate
        with open(RESULTS_DIR / "per_round_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"  R{rnd}: cells={n_cells}, acc={round_result['accuracy']}, time={round_result['time_s']}s", flush=True)

# Summary
print(f"\n{'='*60}", flush=True)
print("PER-ROUND RESULTS:", flush=True)
for strategy in ["greedy", "qd"]:
    print(f"\n  {strategy.upper()}:", flush=True)
    for rnd in range(N_ROUNDS + 1):
        key = f"{strategy}_r{rnd}"
        if key in all_results:
            r = all_results[key]
            print(f"    R{rnd}: acc={r.get('accuracy','?')}, n_train={r.get('n_train','?')}, n_cells={r.get('n_cells','?')}", flush=True)
print(f"{'='*60}", flush=True)

with open(RESULTS_DIR / "per_round_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"Results: {RESULTS_DIR / 'per_round_results.json'}", flush=True)
