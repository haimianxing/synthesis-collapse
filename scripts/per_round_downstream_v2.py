"""
Per-Round Downstream Evaluation v2 (Fixed)
Trains directly on ARCHIVE DATA (API-generated code/math) instead of cell-matched real data.

Key fix: Old version matched archive cells to real MBPP samples (20-32 matches → ~1% pass@1).
         New version uses archive items directly (49-118 items → expected 15-60% pass@1).

GPU assignment:
  GPU_ID=1 DOMAIN=code STRATEGY=greedy   → Code Greedy R0-R7
  GPU_ID=2 DOMAIN=code STRATEGY=qd       → Code QD R0-R7
  GPU_ID=3 DOMAIN=math STRATEGY=greedy   → Math Greedy R0-R7
  GPU_ID=4 DOMAIN=math STRATEGY=qd       → Math QD R0-R7
"""
import os, sys, json, random, re, torch, numpy as np, ast, time
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

GPU_ID = int(os.environ.get("GPU_ID", "1"))
DOMAIN = os.environ.get("DOMAIN", "code")  # "code" or "math"
STRATEGY = os.environ.get("STRATEGY", "greedy")  # "greedy" or "qd"
N_SEEDS = 3

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-1___5B-Instruct"
DEVICE = "cuda:0"

RESULTS_DIR = Path(f"/mnt/data2/zcz/neurIps-emnlp/neurips/results/per_round_v2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Per-config results file to avoid concurrent write conflicts
RESULTS_FILE = RESULTS_DIR / f"{DOMAIN}_{STRATEGY}_results.json"
if RESULTS_FILE.exists():
    with open(RESULTS_FILE) as f:
        all_results = json.load(f)
    print(f"Loaded {len(all_results)} existing results from {RESULTS_FILE.name}", flush=True)
else:
    all_results = {}

print(f"=== Per-Round Downstream v2: {DOMAIN.upper()} {STRATEGY.upper()} (GPU {GPU_ID}) ===", flush=True)

# ============ Domain-specific setup ============
if DOMAIN == "code":
    ITER_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/code_iterative_v2")
    ROUNDS = list(range(8))

    def fmt_sample(item):
        prompt = item.get('prompt', '')[:512]
        code = item.get('code', '')[:1024]
        return f"<|im_start|>system\nComplete the Python function.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{code}<|im_end|>"

    def is_valid(item):
        code = item.get('code', '')
        return code and len(code) > 20

    print("Loading HumanEval for evaluation...", flush=True)
    humaneval = load_dataset("openai_humaneval", split="test")
    print(f"HumanEval: {len(humaneval)} problems", flush=True)

    def eval_model(model, tokenizer):
        correct = total = 0
        for i, ex in enumerate(humaneval):
            msgs = [{"role":"system","content":"Complete the Python function."},
                    {"role":"user","content":ex['prompt']}]
            txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = tokenizer(txt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=256, temperature=0.0,
                                   do_sample=False, pad_token_id=tokenizer.eos_token_id)
            resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
            try:
                exec_globals = {}; exec(ex['prompt'] + resp, exec_globals); exec(ex['test'], exec_globals)
                correct += 1
            except: pass
            total += 1
            if (i+1) % 40 == 0:
                print(f"    Eval: {i+1}/164, correct={correct}", flush=True)
        return correct, total

elif DOMAIN == "math":
    ITER_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/math_iterative_v2")
    ROUNDS = list(range(8))

    def fmt_sample(item):
        question = item.get('question', '')[:512]
        answer = item.get('answer', '')[:1024]
        return f"<|im_start|>system\nSolve the math problem step by step.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"

    def is_valid(item):
        answer = item.get('answer', '')
        return answer and len(answer) > 20

    def extract_answer(text):
        match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
        return match.group(1).replace(',', '') if match else None

    print("Loading GSM8K for evaluation...", flush=True)
    gsm8k_test_full = load_dataset("gsm8k", "main", split="train")
    random.seed(42)
    gsm8k_test = random.sample(list(gsm8k_test_full), min(200, len(gsm8k_test_full)))
    print(f"GSM8K eval: {len(gsm8k_test)} problems", flush=True)

    def eval_model(model, tokenizer):
        correct = total = 0
        for i, ex in enumerate(gsm8k_test):
            msgs = [{"role":"system","content":"Solve the math problem step by step."},
                    {"role":"user","content":ex['question']}]
            txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = tokenizer(txt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=256, temperature=0.0, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
            resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
            pred_match = re.search(r'####\s*(-?[\d,]+\.?\d*)', resp)
            if not pred_match:
                pred_match = re.search(r'(\d+\.?\d*)\s*$', resp.strip())
            pred = pred_match.group(1).replace(',', '') if pred_match else None
            gold = extract_answer(ex['answer'])
            if pred and gold and pred.strip() == gold.strip():
                correct += 1
            total += 1
            if (i+1) % 50 == 0:
                print(f"    Eval: {i+1}/{len(gsm8k_test)}, correct={correct}", flush=True)
        return correct, total

# ============ Fine-tuning and evaluation ============
def finetune_and_eval(archive_items, config_name, seed=42):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    output_dir = RESULTS_DIR / f"model_{config_name}_s{seed}"

    # Filter valid items
    valid_items = [item for item in archive_items if is_valid(item)]
    texts = [fmt_sample(item) for item in valid_items]
    print(f"  [{config_name} s{seed}] Training on {len(texts)} generated samples", flush=True)

    if len(texts) < 5:
        metric_name = "pass_at_1" if DOMAIN == "code" else "accuracy"
        return {metric_name: 0, "correct": 0, "total": 0, "n_train": len(texts)}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
                                                  device_map=DEVICE, trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj"], lora_dropout=0.05, task_type="CAUSAL_LM"))

    ds = Dataset.from_dict({"text": texts})
    trainer = SFTTrainer(model=model, args=SFTConfig(
        output_dir=str(output_dir), num_train_epochs=3,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=2e-4, logging_steps=50, save_strategy="no",
        bf16=True, report_to="none", max_length=1024,
        dataset_text_field="text", packing=False),
        train_dataset=ds, processing_class=tokenizer)
    trainer.train()

    model.eval()
    correct, total = eval_model(model, tokenizer)

    del model; torch.cuda.empty_cache()
    metric = round(correct/total, 4) if total > 0 else 0
    metric_name = "pass_at_1" if DOMAIN == "code" else "accuracy"
    print(f"  [{config_name} s{seed}] {metric_name}={metric} ({correct}/{total})", flush=True)
    return {metric_name: metric, "correct": correct, "total": total, "n_train": len(texts)}

# ============ Main ============
print(f"\nRunning {STRATEGY.upper()} rounds: {ROUNDS}", flush=True)

for rnd in ROUNDS:
    key = f"r{rnd}"
    if key in all_results:
        print(f"  {key}: already done ({all_results[key].get('pass_at_1', all_results[key].get('accuracy', '?'))}), skipping", flush=True)
        continue

    archive_path = ITER_DIR / f"{STRATEGY}_archive_r{rnd}.json"
    if not archive_path.exists():
        print(f"  R{rnd}: archive not found, skipping", flush=True)
        continue

    with open(archive_path) as f:
        archive_items = json.load(f)

    print(f"\n  R{rnd}: {len(archive_items)} items in archive", flush=True)

    seed_results = []
    for seed in [42, 123, 271][:N_SEEDS]:
        result = finetune_and_eval(archive_items, f"{STRATEGY}_r{rnd}", seed)
        result['seed'] = seed
        seed_results.append(result)

    metric_name = "pass_at_1" if DOMAIN == "code" else "accuracy"
    values = [r[metric_name] for r in seed_results]
    agg = {
        "round": rnd,
        "strategy": STRATEGY,
        "domain": DOMAIN,
        "n_archive": len(archive_items),
        "n_valid": sum(1 for item in archive_items if is_valid(item)),
        metric_name: round(np.mean(values), 4),
        "std": round(np.std(values), 4),
        "seeds": seed_results
    }
    all_results[key] = agg
    print(f"  R{rnd}: {metric_name}={agg[metric_name]}±{agg['std']}, n_train={agg['n_valid']}", flush=True)

    # Save after each round
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

# ============ Summary ============
print(f"\n{'='*60}", flush=True)
print(f"PER-ROUND v2 RESULTS ({DOMAIN.upper()} {STRATEGY.upper()})", flush=True)
print(f"{'='*60}", flush=True)
metric_name = "pass_at_1" if DOMAIN == "code" else "accuracy"
for key in sorted(all_results.keys()):
    if key.startswith(f"{DOMAIN}_{STRATEGY}"):
        r = all_results[key]
        print(f"  R{r['round']}: {metric_name}={r[metric_name]}±{r['std']}, n={r['n_valid']}", flush=True)
print(f"{'='*60}", flush=True)

with open(results_file, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"Results saved: {results_file}", flush=True)
