"""
P0: Full Fine-Tuning Experiment (non-LoRA) — Eliminate "LoRA artifact" concern
Compares full parameter fine-tuning vs LoRA on Code domain (1.5B model).

Design:
- 4 strategies × 5 seeds × 2 modes (full-FT vs LoRA) = 40 configs
- Same MBPP pool, same selections as code_8seed experiment
- Full FT: lr=5e-6, 3 epochs, weight_decay=0.01
- Evaluate on HumanEval pass@1 (greedy decoding)

Key question: Does full fine-tuning reduce training noise (std)?
- If yes → coverage gaps become MORE detectable → strengthens paper
- If no → confirms noise is inherent, not LoRA-specific → also strengthens paper
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys, json, random, re, torch, numpy as np, time, ast
from pathlib import Path
from datasets import load_dataset

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-1___5B-Instruct"
SEEDS = [42, 123, 456, 789, 1024]  # 5 seeds (enough for paired t-test)
GRID_RES = 10
GPU_ID = int(os.environ.get("GPU_ID", "1"))
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
DEVICE = "cuda:0"

RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/full_finetune")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = RESULTS_DIR / f"full_ft_gpu{GPU_ID}.json"

MODE = os.environ.get("MODE", "full")  # "full" or "lora_control"

print(f"=== Full Fine-tuning Experiment (GPU {GPU_ID}, mode={MODE}) ===", flush=True)

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

# Build pool
pool = []
for ex in mbpp_train:
    code = ex.get('code', '')
    desc = compute_code_descriptors(ex.get('prompt', ''), code, ex.get('test_list', []))
    quality = min(len(code) / 500.0, 1.0) if code else 0.1
    pool.append({'prompt': ex.get('prompt', ''), 'code': code, 'text': ex.get('text', ''),
                 'descriptors': desc, 'quality': quality})

# Selection strategies (same as code_8seed)
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

qd_test = select_qd(pool, 200, 42)
n_qd_cells = len(set(get_cell(x['descriptors']) for x in qd_test))
print(f"QD unique cells: {n_qd_cells}, Pool: {len(pool)}", flush=True)

def fmt(s):
    return f"<|im_start|>system\nComplete the Python function.<|im_end|>\n<|im_start|>user\n{s['prompt'][:512]}<|im_end|>\n<|im_start|>assistant\n{s['code'][:768]}<|im_end|>"

def finetune_full(train_samples, config_name, seed):
    """Full parameter fine-tuning (no LoRA)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from datasets import Dataset

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    output_dir = RESULTS_DIR / f"model_{config_name}_full"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(DEVICE)

    texts = [fmt(s) for s in train_samples]
    ds = Dataset.from_dict({"text": texts})

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=768)

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,  # 40x lower than LoRA (2e-4)
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="no",
        bf16=True,
        report_to="none",
        max_grad_norm=1.0,
        warmup_ratio=0.03,
    )

    class DataCollator:
        def __call__(self, features):
            batch = {
                "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in features]),
                "attention_mask": torch.stack([torch.tensor(f["attention_mask"]) for f in features]),
                "labels": torch.stack([torch.tensor(f["input_ids"]) for f in features]),
            }
            # Mask padding tokens in labels
            batch["labels"][batch["labels"] == tokenizer.pad_token_id] = -100
            return batch

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollator(),
    )
    trainer.train()

    # Save full model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    del model, trainer
    torch.cuda.empty_cache()
    return str(output_dir)

def finetune_lora_control(train_samples, config_name, seed):
    """LoRA fine-tuning (control group, same as code_8seed)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    output_dir = RESULTS_DIR / f"model_{config_name}_lora"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE, trust_remote_code=True
    )
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, task_type="CAUSAL_LM"
    ))

    ds = Dataset.from_dict({"text": [fmt(s) for s in train_samples]})
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(output_dir),
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
            packing=False,
        ),
        train_dataset=ds,
        processing_class=tokenizer
    )
    trainer.train()
    model.save_pretrained(output_dir / "lora")
    tokenizer.save_pretrained(output_dir / "lora")

    del model, trainer
    torch.cuda.empty_cache()
    return str(output_dir / "lora")

def evaluate_humaneval(model_path, seed, is_lora=False):
    """Evaluate on HumanEval pass@1."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_lora:
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16,
            device_map=DEVICE, trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map=DEVICE, trust_remote_code=True
        )
    model.eval()

    correct = total = 0
    for ex in humaneval:
        msgs = [
            {"role": "system", "content": "Complete the Python function. Return only the function body."},
            {"role": "user", "content": ex['prompt']}
        ]
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(txt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=256, temperature=0.0,
                                do_sample=False, pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        try:
            exec_globals = {}
            exec(ex['prompt'] + resp, exec_globals)
            exec(ex['test'], exec_globals)
            correct += 1
        except:
            pass
        total += 1

    del model
    if is_lora:
        del base
    torch.cuda.empty_cache()
    return {"pass_at_1": round(correct/total, 4), "correct": correct, "total": total}

# All configs on single GPU
ALL_CONFIGS = []
for seed in SEEDS:
    for method_name in ["qd_200", "greedy_200", "random_200", "greedy_per_cell"]:
        ALL_CONFIGS.append((method_name, seed))

configs = ALL_CONFIGS  # Run all on this GPU
print(f"Assigned {len(configs)} configs for mode={MODE}", flush=True)

# Run experiments
all_results = {}
for i, (method_name, seed) in enumerate(configs):
    # Data selection (deterministic for all)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    selections = {
        "qd_200": select_qd(pool, 200, seed),
        "greedy_200": select_greedy(pool, 200),
        "random_200": select_random(pool, 200, seed),
        "greedy_per_cell": select_greedy_per_cell(pool, n_qd_cells),
    }
    sel = selections[method_name]
    cells = len(set(get_cell(x['descriptors']) for x in sel))
    cn = f"{method_name}_s{seed}"

    print(f"\n[{i+1}/{len(configs)}] {cn} ({MODE}): n={len(sel)}, cells={cells}", flush=True)
    t0 = time.time()

    try:
        if MODE == "full":
            model_path = finetune_full(sel, cn, seed)
            ev = evaluate_humaneval(model_path, seed, is_lora=False)
        else:
            model_path = finetune_lora_control(sel, cn, seed)
            ev = evaluate_humaneval(model_path, seed, is_lora=True)

        ev["mode"] = MODE
        ev["method"] = method_name
        ev["seed"] = seed
        ev["n_train"] = len(sel)
        ev["n_cells"] = cells
        ev["elapsed"] = round(time.time() - t0, 1)
        all_results[cn] = ev
        print(f"  pass@1: {ev['pass_at_1']*100:.1f}% ({ev['correct']}/{ev['total']}), {ev['elapsed']}s", flush=True)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        all_results[cn] = {"error": str(e), "method": method_name, "seed": seed, "mode": MODE}

    # Save after each config
    with open(OUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

print(f"\n{'='*60}\nRESULTS SUMMARY (GPU {GPU_ID}, mode={MODE})\n{'='*60}", flush=True)
for m in ["qd_200", "greedy_200", "random_200", "greedy_per_cell"]:
    accs = [all_results[k]["pass_at_1"] for k in all_results
            if all_results[k].get("method") == m and "error" not in all_results[k]]
    if accs:
        print(f"{m}: {np.mean(accs)*100:.1f} ± {np.std(accs)*100:.1f}%", flush=True)

print(f"\nSaved to {OUT_FILE}", flush=True)
