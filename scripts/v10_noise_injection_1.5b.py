"""V10 Noise Injection Experiment: 1.5B Model (Cross-Scale Validation)

Same as v10_noise_injection.py but with Qwen2.5-1.5B-Instruct to show
invisible collapse is not model-scale-specific.

Configs (k=500, 5 LoRA seeds each):
  1. random_incorrect: 500 random from incorrect-only pool (0% correct)
  2. qd_correct_random_incorrect: 300 QD correct + 200 random incorrect (60% correct)
  3. stratified_random: 1 random per cell + random fill (mixed correct/incorrect)

Usage:
  python v10_noise_injection_1.5b.py --gpu 1 --config random_incorrect
  python v10_noise_injection_1.5b.py --gpu 2 --config qd_correct_random_incorrect
  python v10_noise_injection_1.5b.py --gpu 3 --config stratified_random
"""
import os
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import sys, json, torch, numpy as np, subprocess, shutil, random, argparse
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# 1.5B model
MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-1___5B-Instruct"
RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/scale_v10")
OUT_DIR = RESULTS_DIR / "noise_injection_1.5b"
OUT_DIR.mkdir(exist_ok=True)
MAX_SEQ_LENGTH = 512
MAX_CODE_TOKENS = 512

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, required=True)
parser.add_argument('--config', type=str, required=True,
                    choices=['random_incorrect', 'qd_correct_random_incorrect', 'stratified_random'])
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

from datasets import load_dataset
humaneval = load_dataset('openai/openai_humaneval', split='test')

# Load pool
pool = json.load(open(RESULTS_DIR / "merged_pool.json"))
for s in pool:
    if isinstance(s['cell'], list):
        s['cell'] = tuple(s['cell'])

correct_pool = [s for s in pool if s.get('correct', False)]
incorrect_pool = [s for s in pool if not s.get('correct', False)]

ALL_SEEDS = [42, 123, 456, 789, 1024]

def make_random_incorrect(k=500, seed=42):
    rng = random.Random(seed)
    return rng.sample(incorrect_pool, min(k, len(incorrect_pool)))

def make_qd_correct_random_incorrect(k=500, seed=42):
    cells_dict = {}
    for s in correct_pool:
        key = str(s['cell'])
        if key not in cells_dict:
            cells_dict[key] = []
        cells_dict[key].append(s)
    for key in cells_dict:
        cells_dict[key].sort(key=lambda x: x.get('quality', 0), reverse=True)
    qd_correct = [cells_dict[key][0] for key in cells_dict]
    remaining = 300 - len(qd_correct)
    if remaining > 0:
        fill = []
        for key in cells_dict:
            for s in cells_dict[key][1:]:
                fill.append(s)
        fill.sort(key=lambda x: x.get('quality', 0), reverse=True)
        qd_correct.extend(fill[:remaining])
    rng = random.Random(seed)
    random_incorrect = rng.sample(incorrect_pool, k - len(qd_correct))
    return qd_correct + random_incorrect

def make_stratified_random(k=500, seed=42):
    rng = random.Random(seed)
    cells_dict = {}
    for s in pool:
        key = str(s['cell'])
        if key not in cells_dict:
            cells_dict[key] = []
        cells_dict[key].append(s)
    stratified = []
    for key in cells_dict:
        stratified.append(rng.choice(cells_dict[key]))
    remaining = k - len(stratified)
    if remaining > 0:
        used_indices = set()
        for s in stratified:
            idx = pool.index(s)
            used_indices.add(idx)
        available = [s for i, s in enumerate(pool) if i not in used_indices]
        fill = rng.sample(available, min(remaining, len(available)))
        stratified.extend(fill)
    return stratified[:k]

CONFIG_BUILDERS = {
    'random_incorrect': make_random_incorrect,
    'qd_correct_random_incorrect': make_qd_correct_random_incorrect,
    'stratified_random': make_stratified_random,
}

def execute_code_safely(code_str, test_cases, timeout=5):
    passed = 0
    for test in test_cases:
        try:
            result = subprocess.run(['python3', '-c', code_str + "\n" + test],
                                    timeout=timeout, capture_output=True, text=True)
            if result.returncode == 0: passed += 1
        except: pass
    return passed, len(test_cases)

def train_and_eval_he(selected, config_name, lora_seed):
    config_key = f"{config_name}_s{lora_seed}"

    train_texts = [f"### Problem:\n{s['prompt']}\n\n### Solution:\n{s['code']}" for s in selected]
    if len(train_texts) < 10:
        return None

    n_correct = sum(1 for s in selected if s.get('correct', False))
    n_cells = len(set(str(s['cell']) for s in selected))
    print(f"  Data: {len(selected)} samples, {n_correct} correct ({n_correct/len(selected)*100:.1f}%), {n_cells} cells", flush=True)

    random.seed(lora_seed)
    np.random.seed(lora_seed)
    torch.manual_seed(lora_seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

    class CodeDataset(torch.utils.data.Dataset):
        def __init__(self, texts, tokenizer, max_len=512):
            self.texts = texts; self.tokenizer = tokenizer; self.max_len = max_len
        def __len__(self): return len(self.texts)
        def __getitem__(self, idx):
            enc = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_len,
                                 padding='max_length', return_tensors='pt')
            return {'input_ids': enc['input_ids'].squeeze(),
                    'attention_mask': enc['attention_mask'].squeeze(),
                    'labels': enc['input_ids'].squeeze()}

    dataset = CodeDataset(train_texts, tokenizer, MAX_SEQ_LENGTH)
    ckpt_dir = OUT_DIR / f"ckpt_{config_key}"
    training_args = TrainingArguments(
        output_dir=str(ckpt_dir), num_train_epochs=3,
        per_device_train_batch_size=4, gradient_accumulation_steps=2,
        learning_rate=2e-4, bf16=True, logging_steps=10,
        save_strategy="no", report_to="none", seed=lora_seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset,
                      data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False))
    trainer.train()

    merged = model.merge_and_unload()
    merged_path = str(OUT_DIR / f"merged_{config_key}")
    merged.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)

    del trainer, model, merged; torch.cuda.empty_cache()

    print(f"  Evaluating HumanEval for {config_key}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True)
    eval_model = AutoModelForCausalLM.from_pretrained(
        merged_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    eval_model.eval()

    he_correct = 0; he_total = 0
    for item in humaneval:
        prompt = item['prompt']
        test_code = item['test']
        msgs = [{"role": "system", "content": "Complete the Python function."},
                {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=MAX_SEQ_LENGTH).to(eval_model.device)
        with torch.no_grad():
            outputs = eval_model.generate(**inputs, max_new_tokens=MAX_CODE_TOKENS,
                                           do_sample=False, pad_token_id=tokenizer.eos_token_id)
        completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        full_code = prompt + completion
        passed, _ = execute_code_safely(full_code, [test_code])
        if passed > 0: he_correct += 1
        he_total += 1

    he_acc = he_correct / he_total if he_total > 0 else 0
    print(f"  HE: {he_acc:.4f} ({he_correct}/{he_total})", flush=True)

    result = {
        'config': config_name, 'lora_seed': lora_seed,
        'n_train': len(selected),
        'n_correct': n_correct,
        'pct_correct': n_correct / len(selected) * 100,
        'n_cells': n_cells,
        'humaneval_pass1': he_acc, 'he_correct': he_correct, 'he_total': he_total,
        'model': '1.5B',
    }

    del eval_model; torch.cuda.empty_cache()
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    shutil.rmtree(merged_path, ignore_errors=True)

    return result

# Run all seeds for the given config
builder = CONFIG_BUILDERS[args.config]
all_results = {}
results_file = OUT_DIR / f"results_{args.config}_gpu{args.gpu}.json"

for seed in ALL_SEEDS:
    config_key = f"{args.config}_s{seed}"
    print(f"\n=== {config_key} ===", flush=True)

    if results_file.exists():
        existing = json.load(open(results_file))
        if config_key in existing:
            print(f"  SKIP: already done", flush=True)
            all_results.update(existing)
            continue

    selected = builder(k=500, seed=seed)
    print(f"  Selected {len(selected)} solutions", flush=True)

    result = train_and_eval_he(selected, args.config, seed)
    if result:
        all_results[config_key] = result
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

print(f"\n=== DONE: {len(all_results)} configs evaluated ===", flush=True)
with open(results_file, 'w') as f:
    json.dump(all_results, f, indent=2)
