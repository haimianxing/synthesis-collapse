"""V10 Parallel Downstream Eval - Process specific configs on specific GPUs."""
import os, sys, json, re, torch, numpy as np, subprocess, time, shutil, random
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-7B-Instruct"
RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/scale_v10")
MAX_SEQ_LENGTH = 512
MAX_CODE_TOKENS = 512

# Parse args: --configs config1,config2,config3 --output out_file.json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--configs', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

CONFIGS = args.configs.split(',')
OUTPUT_FILE = RESULTS_DIR / args.output

from datasets import load_dataset
humaneval = load_dataset('openai/openai_humaneval', split='test')
mbpp_test = load_dataset('mbpp', 'sanitized', split='test')

# Load selection results and merged pool
sel_results = json.load(open(RESULTS_DIR / "selection_results.json"))
all_solutions = json.load(open(RESULTS_DIR / "merged_pool.json"))
# Normalize cells
for s in all_solutions:
    if isinstance(s['cell'], list):
        s['cell'] = tuple(s['cell'])

def execute_code_safely(code_str, test_cases, timeout=5):
    passed = 0
    for test in test_cases:
        try:
            full_code = code_str + "\n" + test
            result = subprocess.run(['python3', '-c', full_code], timeout=timeout,
                                    capture_output=True, text=True)
            if result.returncode == 0: passed += 1
        except: pass
    return passed, len(test_cases)

downstream_results = {}

for config_key in CONFIGS:
    if config_key not in sel_results:
        print(f"  SKIP {config_key}: not in selection results", flush=True)
        continue
    
    sel = sel_results[config_key]
    print(f"\n=== Training: {config_key} ({sel['n_selected']} samples, {sel['n_cells']} cells) ===", flush=True)
    
    # Prepare training data
    train_texts = []
    for item in sel['selected_data']:
        item_cell = tuple(item['cell']) if isinstance(item['cell'], list) else item['cell']
        matching = [s for s in all_solutions
                    if s['prompt'][:200] == item['prompt'] and s['cell'] == item_cell]
        if matching:
            code = matching[0]['code']
            text = f"### Problem:\n{matching[0]['prompt']}\n\n### Solution:\n{code}"
            train_texts.append(text)
    
    if len(train_texts) < 10:
        print(f"  SKIP: only {len(train_texts)} train texts", flush=True)
        continue
    
    # Train LoRA
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
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
    ckpt_dir = RESULTS_DIR / f"ckpt_{config_key}"
    training_args = TrainingArguments(
        output_dir=str(ckpt_dir), num_train_epochs=3,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        learning_rate=2e-4, bf16=True, logging_steps=10,
        save_strategy="no", report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset,
                      data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False))
    trainer.train()
    
    merged = model.merge_and_unload()
    merged_path = str(RESULTS_DIR / f"merged_{config_key}")
    merged.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    
    del trainer, model, merged; torch.cuda.empty_cache()
    
    # Evaluate HumanEval
    print(f"  Evaluating HumanEval...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True)
    eval_model = AutoModelForCausalLM.from_pretrained(
        merged_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )
    eval_model.eval()
    
    he_correct = 0; he_total = 0
    for idx, item in enumerate(humaneval):
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
        passed, n_tests = execute_code_safely(full_code, [test_code])
        if passed > 0: he_correct += 1
        he_total += 1
    
    he_acc = he_correct / he_total if he_total > 0 else 0
    print(f"  HumanEval: {he_acc:.4f} ({he_correct}/{he_total})", flush=True)
    
    # Evaluate MBPP test
    print(f"  Evaluating MBPP test...", flush=True)
    mbpp_correct = 0; mbpp_total = 0
    for idx, item in enumerate(mbpp_test):
        prompt = item['prompt']
        test_list = item['test_list']
        msgs = [{"role": "system", "content": "Write a Python function to solve the problem."},
                {"role": "user", "content": f"Write a Python function:\n\n{prompt}\n\nProvide the implementation:"}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=MAX_SEQ_LENGTH).to(eval_model.device)
        with torch.no_grad():
            outputs = eval_model.generate(**inputs, max_new_tokens=MAX_CODE_TOKENS,
                                           do_sample=False, pad_token_id=tokenizer.eos_token_id)
        completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        code_blocks = re.findall(r'```python\s*(.*?)```', completion, re.DOTALL)
        code = code_blocks[0] if code_blocks else completion
        passed, n_tests = execute_code_safely(code, test_list)
        if passed == n_tests: mbpp_correct += 1
        mbpp_total += 1
    
    mbpp_acc = mbpp_correct / mbpp_total if mbpp_total > 0 else 0
    print(f"  MBPP: {mbpp_acc:.4f} ({mbpp_correct}/{mbpp_total})", flush=True)
    
    downstream_results[config_key] = {
        'strategy': sel['strategy'], 'k': sel['k'], 'seed': sel['seed'],
        'n_train': sel['n_selected'], 'n_cells': sel['n_cells'],
        'entropy': sel['entropy'],
        'humaneval_pass1': he_acc, 'he_correct': he_correct, 'he_total': he_total,
        'mbpp_acc': mbpp_acc, 'mbpp_correct': mbpp_correct, 'mbpp_total': mbpp_total,
    }
    
    # Save intermediate
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(downstream_results, f, indent=2)
    
    # Cleanup
    del eval_model; torch.cuda.empty_cache()
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    shutil.rmtree(merged_path, ignore_errors=True)
    print(f"  Cleaned up", flush=True)

print(f"\n=== DONE: {len(downstream_results)} configs evaluated ===", flush=True)
