"""
Code Per-Round Downstream: Show that redundant data hurts fine-tuning

Uses the Code iterative greedy archive (41 items) to simulate rounds:
- R0: seeds (20 items)
- R1: 32 items
- R2: 36 items
- R3: 38 items
- R4: 39 items
- R5: all 41 items

Fine-tune Qwen2.5-1.5B on each subset, evaluate on HumanEval.
Expected: performance plateaus or decreases after R3 (data becomes redundant).
"""
import os, sys, json, random, re, time, ast, torch, numpy as np
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
GPU_ID = int(os.environ.get("GPU_ID", "2"))
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-1___5B-Instruct"
RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/code_per_round")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda:0"

print(f"=== Code Per-Round Downstream (GPU {GPU_ID}) ===", flush=True)

# Load greedy archive with round-by-round data
# We use the original greedy archive and sort by "quality" to simulate round-by-round accumulation
with open("/mnt/data2/zcz/neurIps-emnlp/neurips/results/code_iterative/greedy_archive.json") as f:
    archive = json.load(f)
print(f"Archive: {len(archive)} items", flush=True)

# Load HumanEval
print("Loading HumanEval...", flush=True)
humaneval = load_dataset("openai_humaneval", split="test")
print(f"HumanEval: {len(humaneval)}", flush=True)

# Sort by quality (descending) - this simulates greedy selection order
archive_sorted = sorted(archive, key=lambda x: x.get('quality', 0), reverse=True)

# Define round breakpoints (matching the iterative experiment: R0=20, R1=32, R2=36, R3=38, R4=39, R5=41)
rounds_config = [
    ("R0", 20),
    ("R1", 32),
    ("R2", 36),
    ("R3", 38),
    ("R4", 39),
    ("R5", 41),
]

def fmt_code(sample):
    return f"<|im_start|>system\nComplete the Python function.<|im_end|>\n<|im_start|>user\n{sample.get('prompt', '')[:512]}<|im_end|>\n<|im_start|>assistant\n{sample.get('code', '')[:768]}<|im_end|>"

def finetune_and_eval(train_data, config_name):
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    output_dir = RESULTS_DIR / f"model_{config_name}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"], lora_dropout=0.05, task_type="CAUSAL_LM"))

    valid = [s for s in train_data if s.get('code') and len(s['code']) > 20]
    texts = [fmt_code(s) for s in valid]
    print(f"  [{config_name}] Training on {len(texts)} samples", flush=True)

    if len(texts) < 5:
        del model; torch.cuda.empty_cache()
        return {"pass_at_1": 0, "n_train": len(texts)}

    ds = Dataset.from_dict({"text": texts})
    trainer = SFTTrainer(model=model, args=SFTConfig(
        output_dir=str(output_dir), num_train_epochs=3,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=2e-4, logging_steps=50, save_strategy="no",
        bf16=True, report_to="none", max_length=768,
        dataset_text_field="text", packing=False),
        train_dataset=ds, processing_class=tokenizer)
    trainer.train()

    model.eval()
    correct = total = 0
    for ex in humaneval:
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

    del model; torch.cuda.empty_cache()
    result = {"pass_at_1": round(correct/total, 4), "correct": correct, "total": total, "n_train": len(texts)}
    print(f"  [{config_name}] pass@1={result['pass_at_1']} ({correct}/{total})", flush=True)
    return result

# Run per-round evaluation
all_results = {}

for round_name, n_items in rounds_config:
    subset = archive_sorted[:n_items]
    print(f"\n--- {round_name}: {len(subset)} items ---", flush=True)
    result = finetune_and_eval(subset, f"greedy_{round_name}")
    result['round'] = round_name
    result['n_items'] = n_items
    all_results[round_name] = result

    # Save intermediate
    with open(RESULTS_DIR / "code_per_round_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

# Summary
print(f"\n{'='*60}", flush=True)
print("CODE PER-ROUND RESULTS:", flush=True)
for rn, r in all_results.items():
    print(f"  {rn}: pass@1={r['pass_at_1']}, n_train={r['n_train']}", flush=True)
print(f"{'='*60}", flush=True)
