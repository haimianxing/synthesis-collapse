"""
Math Iterative Downstream Fine-tuning
Uses iterative archives to show: collapsed data (Greedy) → worse downstream performance
                                    diverse data (QD) → better downstream performance

Key hypothesis: Greedy's frozen 22 cells (R3-R5) produce worse fine-tuning than QD's 29 cells.
"""
import os, sys, json, random, re, time, torch, numpy as np
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
RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/math_iterative_downstream")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda:0"

print(f"=== Math Iterative Downstream (GPU {GPU_ID}) ===", flush=True)

# Load archives
with open("/mnt/data2/zcz/neurIps-emnlp/neurips/results/math_iterative/greedy_archive.json") as f:
    greedy_archive = json.load(f)
with open("/mnt/data2/zcz/neurIps-emnlp/neurips/results/math_iterative/qd_archive.json") as f:
    qd_archive = json.load(f)

print(f"Greedy archive: {len(greedy_archive)} items", flush=True)
print(f"QD archive: {len(qd_archive)} items", flush=True)

# Load GSM8K test
print("Loading GSM8K...", flush=True)
test_ds = load_dataset("gsm8k", "main", split="test")
print(f"GSM8K test: {len(test_ds)}", flush=True)

def extract_answer(text):
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    return match.group(1).replace(',', '') if match else None

def fmt_math(sample):
    """Format math sample for fine-tuning in Qwen2.5 chat template."""
    question = sample.get('question', '')[:512]
    answer = sample.get('answer', '')[:768]
    return f"<|im_start|>system\nSolve the math problem step by step.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"

def finetune_and_eval(train_data, config_name):
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    output_dir = RESULTS_DIR / f"model_{config_name}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, task_type="CAUSAL_LM"))

    # Format training data
    texts = [fmt_math(s) for s in train_data if len(s.get('answer', '')) > 20]
    print(f"  [{config_name}] Training on {len(texts)} samples", flush=True)

    ds = Dataset.from_dict({"text": texts})
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(output_dir), num_train_epochs=5,
            per_device_train_batch_size=2, gradient_accumulation_steps=8,
            learning_rate=2e-4, logging_steps=10, save_strategy="no",
            bf16=True, report_to="none", max_length=768,
            dataset_text_field="text", packing=False),
        train_dataset=ds, processing_class=tokenizer)
    trainer.train()

    # Evaluate on GSM8K
    model.eval()
    correct = total = 0
    for ex in test_ds:
        msgs = [
            {"role": "system", "content": "Solve the math problem step by step."},
            {"role": "user", "content": ex['question']}
        ]
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(txt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=256, temperature=0.0,
                               do_sample=False, pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)

        # Extract predicted answer
        pred_match = re.search(r'####\s*(-?[\d,]+\.?\d*)', resp)
        if not pred_match:
            # Try to find last number in response
            pred_match = re.search(r'(\d+\.?\d*)\s*$', resp.strip())
        pred = pred_match.group(1).replace(',', '') if pred_match else None

        gold = extract_answer(ex['answer'])
        if pred and gold and pred.strip() == gold.strip():
            correct += 1
        total += 1

        if total % 50 == 0:
            print(f"    {total}/{len(test_ds)}, acc={correct/total:.4f}", flush=True)

    del model; torch.cuda.empty_cache()
    result = {
        "accuracy": round(correct/total, 4),
        "correct": correct, "total": total,
        "n_train": len(texts)
    }
    print(f"  [{config_name}] acc={result['accuracy']} ({correct}/{total}), n_train={len(texts)}", flush=True)
    return result

# Run experiments
results = {}

# 1. Greedy archive → fine-tune
print("\n--- Greedy Archive Fine-tuning ---", flush=True)
greedy_result = finetune_and_eval(greedy_archive, "greedy_iter")
results["greedy_iter"] = greedy_result

# 2. QD archive → fine-tune
print("\n--- QD Archive Fine-tuning ---", flush=True)
qd_result = finetune_and_eval(qd_archive, "qd_iter")
results["qd_iter"] = qd_result

# 3. Combined Greedy+QD (ablation: same total N, both strategies' data)
print("\n--- Combined (Greedy+QD) Fine-tuning ---", flush=True)
combined = greedy_archive[:15] + qd_archive[:14]  # 29 total, matched
combined_result = finetune_and_eval(combined, "combined_iter")
results["combined_iter"] = combined_result

# 4. Random subset of same size (baseline)
random.seed(42)
# Use GSM8K train as pool, sample 22 and 29 random items
train_ds = load_dataset("gsm8k", "main", split="train")
random_22 = random.sample(list(train_ds), 22)
random_29 = random.sample(list(train_ds), 29)

print("\n--- Random 22 Fine-tuning ---", flush=True)
random_22_result = finetune_and_eval(random_22, "random_22")
results["random_22"] = random_22_result

print("\n--- Random 29 Fine-tuning ---", flush=True)
random_29_result = finetune_and_eval(random_29, "random_29")
results["random_29"] = random_29_result

# Save results
with open(RESULTS_DIR / "downstream_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n{'='*60}", flush=True)
print("RESULTS SUMMARY:", flush=True)
for k, v in results.items():
    print(f"  {k}: acc={v['accuracy']}, n_train={v['n_train']}", flush=True)
print(f"{'='*60}", flush=True)
