#!/usr/bin/env python3
"""
Cross-domain evaluation of QD vs Greedy self-synthesis models.
Evaluates GSM8K-trained models on MATH, SVAMP, ASDiv to test generalization.

Usage:
  CUDA_VISIBLE_DEVICES=4 python cross_domain_eval.py
"""
import os, json, time, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# === Config ===
MODELS = {
    "qd_r1": "/mnt/data2/zcz/neurIps-emnlp/neurips/results/self_synthesis_v3_base_reset/qd_s42/merged_qd_s42_r1",
    "qd_r2": "/mnt/data2/zcz/neurIps-emnlp/neurips/results/self_synthesis_v3_base_reset/qd_s42/merged_qd_s42_r2",
    "greedy_r2": "/mnt/data2/zcz/neurIps-emnlp/neurips/results/self_synthesis_v3_base_reset/greedy_s42/merged_greedy_s42_r2",
}
OUTPUT_DIR = "/mnt/data2/zcz/neurIps-emnlp/neurips/results/cross_domain_eval"
N_TEST = 500  # max examples per benchmark
BATCH_SIZE = 1
MAX_NEW_TOKENS = 512

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model(model_path):
    """Load model and tokenizer."""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer

def extract_answer_gsm8k(text):
    """Extract answer from GSM8K ground truth (#### format)."""
    m = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    return m.group(1).replace(',', '') if m else None

def extract_answer_model(text):
    """Extract answer from model output (flexible matching)."""
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

def check_correct(pred, gold):
    """Check if prediction matches gold answer."""
    if not pred or not gold: return False
    try:
        return abs(float(pred.strip()) - float(gold.strip())) < 1e-6
    except:
        return pred.strip() == gold.strip()

def evaluate_benchmark(model, tokenizer, dataset_name, n_test=500):
    """Evaluate model on a benchmark dataset."""
    print(f"\n  Evaluating {dataset_name}...", flush=True)

    # Load dataset
    if dataset_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        questions = [{"q": item["question"], "a": item["answer"]} for item in ds]
    elif dataset_name == "math":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        questions = [{"q": item["problem"], "a": item["answer"]} for item in ds]
    elif dataset_name == "svamp":
        ds = load_dataset("ChilleD/SVAMP", split="test")
        questions = [{"q": item["Body"] + " " + item["Question"], "a": str(item["Answer"])} for item in ds]
    elif dataset_name == "asdiv":
        ds = load_dataset("EleutherAI/asdiv", split="test")
        questions = []
        for item in ds:
            q_text = item.get("input", "") if isinstance(item.get("input"), str) else ""
            a_text = item.get("output", "") if isinstance(item.get("output"), str) else ""
            if q_text and a_text:
                a_num = a_text.split('\n')[0].strip() if '\n' in a_text else a_text.strip()
                questions.append({"q": q_text, "a": a_num})
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    questions = questions[:n_test]
    print(f"  {len(questions)} questions loaded", flush=True)

    correct = 0
    total = 0
    results = []

    for i, item in enumerate(questions):
        # Use chat template like v3 script
        msgs = [
            {"role": "system", "content": "Solve the math problem step by step. Put your final answer after ####."},
            {"role": "user", "content": item['q']}
        ]
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(txt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_answer_model(response)
        gold = extract_answer_gsm8k(item['a']) if dataset_name == "gsm8k" else extract_answer_model(item['a'])
        if not gold:
            gold = item['a'].strip()

        is_correct = check_correct(pred, gold)
        if is_correct:
            correct += 1
        total += 1

        if (i + 1) % 100 == 0:
            print(f"    Eval {i+1}/{total}, acc={correct/total:.4f}", flush=True)

        results.append({
            "idx": i,
            "question": item['q'][:100],
            "pred": pred,
            "gold": gold,
            "correct": is_correct,
        })

    acc = correct / total if total > 0 else 0
    print(f"  {dataset_name}: acc={acc:.4f} ({correct}/{total})", flush=True)
    return {"accuracy": acc, "correct": correct, "total": total, "results": results}

def main():
    benchmarks = ["gsm8k", "math", "svamp", "asdiv"]
    all_results = {}

    for model_name, model_path in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        if not os.path.exists(model_path):
            print(f"  SKIP: {model_path} not found")
            continue

        model, tokenizer = load_model(model_path)
        model_results = {}

        for bench in benchmarks:
            try:
                result = evaluate_benchmark(model, tokenizer, bench, N_TEST)
                model_results[bench] = result
            except Exception as e:
                print(f"  ERROR on {bench}: {e}")
                model_results[bench] = {"error": str(e)}

        all_results[model_name] = model_results

        # Save intermediate results
        save_path = os.path.join(OUTPUT_DIR, f"{model_name}_cross_domain.json")
        # Remove detailed results for saving (too large)
        save_data = {}
        for bench, data in model_results.items():
            save_data[bench] = {k: v for k, v in data.items() if k != "results"}
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"  Saved to {save_path}")

        # Free GPU memory
        del model
        torch.cuda.empty_cache()
        time.sleep(5)

    # Summary
    print(f"\n{'='*60}")
    print("CROSS-DOMAIN SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'GSM8K':>8} {'MATH':>8} {'SVAMP':>8} {'ASDiv':>8} {'Mean':>8}")
    print("-" * 60)
    for model_name, model_results in all_results.items():
        accs = []
        row = f"{model_name:<15}"
        for bench in benchmarks:
            if bench in model_results and "accuracy" in model_results[bench]:
                acc = model_results[bench]["accuracy"]
                accs.append(acc)
                row += f" {acc:>7.1%}"
            else:
                row += f" {'N/A':>7}"
        if accs:
            row += f" {sum(accs)/len(accs):>7.1%}"
        print(row)

    # Save full summary
    summary_path = os.path.join(OUTPUT_DIR, "cross_domain_summary.json")
    summary = {}
    for model_name, model_results in all_results.items():
        summary[model_name] = {}
        for bench, data in model_results.items():
            summary[model_name][bench] = {k: v for k, v in data.items() if k != "results"}
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {summary_path}")

if __name__ == "__main__":
    main()
