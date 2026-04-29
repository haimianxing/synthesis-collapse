#!/usr/bin/env python3
"""
Per-difficulty stratified evaluation of QD vs Greedy models on GSM8K.
Classifies problems by answer complexity (proxy for difficulty) and
evaluates per-difficulty accuracy.

This shows whether QD's broader coverage translates to balanced performance
across difficulty levels.

Usage:
  CUDA_VISIBLE_DEVICES=4 python per_difficulty_eval.py
"""
import os, json, re, torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Config ===
MODELS = {
    "base": "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-7B-Instruct",
    "qd_r1": "/mnt/data2/zcz/neurIps-emnlp/neurips/results/self_synthesis_v3_base_reset/qd_s42/merged_qd_s42_r1",
    "qd_r2": "/mnt/data2/zcz/neurIps-emnlp/neurips/results/self_synthesis_v3_base_reset/qd_s42/merged_qd_s42_r2",
    "greedy_r2": "/mnt/data2/zcz/neurIps-emnlp/neurips/results/self_synthesis_v3_base_reset/greedy_s42/merged_greedy_s42_r2",
}
GSM8K_TEST = "/mnt/data2/zcz/neurIps-emnlp/neurips/data/gsm8k_test.json"
OUTPUT_DIR = "/mnt/data2/zcz/neurIps-emnlp/neurips/results/per_difficulty_eval"
MAX_NEW_TOKENS = 512

os.makedirs(OUTPUT_DIR, exist_ok=True)

def classify_difficulty(problem, answer):
    """Classify problem difficulty based on multiple signals."""
    # Signal 1: Number of reasoning steps (lines in answer starting with numbers or *>>
    steps = len([l for l in answer.split('\n') if l.strip() and re.match(r'[\d>*]', l.strip())])

    # Signal 2: Number of unique numbers in the answer (more = harder)
    numbers_in_answer = len(set(re.findall(r'\d+\.?\d*', answer)))

    # Signal 3: Question length (proxy for complexity)
    q_len = len(problem.split())

    # Signal 4: Has multiple operations (×, +, -, ÷, /)
    ops = len(re.findall(r'[×÷+\-*/]', problem))

    # Combined score
    score = 0
    if steps >= 6: score += 2
    elif steps >= 3: score += 1
    if numbers_in_answer >= 5: score += 2
    elif numbers_in_answer >= 3: score += 1
    if q_len >= 40: score += 1
    if ops >= 3: score += 1

    # Map to difficulty level
    if score >= 4: return "hard"
    elif score >= 2: return "medium"
    else: return "easy"

def extract_answer(text):
    """Extract numerical answer from model output."""
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed[-1].strip().replace(',', '')

    answer_patterns = [
        r'(?:the answer is|The answer is|the final answer is)\s*:?\s*\$?([-\d.,]+)',
        r'(?:therefore|Thus|Hence|So)[^.\n]*?([-\d.,]+)',
        r'=\s*\$?([-\d.,]+)\s*$',
    ]
    for pattern in answer_patterns:
        matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
        if matches:
            return matches[-1].strip().replace(',', '')

    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    if numbers:
        return numbers[-1].replace(',', '')
    return ""

def normalize_answer(ans):
    try:
        return str(float(ans.replace(',', '').replace(' ', '')))
    except:
        return ans.strip().lower()

def evaluate_model(model, tokenizer, problems_with_difficulty):
    """Evaluate model on GSM8K with per-difficulty tracking."""
    results = {"easy": [], "medium": [], "hard": []}

    for i, item in enumerate(problems_with_difficulty):
        prompt = f"Solve the following math problem step by step.\n\nProblem: {item['question']}\n\nSolution:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_answer(response)
        truth_raw = item['answer'].split('####')[-1].strip() if '####' in item['answer'] else item['answer']
        truth = extract_answer(truth_raw) if truth_raw else truth_raw

        pred_norm = normalize_answer(pred)
        truth_norm = normalize_answer(truth)
        is_correct = pred_norm == truth_norm and pred_norm != ""

        results[item['difficulty']].append({
            "idx": i,
            "correct": is_correct,
            "pred": pred,
            "truth": truth,
        })

        if (i + 1) % 100 == 0:
            total_correct = sum(1 for d in ["easy","medium","hard"] for r in results[d] if r["correct"])
            print(f"    {i+1}/{len(problems_with_difficulty)}, running acc={total_correct/(i+1):.4f}")

    return results

def main():
    # Load GSM8K test
    with open(GSM8K_TEST) as f:
        gsm8k_data = json.load(f)

    # Use first 500 problems (same as v3 eval)
    gsm8k_data = gsm8k_data[:500]

    # Classify difficulty
    problems_with_difficulty = []
    for item in gsm8k_data:
        diff = classify_difficulty(item['question'], item['answer'])
        problems_with_difficulty.append({
            'question': item['question'],
            'answer': item['answer'],
            'difficulty': diff,
        })

    # Print distribution
    dist = {"easy": 0, "medium": 0, "hard": 0}
    for p in problems_with_difficulty:
        dist[p['difficulty']] += 1
    print(f"GSM8K difficulty distribution: easy={dist['easy']}, medium={dist['medium']}, hard={dist['hard']}")

    all_results = {}

    for model_name, model_path in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        if not os.path.exists(model_path):
            print(f"  SKIP: {model_path} not found")
            continue

        print(f"Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )
        model.eval()

        print(f"Evaluating on {len(problems_with_difficulty)} problems...")
        results = evaluate_model(model, tokenizer, problems_with_difficulty)

        # Summarize
        summary = {}
        for diff in ["easy", "medium", "hard"]:
            items = results[diff]
            correct = sum(1 for r in items if r['correct'])
            total = len(items)
            acc = correct / total if total > 0 else 0
            summary[diff] = {"correct": correct, "total": total, "accuracy": acc}

        # Overall
        total_correct = sum(s["correct"] for s in summary.values())
        total_count = sum(s["total"] for s in summary.values())
        summary["overall"] = {"correct": total_correct, "total": total_count, "accuracy": total_correct / total_count}

        all_results[model_name] = summary

        print(f"\n  Results for {model_name}:")
        print(f"    Easy:   {summary['easy']['accuracy']:.1%} ({summary['easy']['correct']}/{summary['easy']['total']})")
        print(f"    Medium: {summary['medium']['accuracy']:.1%} ({summary['medium']['correct']}/{summary['medium']['total']})")
        print(f"    Hard:   {summary['hard']['accuracy']:.1%} ({summary['hard']['correct']}/{summary['hard']['total']})")
        print(f"    Overall:{summary['overall']['accuracy']:.1%} ({summary['overall']['correct']}/{summary['overall']['total']})")

        # Save
        save_path = os.path.join(OUTPUT_DIR, f"{model_name}_per_difficulty.json")
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Free GPU
        del model
        torch.cuda.empty_cache()
        time.sleep(5)

    # Final comparison table
    print(f"\n{'='*70}")
    print("PER-DIFFICULTY COMPARISON")
    print(f"{'='*70}")
    print(f"{'Model':<15} {'Easy':>10} {'Medium':>10} {'Hard':>10} {'Overall':>10}")
    print("-" * 70)
    for model_name, summary in all_results.items():
        print(f"{model_name:<15} {summary['easy']['accuracy']:>9.1%} {summary['medium']['accuracy']:>9.1%} {summary['hard']['accuracy']:>9.1%} {summary['overall']['accuracy']:>9.1%}")

if __name__ == "__main__":
    main()
