"""
P2: Harder Benchmark Evaluation — Find another detectable regime
Re-evaluates existing 32 LoRA models from code_8seed on harder benchmarks.

Benchmarks:
1. HumanEval+ (EvalPlus): Extended test cases, stricter evaluation
2. MBPP pass@1: In-domain evaluation

Key question: Does the coverage gap become detectable on harder benchmarks?
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys, json, random, torch, numpy as np, time
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-1___5B-Instruct"
GPU_ID = int(os.environ.get("GPU_ID", "1"))
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
DEVICE = "cuda:0"

RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/code_8seed")
OUT_FILE = RESULTS_DIR / f"hard_eval_gpu{GPU_ID}.json"

SEEDS = [42, 123, 271, 456, 789, 2024, 314, 159]
METHODS = ["qd_200", "greedy_200", "random_200", "greedy_per_cell"]

print(f"=== Hard Benchmark Evaluation (GPU {GPU_ID}) ===", flush=True)

# Load benchmarks
print("Loading benchmarks...", flush=True)
humaneval = load_dataset("openai_humaneval", split="test")

# Load MBPP for in-domain eval
try:
    mbpp = load_dataset("mbpp", "sanitized", split="test")
except:
    mbpp = load_dataset("mbpp", split="test")
print(f"HumanEval: {len(humaneval)}, MBPP: {len(mbpp)}", flush=True)

# Try to load EvalPlus HumanEval+
try:
    from evalplus.data import get_human_eval_plus
    humaneval_plus = get_human_eval_plus()
    has_evalplus = True
    print(f"HumanEval+ loaded: {len(humaneval_plus)} problems", flush=True)
except ImportError:
    has_evalplus = False
    print("EvalPlus not available, using standard HumanEval only", flush=True)

# All configs on single GPU
ALL_CONFIGS = [(m, s) for m in METHODS for s in SEEDS]
configs = ALL_CONFIGS
print(f"Assigned {len(configs)} configs", flush=True)

def evaluate_code(model, tokenizer, prompt, test_code="", max_new_tokens=256):
    """Generate and evaluate a single code completion."""
    msgs = [
        {"role": "system", "content": "Complete the Python function. Return only the function body."},
        {"role": "user", "content": prompt}
    ]
    txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(txt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new_tokens, temperature=0.0,
                            do_sample=False, pad_token_id=tokenizer.eos_token_id)
    resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)

    code = prompt + "\n" + resp
    passed = False
    try:
        exec_globals = {}
        exec(code, exec_globals)
        if test_code:
            exec(test_code, exec_globals)
        passed = True
    except:
        pass
    return passed

all_results = {}
for i, (method, seed) in enumerate(configs):
    model_dir = RESULTS_DIR / f"model_{method}_s{seed}" / "lora"
    if not model_dir.exists():
        print(f"  SKIP {method}_s{seed}: no model at {model_dir}", flush=True)
        continue

    print(f"\n[{i+1}/{len(configs)}] {method}_s{seed} ...", flush=True)
    t0 = time.time()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(DEVICE)
        model = PeftModel.from_pretrained(base, str(model_dir))
        model.eval()

        # 1. Standard HumanEval
        he_correct = he_total = 0
        he_plus_correct = he_plus_total = 0
        for ex in humaneval:
            passed = evaluate_code(model, tokenizer, ex['prompt'], ex.get('test', ''))
            if passed: he_correct += 1
            he_total += 1

            # HumanEval+ (if available): use same test
            if has_evalplus:
                # EvalPlus has more test cases per problem
                task_id = ex.get('task_id', '')
                if task_id in humaneval_plus:
                    prob_plus = humaneval_plus[task_id]
                    plus_test = prob_plus.get('test', ex.get('test', ''))
                    passed_plus = evaluate_code(model, tokenizer, ex['prompt'], plus_test)
                    if passed_plus: he_plus_correct += 1
                    he_plus_total += 1

        # 2. MBPP
        mbpp_correct = mbpp_total = 0
        for ex in list(mbpp)[:200]:
            prompt = ex.get('text', '') + "\n" + ex.get('prompt', '') if ex.get('prompt') else ex.get('text', '')
            test_code = "\n".join(ex.get('test_list', []))
            passed = evaluate_code(model, tokenizer, prompt, test_code)
            if passed: mbpp_correct += 1
            mbpp_total += 1

        del model, base
        torch.cuda.empty_cache()

        elapsed = round(time.time() - t0, 1)
        result = {
            "method": method,
            "seed": seed,
            "humaneval": {
                "pass_at_1": round(he_correct/he_total, 4) if he_total > 0 else 0,
                "correct": he_correct, "total": he_total
            },
            "mbpp": {
                "pass_at_1": round(mbpp_correct/mbpp_total, 4) if mbpp_total > 0 else 0,
                "correct": mbpp_correct, "total": mbpp_total
            },
            "elapsed": elapsed,
        }
        if has_evalplus and he_plus_total > 0:
            result["humaneval_plus"] = {
                "pass_at_1": round(he_plus_correct/he_plus_total, 4),
                "correct": he_plus_correct, "total": he_plus_total
            }

        all_results[f"{method}_s{seed}"] = result

        he_pct = result["humaneval"]["pass_at_1"] * 100
        mbpp_pct = result["mbpp"]["pass_at_1"] * 100
        hep_pct = result.get("humaneval_plus", {}).get("pass_at_1", 0) * 100
        print(f"  HE: {he_pct:.1f}% | HE+: {hep_pct:.1f}% | MBPP: {mbpp_pct:.1f}% | {elapsed}s", flush=True)

    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        all_results[f"{method}_s{seed}"] = {"error": str(e)}

    with open(OUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

# Aggregate
print(f"\n{'='*60}\nAGGREGATE\n{'='*60}", flush=True)
for m in METHODS:
    print(f"\n{m}:", flush=True)
    for bench in ["humaneval", "mbpp", "humaneval_plus"]:
        accs = [all_results[k][bench]["pass_at_1"]
                for k in all_results
                if k.startswith(m) and "error" not in all_results[k] and bench in all_results[k]]
        if accs:
            print(f"  {bench:15s}: {np.mean(accs)*100:.1f} ± {np.std(accs)*100:.1f}% (n={len(accs)})", flush=True)

# Statistical tests
try:
    from scipy.stats import wilcoxon
    print(f"\n--- Statistical Tests ---", flush=True)
    for bench in ["humaneval", "mbpp", "humaneval_plus"]:
        qa = [all_results[f"qd_200_s{s}"][bench]["pass_at_1"]
              for s in SEEDS if f"qd_200_s{s}" in all_results and "error" not in all_results[f"qd_200_s{s}"] and bench in all_results[f"qd_200_s{s}"]]
        ga = [all_results[f"greedy_200_s{s}"][bench]["pass_at_1"]
              for s in SEEDS if f"greedy_200_s{s}" in all_results and "error" not in all_results[f"greedy_200_s{s}"] and bench in all_results[f"greedy_200_s{s}"]]

        n = min(len(qa), len(ga))
        if n >= 5:
            qa, ga = qa[:n], ga[:n]
            stat, p = wilcoxon(qa, ga)
            d = abs(np.mean(qa) - np.mean(ga)) / (np.std([a-b for a,b in zip(qa, ga)]) + 1e-8)
            print(f"QD vs Greedy ({bench:15s}): p={p:.4f}, d={d:.2f}, QD wins {sum(1 for a,b in zip(qa,ga) if a>b)}/{n}", flush=True)
except Exception as e:
    print(f"Stats error: {e}", flush=True)

print(f"\nSaved to {OUT_FILE}", flush=True)
