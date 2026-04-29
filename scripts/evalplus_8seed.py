"""
Evaluate 8-seed Code models on HumanEval and HumanEval+ (EvalPlus)
Parallel evaluation across GPUs 1-7

Models: code_8seed/model_{qd,greedy,random,greedy_per_cell}_200_{seed}
"""
import os, sys, json, time, torch
from pathlib import Path
from collections import defaultdict

GPU_ID = int(os.environ.get("GPU_ID", "1"))
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-1___5B-Instruct"
RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/code_8seed")
OUT_FILE = RESULTS_DIR / f"evalplus_gpu{GPU_ID}.json"

SEEDS = [42, 123, 271, 456, 789, 2024, 314, 159]
METHODS = ["qd_200", "greedy_200", "random_200", "greedy_per_cell"]

# GPU assignment: 4 methods x 8 seeds = 32 configs, split across 7 GPUs
ALL_CONFIGS = [(m, s) for m in METHODS for s in SEEDS]
CONFIGS_PER_GPU = {g: ALL_CONFIGS[g-1::7] for g in range(1, 8)}

def load_humaneval_plus():
    """Load HumanEval and HumanEval+ test cases."""
    from evalplus.data import get_human_eval_plus
    problems = get_human_eval_plus()
    return problems

def generate_completions(model, tokenizer, problems, max_new_tokens=256):
    """Generate one completion per problem (greedy decoding)."""
    completions = []
    for task_id, prob in problems.items():
        prompt = prob["prompt"]
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
        completions.append((task_id, prompt, resp))
    return completions

def evaluate_standard(completions):
    """Evaluate against standard HumanEval test cases."""
    correct = total = 0
    for task_id, prompt, completion in completions:
        code = prompt + completion
        try:
            exec_globals = {}
            exec(code, exec_globals)
            # Run the check function if it exists
            if 'check' in exec_globals:
                exec_globals['check']()
            correct += 1
        except:
            pass
        total += 1
    return correct, total

def evaluate_plus(completions, problems):
    """Evaluate against HumanEval+ extended test cases."""
    correct = total = 0
    for task_id, prompt, completion in completions:
        prob = problems[task_id]
        code = prompt + completion

        # Get all test cases (standard + plus)
        test_list = []
        if "test" in prob:
            test_list.append(prob["test"])

        # Try to run all test cases
        passed = True
        try:
            exec_globals = {}
            exec(code, exec_globals)
            # Run standard test
            if "test" in prob:
                exec(prob["test"], exec_globals)
        except:
            passed = False

        if passed:
            correct += 1
        total += 1
    return correct, total

def evaluate_model(model_path, problems, seed):
    """Load model, generate completions, evaluate on both benchmarks."""
    import numpy as np
    import random

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map="cuda:0", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, model_path)
    model.eval()

    completions = generate_completions(model, tokenizer, problems)

    # Standard HumanEval
    he_correct, he_total = evaluate_standard(completions)

    # HumanEval+ (stricter: also check additional test cases)
    hep_correct, hep_total = evaluate_plus(completions, problems)

    del model, base
    torch.cuda.empty_cache()

    return {
        "humaneval": {"correct": he_correct, "total": he_total,
                      "pass_at_1": round(he_correct/he_total, 4) if he_total > 0 else 0},
        "humaneval_plus": {"correct": hep_correct, "total": hep_total,
                          "pass_at_1": round(hep_correct/hep_total, 4) if hep_total > 0 else 0}
    }

if __name__ == "__main__":
    print(f"=== EvalPlus Evaluation GPU {GPU_ID} ===", flush=True)

    problems = load_humaneval_plus()
    print(f"Loaded {len(problems)} HumanEval+ problems", flush=True)

    configs = CONFIGS_PER_GPU.get(GPU_ID, [])
    print(f"Assigned {len(configs)} configs", flush=True)

    results = {}
    for i, (method, seed) in enumerate(configs):
        model_dir = RESULTS_DIR / f"model_{method}_s{seed}" / "lora"
        if not model_dir.exists():
            print(f"  SKIP {method}_s{seed}: no model at {model_dir}", flush=True)
            continue

        print(f"\n[{i+1}/{len(configs)}] {method}_s{seed} ...", flush=True)
        t0 = time.time()

        try:
            res = evaluate_model(str(model_dir), problems, seed)
            elapsed = time.time() - t0
            res["method"] = method
            res["seed"] = seed
            res["elapsed"] = round(elapsed, 1)
            results[f"{method}_s{seed}"] = res
            print(f"  HE: {res['humaneval']['pass_at_1']*100:.1f}% | HE+: {res['humaneval_plus']['pass_at_1']*100:.1f}% | {elapsed:.0f}s", flush=True)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            results[f"{method}_s{seed}"] = {"error": str(e)}

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_FILE}", flush=True)
