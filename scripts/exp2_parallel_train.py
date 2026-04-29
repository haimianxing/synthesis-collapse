"""
Experiment 2: Parallel Downstream Fine-tuning (4 GPUs)
Train Qwen2.5-1.5B with trl SFTTrainer + LoRA on 4 data configs simultaneously.
Usage: python exp2_parallel_train.py
"""
import json
import os
import sys
import torch
import random
import numpy as np
import subprocess
import time
from pathlib import Path
from collections import Counter

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/downstream")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-1___5B-Instruct"
DATA_PATH = "/mnt/data2/zcz/neurIps-emnlp/data/raw/all_dialogues_final.json"

GPU_MAP = {
    "greedy_57": 1,
    "qd_57": 2,
    "random_57": 3,
    "full": 4,
}


def prepare_and_save_datasets():
    """Prepare 4 datasets and save as JSONL for worker scripts"""
    with open(DATA_PATH) as f:
        all_data = json.load(f)

    print(f"Total dialogues: {len(all_data)}")

    samples = []
    for d in all_data:
        text = ""
        if isinstance(d.get("dialogue"), list):
            for turn in d["dialogue"]:
                if isinstance(turn, dict):
                    role = turn.get("role", turn.get("speaker", ""))
                    content = turn.get("content", turn.get("text", ""))
                    text += f"{role}: {content}\n"
        if not text.strip():
            continue
        meta = d.get("metadata", {})
        strategy = meta.get("strategies_needed", ["S1"])[0] if meta.get("strategies_needed") else "S1"
        conflict = meta.get("conflict_level", "中")
        samples.append({"text": text, "strategy": strategy, "conflict": conflict})

    print(f"Valid samples: {len(samples)}")

    # Quality proxy
    for s in samples:
        s["quality"] = min(len(s["text"]) / 2000.0, 1.0)

    # Greedy-57
    sorted_by_quality = sorted(samples, key=lambda x: x["quality"], reverse=True)
    greedy_57 = sorted_by_quality[:57]

    # QD-57: balanced across strategies
    strat_pools = {}
    for s in samples:
        strat_pools.setdefault(s["strategy"], []).append(s)
    qd_57 = []
    for strat, pool in strat_pools.items():
        qd_57.extend(pool[:3])
    qd_57 = qd_57[:57]

    # Random-57
    random_57 = random.sample(samples, min(57, len(samples)))

    datasets = {
        "greedy_57": greedy_57,
        "qd_57": qd_57,
        "random_57": random_57,
        "full": samples,
    }

    # Save each dataset as JSONL
    for name, data in datasets.items():
        out_path = OUTPUT_DIR / f"data_{name}.jsonl"
        with open(out_path, "w") as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        stats = {
            "n_samples": len(data),
            "strategy_coverage": len(set(s["strategy"] for s in data)) / 18.0,
            "conflict_dist": dict(Counter(s["conflict"] for s in data)),
            "avg_quality": float(np.mean([s["quality"] for s in data]))
        }
        print(f"  {name}: {stats}")

    return datasets.keys()


WORKER_SCRIPT = r'''
"""Single GPU training worker - launched per data config"""
import json, os, sys, torch, random, numpy as np
from pathlib import Path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

name = sys.argv[1]
gpu_id = int(sys.argv[2])
model_path = sys.argv[3]
data_dir = sys.argv[4]
output_base = sys.argv[5]

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

print(f"[{name}] Starting on GPU:{gpu_id}", flush=True)

# Load data
samples = []
with open(f"{data_dir}/data_{name}.jsonl") as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line))

print(f"[{name}] Loaded {len(samples)} samples", flush=True)

# Format samples
def format_sample(sample):
    return (
        f"<|im_start|>system\n你是专业的客服人员，需要用合适的策略回应客户。<|im_end|>\n"
        f"<|im_start|>user\n请用{sample['strategy']}策略处理一个{sample['conflict']}冲突级别的客户问题。<|im_end|>\n"
        f"<|im_start|>assistant\n{sample['text'][:512]}<|im_end|>"
    )

train_texts = [format_sample(s) for s in samples]
train_dataset = Dataset.from_dict({"text": train_texts})

# Load model & tokenizer
print(f"[{name}] Loading model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16,
    device_map="cuda", trust_remote_code=True
)

# LoRA
lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

output_dir = f"{output_base}/model_{name}"

# Train
print(f"[{name}] Training...", flush=True)
sft_config = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="no",
    bf16=True,
    report_to="none",
    max_length=512,
    dataset_text_field="text",
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)
trainer.train()

# Save
lora_path = f"{output_dir}/lora"
model.save_pretrained(lora_path)
tokenizer.save_pretrained(lora_path)
print(f"[{name}] Saved to {lora_path}", flush=True)

# Cleanup
del model, trainer
torch.cuda.empty_cache()
print(f"[{name}] Done!", flush=True)
'''

EVAL_SCRIPT = r'''
"""Evaluate all 4 trained models sequentially on one GPU"""
import json, os, sys, torch, random, numpy as np
from pathlib import Path
from collections import Counter

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

gpu_id = sys.argv[1]
model_path = sys.argv[2]
output_base = sys.argv[3]
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

all_strategies = [f"S{i}" for i in range(1, 19)]
eval_prompts = [
    f"<|im_start|>system\n你是专业的客服人员。<|im_end|>\n"
    f"<|im_start|>user\n请用{random.choice(all_strategies)}策略处理一个客户问题。<|im_end|>\n"
    f"<|im_start|>assistant\n"
    for _ in range(50)
]

all_results = {}

for name in ["greedy_57", "qd_57", "random_57", "full"]:
    lora_path = f"{output_base}/model_{name}/lora"
    if not os.path.exists(lora_path):
        print(f"[{name}] No model found at {lora_path}, skipping", flush=True)
        continue

    print(f"\n[{name}] Evaluating...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    generated = []
    for i, prompt in enumerate(eval_prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=512,
                temperature=0.85, top_p=0.9, do_sample=True
            )
        resp = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        generated.append(resp)
        if (i + 1) % 10 == 0:
            print(f"  [{name}] {i+1}/50 done", flush=True)

    del model, base_model
    torch.cuda.empty_cache()

    # Compute metrics
    strategy_keywords = {
        "S1": "道歉", "S2": "解释", "S3": "补偿", "S4": "倾听",
        "S5": "安抚", "S6": "建议", "S7": "关注", "S8": "理解",
        "S9": "感谢", "S10": "承诺", "S11": "共情", "S12": "肯定",
        "S13": "引导", "S14": "鼓励", "S15": "澄清", "S16": "尊重",
        "S17": "关怀", "S18": "专业"
    }
    strategies_found = set()
    for text in generated:
        for strat, kw in strategy_keywords.items():
            if kw in text:
                strategies_found.add(strat)

    conflict_high = sum(1 for t in generated if any(w in t for w in ["不满", "投诉", "愤怒", "差评"]))
    conflict_med = sum(1 for t in generated if any(w in t for w in ["问题", "疑问", "不太满意"]))
    conflict_low = sum(1 for t in generated if any(w in t for w in ["咨询", "了解一下", "请问"]))

    all_tokens = " ".join(generated).split()
    diversity = len(set(all_tokens)) / max(len(all_tokens), 1)

    metrics = {
        "strategy_coverage": len(strategies_found) / 18,
        "strategies_found": sorted(list(strategies_found)),
        "conflict_dist": {"高": conflict_high, "中": conflict_med, "低": conflict_low},
        "vocab_diversity": diversity,
        "n_generated": len(generated),
        "avg_length": float(np.mean([len(t) for t in generated])),
        "train_data": name,
    }

    # Count train samples from JSONL
    data_path = f"{output_base}/data_{name}.jsonl"
    with open(data_path) as f:
        metrics["n_train"] = sum(1 for line in f if line.strip())

    all_results[name] = metrics
    print(f"[{name}] StratCov={metrics['strategy_coverage']:.2%} Div={metrics['vocab_diversity']:.4f} AvgLen={metrics['avg_length']:.0f}", flush=True)

# Save
out_path = f"{output_base}/downstream_results.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
print(f"\nResults saved to {out_path}", flush=True)

# Print table
print(f"\n{'='*60}")
print("COMPARISON TABLE")
print(f"{'='*60}")
print(f"{'Method':<15} {'Strat Cov':>10} {'Vocab Div':>10} {'Avg Len':>10} {'N Train':>10}")
print("-" * 60)
for name in ["greedy_57", "qd_57", "random_57", "full"]:
    if name in all_results:
        r = all_results[name]
        print(f"{name:<15} {r['strategy_coverage']:>10.2%} {r['vocab_diversity']:>10.4f} {r['avg_length']:>10.1f} {r.get('n_train', '?'):>10}")
'''


def main():
    # Step 1: Prepare and save datasets
    print("=" * 60)
    print("Step 1: Preparing datasets")
    print("=" * 60)
    dataset_names = prepare_and_save_datasets()

    # Step 2: Write worker script
    worker_path = OUTPUT_DIR / "worker_train.py"
    with open(worker_path, "w") as f:
        f.write(WORKER_SCRIPT)
    print(f"\nWorker script: {worker_path}")

    # Step 3: Write eval script
    eval_path = OUTPUT_DIR / "worker_eval.py"
    with open(eval_path, "w") as f:
        f.write(EVAL_SCRIPT)
    print(f"Eval script: {eval_path}")

    # Step 4: Launch 4 parallel training processes
    print("\n" + "=" * 60)
    print("Step 2: Launching 4 parallel training processes")
    print("=" * 60)

    processes = {}
    for name in ["greedy_57", "qd_57", "random_57", "full"]:
        gpu_id = GPU_MAP[name]
        cmd = [
            "python3.9", "-u", str(worker_path),
            name, str(gpu_id), MODEL_PATH,
            str(OUTPUT_DIR), str(OUTPUT_DIR)
        ]
        log_file = f"/tmp/exp2_{name}.log"
        print(f"  [{name}] GPU:{gpu_id} -> {log_file}")
        proc = subprocess.Popen(
            cmd, stdout=open(log_file, "w"), stderr=subprocess.STDOUT
        )
        processes[name] = (proc, log_file)
        time.sleep(2)  # Stagger launches to avoid peak memory

    # Step 5: Monitor training
    print("\nMonitoring training progress...")
    while processes:
        done = []
        for name, (proc, log) in processes.items():
            ret = proc.poll()
            if ret is not None:
                if ret == 0:
                    print(f"  [{name}] DONE (exit 0)")
                else:
                    print(f"  [{name}] FAILED (exit {ret})")
                    with open(log) as f:
                        lines = f.readlines()
                    print(f"    Last 5 lines:")
                    for l in lines[-5:]:
                        print(f"      {l.rstrip()}")
                done.append(name)

        for name in done:
            del processes[name]

        if processes:
            time.sleep(10)

    # Step 6: Evaluate all models on GPU 1
    print("\n" + "=" * 60)
    print("Step 3: Evaluating all models")
    print("=" * 60)

    eval_cmd = [
        "python3.9", "-u", str(eval_path),
        "1", MODEL_PATH, str(OUTPUT_DIR)
    ]
    eval_log = "/tmp/exp2_eval.log"
    proc = subprocess.Popen(eval_cmd, stdout=open(eval_log, "w"), stderr=subprocess.STDOUT)
    proc.wait()

    if proc.returncode == 0:
        print("Evaluation complete!")
        # Print results
        with open(OUTPUT_DIR / "downstream_results.json") as f:
            results = json.load(f)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(f"Evaluation failed (exit {proc.returncode})")
        with open(eval_log) as f:
            print(f.read()[-2000:])


if __name__ == "__main__":
    main()
