"""
Experiment 2: Parallel Downstream Fine-tuning with Unsloth (4 GPUs)
Uses conda env: /home/zcz/miniconda3/envs/unsloth/bin/python
"""
import json, os, sys, torch, random, numpy as np
import subprocess, time
from pathlib import Path
from collections import Counter

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/downstream")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-1___5B-Instruct"
DATA_PATH = "/mnt/data2/zcz/neurIps-emnlp/data/raw/all_dialogues_final.json"
PYTHON = "/home/zcz/miniconda3/envs/unsloth/bin/python"

GPU_MAP = {"greedy_57": 1, "qd_57": 2, "random_57": 3, "full": 4}


def prepare_and_save_datasets():
    with open(DATA_PATH) as f:
        all_data = json.load(f)

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

    for s in samples:
        s["quality"] = min(len(s["text"]) / 2000.0, 1.0)

    sorted_by_quality = sorted(samples, key=lambda x: x["quality"], reverse=True)
    greedy_57 = sorted_by_quality[:57]

    strat_pools = {}
    for s in samples:
        strat_pools.setdefault(s["strategy"], []).append(s)
    qd_57 = []
    for strat, pool in strat_pools.items():
        qd_57.extend(pool[:3])
    qd_57 = qd_57[:57]

    random_57 = random.sample(samples, min(57, len(samples)))

    datasets = {"greedy_57": greedy_57, "qd_57": qd_57, "random_57": random_57, "full": samples}

    for name, data in datasets.items():
        out_path = OUTPUT_DIR / f"data_{name}.jsonl"
        with open(out_path, "w") as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        stats = {
            "n_samples": len(data),
            "strategy_coverage": len(set(s["strategy"] for s in data)) / 18.0,
            "avg_quality": float(np.mean([s["quality"] for s in data]))
        }
        print(f"  {name}: {stats}")

    return datasets.keys()


# Worker script for each GPU
WORKER_SCRIPT = r'''
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
print(f"[{name}] Starting on GPU:{gpu_id}", flush=True)

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# Load data
samples = []
with open(f"{data_dir}/data_{name}.jsonl") as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line))
print(f"[{name}] Loaded {len(samples)} samples", flush=True)

def format_sample(sample):
    return (
        f"<|im_start|>system\n你是专业的客服人员，需要用合适的策略回应客户。<|im_end|>\n"
        f"<|im_start|>user\n请用{sample['strategy']}策略处理一个{sample['conflict']}冲突级别的客户问题。<|im_end|>\n"
        f"<|im_start|>assistant\n{sample['text'][:512]}<|im_end|>"
    )

train_texts = [format_sample(s) for s in samples]
train_dataset = Dataset.from_dict({"text": train_texts})

# Load model with Unsloth (4-bit for speed)
print(f"[{name}] Loading model with Unsloth...", flush=True)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=512,
    load_in_4bit=True,
    dtype=None,  # auto
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

output_dir = f"{output_base}/model_{name}"

# Train
print(f"[{name}] Training with Unsloth SFT...", flush=True)
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
    optim="adamw_8bit",
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

del model, trainer
torch.cuda.empty_cache()
print(f"[{name}] Training DONE!", flush=True)
'''

EVAL_SCRIPT = r'''
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

from unsloth import FastLanguageModel

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
        print(f"[{name}] No model found, skipping", flush=True)
        continue

    print(f"\n[{name}] Evaluating...", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=512,
        load_in_4bit=True,
    )
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, lora_path)
    FastLanguageModel.for_inference(model)

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

    del model
    torch.cuda.empty_cache()

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

    data_path = f"{output_base}/data_{name}.jsonl"
    with open(data_path) as f:
        n_train = sum(1 for line in f if line.strip())

    all_results[name] = {
        "strategy_coverage": len(strategies_found) / 18,
        "strategies_found": sorted(list(strategies_found)),
        "conflict_dist": {"高": conflict_high, "中": conflict_med, "低": conflict_low},
        "vocab_diversity": diversity,
        "n_generated": len(generated),
        "avg_length": float(np.mean([len(t) for t in generated])),
        "train_data": name,
        "n_train": n_train,
    }
    print(f"[{name}] StratCov={len(strategies_found)/18:.2%} Div={diversity:.4f} AvgLen={np.mean([len(t) for t in generated]):.0f}", flush=True)

out_path = f"{output_base}/downstream_results.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
print(f"\nResults saved to {out_path}", flush=True)

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
    print("=" * 60)
    print("Step 1: Preparing datasets")
    print("=" * 60)
    dataset_names = prepare_and_save_datasets()

    # Write worker scripts
    worker_path = OUTPUT_DIR / "worker_train_unsloth.py"
    with open(worker_path, "w") as f:
        f.write(WORKER_SCRIPT)

    eval_path = OUTPUT_DIR / "worker_eval_unsloth.py"
    with open(eval_path, "w") as f:
        f.write(EVAL_SCRIPT)

    # Launch 4 parallel training processes
    print("\n" + "=" * 60)
    print("Step 2: Launching 4 parallel Unsloth training processes")
    print("=" * 60)

    processes = {}
    for name in ["greedy_57", "qd_57", "random_57", "full"]:
        gpu_id = GPU_MAP[name]
        cmd = [PYTHON, "-u", str(worker_path), name, str(gpu_id), MODEL_PATH, str(OUTPUT_DIR), str(OUTPUT_DIR)]
        log_file = f"/tmp/exp2_{name}.log"
        print(f"  [{name}] GPU:{gpu_id} -> {log_file}")
        proc = subprocess.Popen(cmd, stdout=open(log_file, "w"), stderr=subprocess.STDOUT)
        processes[name] = (proc, log_file)
        time.sleep(3)

    # Monitor
    print("\nMonitoring training...")
    while processes:
        done = []
        for name, (proc, log) in processes.items():
            ret = proc.poll()
            if ret is not None:
                if ret == 0:
                    print(f"  [{name}] DONE")
                else:
                    print(f"  [{name}] FAILED (exit {ret})")
                    with open(log) as f:
                        for l in f.readlines()[-5:]:
                            print(f"      {l.rstrip()}")
                done.append(name)
        for name in done:
            del processes[name]
        if processes:
            time.sleep(10)

    # Evaluate
    print("\n" + "=" * 60)
    print("Step 3: Evaluating all models on GPU 1")
    print("=" * 60)
    eval_cmd = [PYTHON, "-u", str(eval_path), "1", MODEL_PATH, str(OUTPUT_DIR)]
    eval_log = "/tmp/exp2_eval.log"
    proc = subprocess.Popen(eval_cmd, stdout=open(eval_log, "w"), stderr=subprocess.STDOUT)
    proc.wait()

    if proc.returncode == 0:
        print("Evaluation complete!")
        with open(OUTPUT_DIR / "downstream_results.json") as f:
            print(f.read())
    else:
        print(f"Evaluation failed (exit {proc.returncode})")
        with open(eval_log) as f:
            print(f.read()[-2000:])


if __name__ == "__main__":
    main()
