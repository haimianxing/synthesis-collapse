"""
Experiment 2: Downstream fine-tuning evaluation
Fine-tune Qwen2.5-1.5B on Greedy vs QD vs Random vs Full dialogue data.
Evaluate on strategy coverage, empathy, conflict balance, diversity.
"""
import json
import os
import sys
import torch
import random
import numpy as np
from pathlib import Path
from collections import Counter

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OUTPUT_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/downstream")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-1___5B-Instruct"
DEVICE = "cuda:0"  # Will use CUDA_VISIBLE_DEVICES


def prepare_datasets():
    """Prepare 4 training datasets from CCSE-CS dialogues"""
    with open("/mnt/data2/zcz/neurIps-emnlp/data/raw/all_dialogues_final.json") as f:
        all_data = json.load(f)

    print(f"Total dialogues: {len(all_data)}")

    # Extract training samples
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

    # Quality scores (use text length as proxy for quality ranking)
    for s in samples:
        s["quality"] = min(len(s["text"]) / 2000.0, 1.0)

    # Greedy-57: top 57 by quality
    sorted_by_quality = sorted(samples, key=lambda x: x["quality"], reverse=True)
    greedy_57 = sorted_by_quality[:57]

    # QD-57: balanced selection across strategies
    strat_pools = {}
    for i, s in enumerate(samples):
        if s["strategy"] not in strat_pools:
            strat_pools[s["strategy"]] = []
        strat_pools[s["strategy"]].append(s)

    qd_57 = []
    per_strat = 3
    for strat, pool in strat_pools.items():
        qd_57.extend(pool[:per_strat])
    qd_57 = qd_57[:57]

    # Random-57
    random_57 = random.sample(samples, min(57, len(samples)))

    # Full (all)
    full = samples

    datasets = {
        "greedy_57": greedy_57,
        "qd_57": qd_57,
        "random_57": random_57,
        "full": full
    }

    for name, data in datasets.items():
        stats = {
            "n_samples": len(data),
            "strategy_coverage": len(set(s["strategy"] for s in data)) / 18.0,
            "conflict_dist": dict(Counter(s["conflict"] for s in data)),
            "avg_quality": np.mean([s["quality"] for s in data])
        }
        print(f"{name}: {stats}")

    return datasets


def finetune_lora(train_data, output_name, epochs=3):
    """Fine-tune Qwen2.5-1.5B with LoRA using trl SFTTrainer"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    print(f"\n{'='*60}")
    print(f"Fine-tuning: {output_name} ({len(train_data)} samples)")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True
    )

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Format training samples
    def format_sample(sample):
        return f"<|im_start|>system\n你是专业的客服人员，需要用合适的策略回应客户。<|im_end|>\n<|im_start|>user\n请用{sample['strategy']}策略处理一个{sample['conflict']}冲突级别的客户问题。<|im_end|>\n<|im_start|>assistant\n{sample['text'][:512]}<|im_end|>"

    train_texts = [format_sample(s) for s in train_data]

    # Create HuggingFace Dataset
    train_dataset = Dataset.from_dict({"text": train_texts})

    output_dir = OUTPUT_DIR / f"model_{output_name}"

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
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

    # Save model
    model.save_pretrained(output_dir / "lora")
    tokenizer.save_pretrained(output_dir / "lora")

    # Clean up GPU memory
    del model, trainer
    torch.cuda.empty_cache()

    return output_dir / "lora"


def evaluate_model(model_path, eval_prompts):
    """Evaluate a fine-tuned model"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    generated_texts = []
    for prompt in eval_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.85,
                top_p=0.9,
                do_sample=True
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        generated_texts.append(response)

    del model
    del base_model
    torch.cuda.empty_cache()

    return generated_texts


def compute_eval_metrics(texts, strategies_expected=18):
    """Compute evaluation metrics for generated texts"""
    # Strategy coverage (simplified: check for strategy keywords)
    strategy_keywords = {
        "S1": "道歉", "S2": "解释", "S3": "补偿", "S4": "倾听",
        "S5": "安抚", "S6": "建议", "S7": "关注", "S8": "理解",
        "S9": "感谢", "S10": "承诺", "S11": "共情", "S12": "肯定",
        "S13": "引导", "S14": "鼓励", "S15": "澄清", "S16": "尊重",
        "S17": "关怀", "S18": "专业"
    }

    strategies_found = set()
    for text in texts:
        for strat, keyword in strategy_keywords.items():
            if keyword in text:
                strategies_found.add(strat)

    # Conflict coverage
    conflict_high = sum(1 for t in texts if any(w in t for w in ["不满", "投诉", "愤怒", "差评"]))
    conflict_med = sum(1 for t in texts if any(w in t for w in ["问题", "疑问", "不太满意"]))
    conflict_low = sum(1 for t in texts if any(w in t for w in ["咨询", "了解一下", "请问"]))

    # Diversity
    unique_tokens = len(set(" ".join(texts).split()))
    total_tokens = len(" ".join(texts).split())
    diversity = unique_tokens / max(total_tokens, 1)

    return {
        "strategy_coverage": len(strategies_found) / strategies_expected,
        "strategies_found": sorted(list(strategies_found)),
        "conflict_dist": {"高": conflict_high, "中": conflict_med, "低": conflict_low},
        "vocab_diversity": diversity,
        "n_generated": len(texts),
        "avg_length": np.mean([len(t) for t in texts])
    }


def main():
    datasets = prepare_datasets()

    # Eval prompts: ask model to generate dialogue for each strategy
    all_strategies = [f"S{i}" for i in range(1, 19)]
    eval_prompts = [
        f"<|im_start|>system\n你是专业的客服人员。<|im_end|>\n<|im_start|>user\n请用{random.choice(all_strategies)}策略处理一个客户问题。<|im_end|>\n<|im_start|>assistant\n"
        for _ in range(50)
    ]

    all_results = {}

    for name, data in datasets.items():
        print(f"\n{'='*60}")
        print(f"Processing: {name}")
        print(f"{'='*60}")

        # Fine-tune
        model_path = finetune_lora(data, name, epochs=3)

        # Evaluate
        generated = evaluate_model(model_path, eval_prompts)
        metrics = compute_eval_metrics(generated)
        metrics["train_data"] = name
        metrics["n_train"] = len(data)
        all_results[name] = metrics

        print(f"Results for {name}:")
        for k, v in metrics.items():
            if k != "strategies_found":
                print(f"  {k}: {v}")

    # Save results
    with open(OUTPUT_DIR / "downstream_results.json", "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_DIR / 'downstream_results.json'}")

    # Print comparison table
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Method':<15} {'Strat Cov':>10} {'Vocab Div':>10} {'Avg Len':>10}")
    print("-" * 50)
    for name in ["greedy_57", "qd_57", "random_57", "full"]:
        r = all_results[name]
        print(f"{name:<15} {r['strategy_coverage']:>10.2%} {r['vocab_diversity']:>10.4f} {r['avg_length']:>10.1f}")


if __name__ == "__main__":
    main()
