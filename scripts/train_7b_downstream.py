"""
Fine-tune Qwen2.5-7B-Instruct with LoRA on 4 training data configs.
Uses system python3.9 with trl+peft (NOT unsloth).
"""
import sys, os, json, torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# Config
MODEL_PATH = sys.argv[1]  # Will be the 7B model path
DATA_PATH = "/mnt/data2/zcz/neurIps-emnlp/data/raw/all_dialogues_final.json"
OUTPUT_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/downstream_7b")
GPU_ID = int(sys.argv[2])

# Load data
with open(DATA_PATH) as f:
    all_data = json.load(f)

# Quality scoring (same as 1.5B experiments)
def quality_score(dialogue):
    agent_turns = [t for t in dialogue.get("turns", []) if t.get("role") == "agent"]
    if not agent_turns:
        return 0.0
    total_len = sum(len(t.get("content", "")) for t in agent_turns)
    return min(total_len / 500.0, 1.0)

# Prepare training data
def prepare_data(dialogues, max_len=512):
    texts = []
    for d in dialogues:
        turns = d.get("turns", [])
        messages = []
        for t in turns:
            role = "user" if t.get("role") == "customer" else "assistant"
            messages.append({"role": role, "content": t.get("content", "")})
        if messages:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append({"text": text})
    return Dataset.from_list(texts)

# 4 configurations
def get_config():
    config_name = sys.argv[3]  # greedy_57, qd_57, random_57, full

    if config_name == "full":
        selected = all_data
    elif config_name == "greedy_57":
        scored = sorted(all_data, key=quality_score, reverse=True)
        selected = scored[:57]
    elif config_name == "random_57":
        import random
        random.seed(42)
        selected = random.sample(all_data, 57)
    elif config_name == "qd_57":
        # QD selection: balanced across strategies
        from collections import defaultdict
        strategy_keywords = {
            "S1": ["道歉", "对不起", "抱歉"], "S2": ["解释", "说明"], "S3": ["补偿", "赔偿"],
            "S4": ["倾听", "了解", "明白"], "S5": ["安抚", "放心"], "S6": ["建议", "推荐"],
            "S7": ["理解", "共情"], "S8": ["感谢", "谢谢"], "S9": ["承诺", "保证"],
            "S10": ["转接", "专员"], "S11": ["记录", "登记"], "S12": ["跟进", "跟踪"],
            "S13": ["协商", "商量"], "S14": ["特殊", "破例"], "S15": ["指导", "教程"],
            "S16": ["预防", "改进"], "S17": ["确认", "核实"], "S18": ["升级", "主管"],
        }
        def get_strategies(dialogue):
            strategies = set()
            for t in dialogue.get("turns", []):
                if t.get("role") == "agent":
                    for s, kws in strategy_keywords.items():
                        if any(kw in t.get("content", "") for kw in kws):
                            strategies.add(s)
            return strategies

        # Group by strategy, sample balanced
        by_strat = defaultdict(list)
        for d in all_data:
            for s in get_strategies(d):
                by_strat[s].append(d)

        selected = []
        per_strat = max(1, 57 // len(by_strat))
        for s, dialogs in by_strat.items():
            dialogs.sort(key=quality_score, reverse=True)
            selected.extend(dialogs[:per_strat])

        # Deduplicate and take top-57 by quality
        seen = set()
        unique = []
        for d in selected:
            idx = id(d)
            if idx not in seen:
                seen.add(idx)
                unique.append(d)
        unique.sort(key=quality_score, reverse=True)
        selected = unique[:57]
    else:
        raise ValueError(f"Unknown config: {config_name}")

    return config_name, selected

# Quantization for 7B on single A800
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

device = f"cuda:{GPU_ID}"
config_name, selected_data = get_config()
print(f"Config: {config_name}, {len(selected_data)} samples, GPU {GPU_ID}")

# Load tokenizer first (needed for prepare_data)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, trust_remote_code=True, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Prepare dataset
dataset = prepare_data(selected_data)

# Load model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map=device,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# LoRA config (same as 1.5B)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training arguments
output_dir = OUTPUT_DIR / f"model_{config_name}"
training_args = TrainingArguments(
    output_dir=str(output_dir),
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="adamw_8bit",
    gradient_checkpointing=True,
    report_to="none",
)

# Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    args=training_args,
)

print(f"Starting training for {config_name}...")
trainer.train()

# Save LoRA adapter
lora_path = output_dir / "lora"
model.save_pretrained(str(lora_path))
tokenizer.save_pretrained(str(lora_path))
print(f"Saved LoRA adapter to {lora_path}")
