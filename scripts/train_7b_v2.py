"""
7B downstream fine-tuning: Reuse data preparation from exp2.
Train Qwen2.5-7B-Instruct with LoRA on 4 configs.
"""
import sys, os, json, torch, random
import numpy as np
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_PATH = sys.argv[1]
CONFIG_NAME = sys.argv[2]  # greedy_57, qd_57, random_57, full
OUTPUT_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/downstream_7b")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = "/mnt/data2/zcz/neurIps-emnlp/data/raw/all_dialogues_final.json"

# Load and prepare data (same logic as exp2)
with open(DATA_PATH) as f:
    all_raw = json.load(f)

samples = []
for d in all_raw:
    turns = d.get("dialogue", [])
    if not turns:
        continue
    messages = []
    for turn in turns:
        if isinstance(turn, dict):
            speaker = turn.get("speaker", turn.get("role", ""))
            content = turn.get("content", "")
            role = "user" if speaker in ("user", "customer") else "assistant"
            messages.append({"role": role, "content": content})
    if not messages:
        continue
    meta = d.get("metadata", {})
    strategy = meta.get("strategies_needed", ["S1"])[0] if meta.get("strategies_needed") else "S1"
    conflict = meta.get("conflict_level", "中")
    quality = min(sum(len(m["content"]) for m in messages) / 2000.0, 1.0)
    samples.append({"messages": messages, "strategy": strategy, "conflict": conflict, "quality": quality})

# Select data based on config
if CONFIG_NAME == "full":
    selected = samples
elif CONFIG_NAME == "greedy_57":
    selected = sorted(samples, key=lambda x: x["quality"], reverse=True)[:57]
elif CONFIG_NAME == "random_57":
    selected = random.sample(samples, 57)
elif CONFIG_NAME == "qd_57":
    strat_pools = {}
    for s in samples:
        strat_pools.setdefault(s["strategy"], []).append(s)
    qd = []
    per_strat = 3
    for strat, pool in strat_pools.items():
        qd.extend(pool[:per_strat])
    selected = qd[:57]

print(f"Config: {CONFIG_NAME}, {len(selected)} samples")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Format data as chat
def format_sample(s):
    text = tokenizer.apply_chat_template(s["messages"], tokenize=False, add_generation_prompt=False)
    return {"text": text}

dataset_list = [format_sample(s) for s in selected]
dataset = Dataset.from_list(dataset_list)

# Load model with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training
out_path = OUTPUT_DIR / f"model_{CONFIG_NAME}"
training_args = TrainingArguments(
    output_dir=str(out_path),
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

# Formatting function for trl 0.24.0
def formatting_func(examples):
    return examples["text"]

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    formatting_func=formatting_func,
)

print(f"Training {CONFIG_NAME}...")
trainer.train()

# Save
lora_path = out_path / "lora"
model.save_pretrained(str(lora_path))
tokenizer.save_pretrained(str(lora_path))
print(f"Saved to {lora_path}")
