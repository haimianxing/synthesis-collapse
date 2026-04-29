#!/usr/bin/env python3
"""Quick HumanEval evaluation of base model."""
import os
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-7B-Instruct"

print(f"Loading base model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

# Load HumanEval
humaneval = load_dataset("openai/openai_humaneval", "openai_humaneval", split="test")
print(f"HumanEval: {len(humaneval)} problems")

correct = 0
total = 0

for i, item in enumerate(humaneval):
    prompt = item['prompt']
    # Format for code generation
    msgs = [
        {"role": "system", "content": "You are a helpful coding assistant. Complete the function as requested. Only output the completed function."},
        {"role": "user", "content": prompt}
    ]
    txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(txt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Simple pass@1: check if the response contains the correct test output
    test = item['test']
    entry_point = item['entry_point']

    # Execute to check
    try:
        exec_globals = {}
        full_code = prompt + response.split('\n\ndef ')[0].split('\n\nclass ')[0].split('\n\n# ')[0]
        # Add test
        full_test = full_code + '\n' + test + f'\ncheck({entry_point})'
        exec(full_test, exec_globals)
        correct += 1
    except Exception as e:
        pass  # Failed

    total += 1
    if (i + 1) % 20 == 0:
        print(f"  Eval {i+1}/{total}, pass@1={correct/total:.4f}")

acc = correct / total if total > 0 else 0
print(f"\nBase model HumanEval pass@1: {acc:.4f} ({correct}/{total})")
