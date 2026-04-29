"""
Dialogue Domain Matched Comparison: QD-57 vs Greedy-per-cell (matched cell count)
Addresses SAC: Does QD's diversity advantage hold when Greedy also picks 1 sample per cell?

Also adds: Greedy-26 (26 unique cells from Greedy-57) vs QD-57 (57 unique cells)
This isolates the "cell count matters" hypothesis.
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys, json, random, re, torch, numpy as np, time
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset

MODEL_PATH = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-1___5B-Instruct"
SEEDS = [42, 123, 271, 456, 789, 2024, 314, 159]
GRID_RES = 10
GPU_ID = int(os.environ.get("GPU_ID", "6"))
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
DEVICE = "cuda:0"

RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/dialogue_matched")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"=== Dialogue Matched Comparison (GPU {GPU_ID}) ===", flush=True)

# Load dialogue data
DATA_PATH = "/mnt/data2/zcz/neurIps-emnlp/data/raw/all_dialogues_final.json"
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    dialogues = json.load(f)
print(f"Loaded {len(dialogues)} dialogues", flush=True)

# Strategy keywords
STRATEGY_KEYWORDS = {
    'S1': ['倾听', '理解', '感受', '认同'],
    'S2': ['安慰', '鼓励', '支持', '温暖'],
    'S3': ['建议', '推荐', '方案', '措施'],
    'S4': ['解释', '说明', '澄清', '阐述'],
    'S5': ['引导', '启发', '提问', '探索'],
    'S6': ['分享', '经验', '案例', '经历'],
    'S7': ['确认', '核实', '验证', '确认'],
    'S8': ['道歉', '抱歉', '对不起', '谅解'],
    'S9': ['感谢', '谢谢', '感恩', '感谢'],
    'S10': ['提醒', '注意', '小心', '警告'],
    'S11': ['转移', '转换', '切换', '改变'],
    'S12': ['总结', '归纳', '概括', '回顾'],
    'S13': ['承诺', '保证', '确保', '承诺'],
    'S14': ['协商', '商量', '讨论', '协商'],
    'S15': ['教育', '指导', '培训', '学习'],
    'S16': ['尊重', '重视', '认可', '尊重'],
    'S17': ['关心', '关怀', '关注', '关爱'],
    'S18': ['幽默', '轻松', '诙谐', '调侃'],
}

def detect_strategies(text):
    found = set()
    for sid, kws in STRATEGY_KEYWORDS.items():
        for kw in kws:
            if kw in text:
                found.add(sid)
                break
    return found

def compute_dialogue_descriptors(dialogue):
    text = dialogue.get('dialogue', '') or dialogue.get('text', '')
    all_text = json.dumps(dialogue, ensure_ascii=False) if isinstance(dialogue, dict) else str(dialogue)

    # Empathy: fraction of strategy-bearing turns
    strategies = detect_strategies(all_text)
    n_strategies = len(strategies)

    # Get agent turns
    agent_turns = 0
    if 'conversations' in dialogue:
        for turn in dialogue['conversations']:
            if turn.get('from') == 'gpt' or turn.get('role') == 'assistant':
                agent_turns += 1
    elif 'turns' in dialogue:
        agent_turns = len([t for t in dialogue['turns'] if t.get('speaker') == 'agent'])
    else:
        agent_turns = all_text.count('客服') + all_text.count('Agent')

    agent_turns = max(agent_turns, 1)
    empathy = min(n_strategies / 10.0, 1.0)

    # Strategy type: dominant strategy / 18
    strategy_list = sorted(list(strategies))
    strategy_type = 0
    if strategy_list:
        strategy_type = (sum(ord(c) for c in ''.join(strategy_list)) % 18) / 18.0

    # Conflict intensity
    conflict_words = ['不满', '投诉', '愤怒', '差评', '退款', '投诉', '差评', '骗', '烂', '垃圾']
    conflict_count = sum(1 for cw in conflict_words if cw in all_text)
    conflict = min(conflict_count / 4.0, 1.0)

    return {'empathy': empathy, 'strategy': strategy_type, 'conflict': conflict}

def get_cell(desc):
    return (int(desc['empathy'] * GRID_RES), int(desc['strategy'] * GRID_RES), int(desc['conflict'] * GRID_RES))

def compute_quality(dialogue):
    text = json.dumps(dialogue, ensure_ascii=False) if isinstance(dialogue, dict) else str(dialogue)
    return min(len(text) / 3000.0, 1.0)

# Build pool
pool = []
for d in dialogues:
    desc = compute_dialogue_descriptors(d)
    quality = compute_quality(d)
    pool.append({'dialogue': d, 'descriptors': desc, 'quality': quality})
print(f"Pool: {len(pool)}", flush=True)

# Selection
def select_qd(items, k, seed=42):
    grid = {}
    for item in items:
        cell = get_cell(item['descriptors'])
        if cell not in grid or item['quality'] > grid[cell]['quality']:
            grid[cell] = item
    return sorted(grid.values(), key=lambda x: x['quality'], reverse=True)[:k]

def select_greedy(items, k):
    return sorted(items, key=lambda x: x['quality'], reverse=True)[:k]

def select_greedy_per_cell(items, n_cells):
    sorted_items = sorted(items, key=lambda x: x['quality'], reverse=True)
    seen = set(); sel = []
    for item in sorted_items:
        cell = get_cell(item['descriptors'])
        if cell not in seen:
            seen.add(cell); sel.append(item)
            if len(sel) >= n_cells: break
    return sel

# Get greedy's cell count for matching
greedy_57 = select_greedy(pool, 57)
greedy_cells = set(get_cell(x['descriptors']) for x in greedy_57)
n_greedy_cells = len(greedy_cells)
qd_57 = select_qd(pool, 57, 42)
qd_cells = set(get_cell(x['descriptors']) for x in qd_57)
n_qd_cells = len(qd_cells)
print(f"Greedy-57 cells: {n_greedy_cells}, QD-57 cells: {n_qd_cells}", flush=True)

# Fine-tuning
def finetune(train_samples, config_name, seed):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    output_dir = RESULTS_DIR / f"model_{config_name}"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"], lora_dropout=0.05, task_type="CAUSAL_LM"))

    def fmt(sample):
        d = sample['dialogue']
        text = json.dumps(d, ensure_ascii=False) if isinstance(d, dict) else str(d)
        return f"<|im_start|>system\nYou are a helpful customer service agent.<|im_end|>\n<|im_start|>user\nGenerate an empathetic response.<|im_end|>\n<|im_start|>assistant\n{text[:768]}<|im_end|>"

    ds = Dataset.from_dict({"text": [fmt(s) for s in train_samples]})
    trainer = SFTTrainer(model=model, args=SFTConfig(output_dir=str(output_dir), num_train_epochs=3, per_device_train_batch_size=4, gradient_accumulation_steps=4, learning_rate=2e-4, logging_steps=50, save_strategy="no", bf16=True, report_to="none", max_length=768, dataset_text_field="text", packing=False), train_dataset=ds, processing_class=tokenizer)
    trainer.train()
    model.save_pretrained(output_dir / "lora"); tokenizer.save_pretrained(output_dir / "lora")
    del model, trainer; torch.cuda.empty_cache()
    return output_dir / "lora"

def evaluate(model_path, seed):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, model_path); model.eval()

    # Generate 25 responses per seed
    prompts = [
        "客户说他的订单一直没有收到，非常生气",
        "用户反馈产品质量有问题，要求退货",
        "客户抱怨服务态度不好，想要投诉",
        "用户询问如何使用新购买的产品",
        "客户说他的会员积分没有到账",
        "用户反馈物流太慢了，等了半个月",
        "客户说客服之前承诺的优惠没有兑现",
        "用户对产品使用效果不满意",
        "客户说他的账户被盗了，非常着急",
        "用户询问退款进度，已经等了很久",
        "客户说他的优惠券无法使用",
        "用户反馈收到的是破损商品",
        "客户抱怨等待时间太长，说要给差评",
        "用户询问关于产品的保修政策",
        "客户说他的换货申请一直没有处理",
        "用户对配送时间不满意",
        "客户说之前沟通的问题没有解决",
        "用户反馈产品与描述不符",
        "客户说他已经投诉多次但没人处理",
        "用户询问如何申请售后服务",
        "客户说他的支付出了问题，被扣了两次",
        "用户对客服回复速度不满意",
        "客户说他的预约被取消了",
        "用户反馈APP经常闪退",
        "客户说他要注销账号，非常失望",
    ]

    all_responses = []
    for prompt in prompts[:25]:
        msgs = [{"role":"system","content":"You are a helpful customer service agent. Respond in Chinese."},{"role":"user","content":prompt}]
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(txt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        all_responses.append(resp)

    del model, base; torch.cuda.empty_cache()

    # Compute metrics
    # Self-BLEU
    from collections import Counter
    def char_ngrams(text, n=4):
        return [text[i:i+n] for i in range(max(len(text)-n+1, 0))]

    scores = []
    for i, resp in enumerate(all_responses):
        refs = [all_responses[j] for j in range(len(all_responses)) if j != i]
        if not refs: continue
        ngrams_i = Counter(char_ngrams(resp))
        total = sum(ngrams_i.values())
        if total == 0: continue
        clipped = 0
        for ng, cnt in ngrams_i.items():
            max_ref = max(Counter(char_ngrams(r))[ng] for r in refs)
            clipped += min(cnt, max_ref)
        scores.append(clipped / total)
    self_bleu = np.mean(scores) if scores else 1.0

    # Vocab diversity
    all_tokens = []
    for r in all_responses:
        all_tokens.extend(list(r))
    vocab_div = len(set(all_tokens)) / max(len(all_tokens), 1)

    # Strategy coverage
    all_text = ' '.join(all_responses)
    detected = detect_strategies(all_text)
    strat_cov = len(detected) / 18.0

    # Empathy score
    empathy_words = ['理解', '抱歉', '帮您', '放心', '一定', '尽快', '感谢']
    empathy_count = sum(1 for w in empathy_words if w in all_text)
    empathy = min(empathy_count / (len(all_responses) * 0.5), 1.0)

    return {
        "self_bleu": round(self_bleu, 4),
        "vocab_div": round(vocab_div, 4),
        "strat_cov": round(strat_cov, 4),
        "empathy": round(empathy, 4),
    }

# Run
all_results = {}
for si, seed in enumerate(SEEDS):
    print(f"\n{'='*60}\nSEED {seed} ({si+1}/{len(SEEDS)})\n{'='*60}", flush=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    methods = {
        "qd_57": select_qd(pool, 57, seed),
        "greedy_57": select_greedy(pool, 57),
        "greedy_per_cell": select_greedy_per_cell(pool, n_qd_cells),  # Same cell count as QD
    }

    results = {}
    for name, sel in methods.items():
        cn = f"{name}_s{seed}"
        cells = len(set(get_cell(x['descriptors']) for x in sel))
        print(f"--- {cn}: n={len(sel)}, cells={cells} ---", flush=True)
        t0 = time.time()
        mp = finetune(sel, cn, seed)
        ev = evaluate(mp, seed)
        ev["seed"] = seed; ev["n_train"] = len(sel); ev["n_cells"] = cells
        results[name] = ev
        print(f"  Self-BLEU: {ev['self_bleu']:.4f}, VocabDiv: {ev['vocab_div']:.4f}, StratCov: {ev['strat_cov']:.4f}, {time.time()-t0:.0f}s", flush=True)

    all_results[seed] = results
    with open(RESULTS_DIR / "dialogue_matched_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

# Aggregate
print(f"\n{'='*60}\nAGGREGATE\n{'='*60}", flush=True)
for m in ["qd_57", "greedy_57", "greedy_per_cell"]:
    sb = [all_results[s][m]["self_bleu"] for s in SEEDS]
    vd = [all_results[s][m]["vocab_div"] for s in SEEDS]
    print(f"{m}: Self-BLEU={np.mean(sb):.4f}±{np.std(sb):.4f}, VocabDiv={np.mean(vd):.4f}±{np.std(vd):.4f}", flush=True)

try:
    from scipy.stats import wilcoxon
    qa = [all_results[s]["qd_57"]["self_bleu"] for s in SEEDS]
    ga = [all_results[s]["greedy_57"]["self_bleu"] for s in SEEDS]
    gca = [all_results[s]["greedy_per_cell"]["self_bleu"] for s in SEEDS]

    s1, p1 = wilcoxon(qa, ga)
    print(f"\nQD-57 vs Greedy-57: p={p1:.4f}", flush=True)

    s2, p2 = wilcoxon(qa, gca)
    print(f"QD-57 vs Greedy-per-cell: p={p2:.4f}", flush=True)
except Exception as e:
    print(f"Stats error: {e}", flush=True)

print(f"\nSaved to {RESULTS_DIR / 'dialogue_matched_results.json'}", flush=True)
