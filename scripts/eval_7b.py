"""
Evaluate 7B downstream models. Same metrics as stat_eval.py but for 7B.
"""
import sys, os, json, torch, numpy as np, random
from pathlib import Path
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformers import BitsAndBytesConfig

BASE_MODEL = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-7B-Instruct/"
RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/downstream_7b")

TEST_PROMPTS = [
    "作为客服，如何回应一位因产品质量问题而愤怒的客户？",
    "客户投诉快递延误已经一周，情绪激动。请回应。",
    "一位客户对退款流程不满，多次联系未解决。请处理。",
    "客户反映商品与描述不符，要求退货但担心运费。请处理。",
    "VIP客户投诉服务态度差，威胁在社交媒体曝光。请妥善处理。",
    "客户因系统故障导致订单取消，要求赔偿。请回应。",
    "一位老年客户不会使用线上退换货功能，来电求助。请耐心指导。",
    "客户质疑会员积分规则变更不公平。请解释并安抚。",
    "客户收到破损商品，已经第三次出现类似问题。请提供方案。",
    "客户投诉客服之前承诺没兑现，信任度下降。请修复关系。",
    "客户对促销规则有异议，认为存在误导消费。请解释并处理。",
    "客户因个人信息泄露而恐慌，要求立即处理。请回应。",
    "客户投诉配送员态度恶劣，要求投诉并赔偿。请处理。",
    "客户对产品质量满意但售后服务差，建议改进。请回应。",
    "客户因长时间排队等待客服而愤怒。请安抚并快速解决。",
    "客户投诉广告宣传与实际产品不符，要求三倍赔偿。请依法处理。",
    "一位新客户对产品使用有疑问，需要详细指导。请耐心解答。",
    "客户因多次转接电话而烦躁，希望一次性解决。请高效处理。",
    "客户投诉在门店受到歧视性对待，非常愤怒。请认真处理。",
    "客户对积分兑换礼品质量不满意，要求更换或退款。请协调。",
    "客户投诉App频繁闪退影响使用体验。请记录并反馈。",
    "客户因忘记密码无法登录账户，急需查询订单状态。请协助。",
    "客户投诉客服热线等待时间过长。请致歉并处理。",
    "客户对包装设计不满意，认为不够环保。请回应。",
    "客户投诉商品保质期标注不清晰，担心食品安全。请专业回应。",
]

N_PROMPTS = 25

def evaluate_once(model_name, seed):
    adapter_path = RESULTS_DIR / f"model_{model_name}" / "lora"
    if not adapter_path.exists():
        print(f"  Adapter not found: {adapter_path}")
        return None

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    indices = list(range(N_PROMPTS))
    random.shuffle(indices)
    prompts = [TEST_PROMPTS[i] for i in indices]

    generated = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True,
                                     temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        generated.append(response)

    strategy_keywords = {
        "S1": ["道歉", "对不起", "抱歉"], "S2": ["解释", "说明"], "S3": ["补偿", "赔偿"],
        "S4": ["倾听", "了解", "明白"], "S5": ["安抚", "放心"], "S6": ["建议", "推荐"],
        "S7": ["理解", "共情"], "S8": ["感谢", "谢谢"], "S9": ["承诺", "保证"],
        "S10": ["转接", "专员"], "S11": ["记录", "登记"], "S12": ["跟进", "跟踪"],
        "S13": ["协商", "商量"], "S14": ["特殊", "破例"], "S15": ["指导", "教程"],
        "S16": ["预防", "改进"], "S17": ["确认", "核实"], "S18": ["升级", "主管"],
    }
    empathy_words = ["理解", "抱歉", "感谢", "体谅", "关心"]

    strategies = set()
    for text in generated:
        for strat, kws in strategy_keywords.items():
            if any(kw in text for kw in kws):
                strategies.add(strat)

    empathy_scores = [min(sum(1 for w in empathy_words if w in t) / 5.0, 1.0) for t in generated]

    tokenized = [set(list(t)) for t in generated]
    overlaps = []
    for i in range(min(len(tokenized), 15)):
        for j in range(i+1, min(len(tokenized), 15)):
            if tokenized[i] and tokenized[j]:
                overlaps.append(len(tokenized[i] & tokenized[j]) / min(len(tokenized[i]), len(tokenized[j])))
    self_bleu = float(np.mean(overlaps)) if overlaps else 0.0

    vocab_divs = [len(set(t)) / max(len(t), 1) for t in generated]

    del model
    torch.cuda.empty_cache()

    return {
        "strategy_coverage": len(strategies) / 18.0,
        "n_strategies": len(strategies),
        "avg_empathy": float(np.mean(empathy_scores)),
        "self_bleu": self_bleu,
        "vocab_diversity": float(np.mean(vocab_divs)),
        "avg_length": float(np.mean([len(t) for t in generated])),
    }


if __name__ == "__main__":
    model_name = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    print(f"Evaluating 7B {model_name} with seed={seed}")
    result = evaluate_once(model_name, seed)
    if result:
        result["model"] = model_name
        result["model_size"] = "7B"
        result["seed"] = seed

        out_path = RESULTS_DIR / f"eval_{model_name}_seed{seed}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"StratCov={result['strategy_coverage']:.3f}, Self-BLEU={result['self_bleu']:.3f}, "
              f"Empathy={result['avg_empathy']:.3f}, VocabDiv={result['vocab_diversity']:.3f}")
        print(f"Saved to {out_path}")
