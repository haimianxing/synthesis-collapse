"""
Full downstream evaluation with Self-BLEU + Empathy + Strategy Coverage + Vocab Diversity
Parallel evaluation of 4 models on different GPUs.
Usage: CUDA_VISIBLE_DEVICES=X python full_eval_parallel.py [model_name]
"""
import sys, os, json, torch, numpy as np
from pathlib import Path
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Paths
BASE_MODEL = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-1___5B-Instruct/"
RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/downstream")
DATA_PATH = "/mnt/data2/zcz/neurIps-emnlp/data/raw/all_dialogues_final.json"

# Test prompts for generation
TEST_PROMPTS = [
    "作为客服，如何回应一位因产品质量问题而愤怒的客户？请使用道歉和补偿策略。",
    "客户投诉快递延误，已经等了一周，情绪激动。请用倾听和安抚的方式回复。",
    "一位客户对退款流程不满，已经多次联系未解决。请提供解决方案并表达理解。",
    "客户反映收到的商品与描述不符，要求退货但担心运费。请处理这个投诉。",
    "一位VIP客户投诉服务态度差，威胁要在社交媒体曝光。请妥善处理。",
    "客户因系统故障导致订单取消，要求赔偿。请回应并安抚客户情绪。",
    "一位老年客户不会使用线上退换货功能，来电求助。请耐心指导。",
    "客户质疑会员积分规则变更不公平，要求恢复原积分。请解释并安抚。",
    "客户收到破损商品，已经第三次出现类似问题。请提供满意的解决方案。",
    "客户投诉客服之前给的承诺没有兑现，对品牌信任度下降。请修复信任关系。",
    "客户对促销活动的规则有异议，认为存在误导消费。请解释并处理。",
    "一位客户因个人信息泄露而恐慌，要求立即处理。请回应并保证安全。",
    "客户投诉配送员态度恶劣，要求投诉并赔偿。请妥善处理。",
    "客户对产品质量满意但售后服务体验差，建议改进。请回应并表示感谢。",
    "客户因长时间排队等待客服而愤怒，要求优先处理。请安抚并快速解决。",
    "客户投诉广告宣传与实际产品不符，要求三倍赔偿。请依法处理。",
    "一位新客户对产品使用有疑问，需要详细指导。请耐心解答。",
    "客户因多次转接电话而烦躁，希望一次性解决问题。请高效处理。",
    "客户投诉在门店受到歧视性对待，非常愤怒。请认真对待并处理。",
    "客户对积分兑换礼品的质量不满意，要求更换或退款。请协调处理。",
    "客户投诉App频繁闪退影响使用体验，要求技术修复。请记录并反馈。",
    "一位客户因忘记密码无法登录账户，急需查询订单状态。请协助解决。",
    "客户投诉客服热线等待时间过长，多次尝试未果。请致歉并处理。",
    "客户对包装设计不满意，认为不够环保。请回应并提供建议。",
    "客户投诉商品保质期标注不清晰，担心食品安全。请专业回应。",
    "客户因订单金额错误而投诉，要求立即更正。请核查并处理。",
    "客户投诉售后服务响应慢，已等待48小时未收到回复。请跟进处理。",
    "客户对物流信息更新不及时不满，无法追踪包裹。请查询并反馈。",
    "客户投诉优惠券无法使用，怀疑系统故障。请排查并解决。",
    "客户因商品缺货但网站仍显示有货而投诉。请解释并提供替代方案。",
    "客户投诉退换货流程太复杂，要求简化。请理解并优化。",
    "客户对客服人员专业知识不足感到失望。请展现专业能力。",
    "客户投诉虚假宣传，要求合理解释。请诚恳回应。",
    "客户因重复扣款而愤怒，要求立即退款。请紧急处理。",
    "客户投诉客服态度冷漠，沟通体验极差。请温暖回应。",
    "客户对会员等级制度不满，认为不公平。请解释并安抚。",
    "客户投诉电话客服语音菜单太复杂，无法找到人工服务。请直接帮助。",
    "客户因产品召回而担忧，要求详细说明。请专业回应。",
    "客户投诉赠品未随订单寄出，要求补发。请确认并处理。",
    "客户投诉售后维修周期太长，已等待一个月。请跟进并补偿。",
    "客户反映客服多次承诺回电但从未兑现。请打破这个循环。",
    "客户投诉商品描述页面存在错误信息。请核实并更正。",
    "客户对价格调整不满，刚购买就降价了。请处理价保申请。",
    "客户投诉物流配送时商品被雨淋湿。请赔偿并补发。",
    "客户因被误认为是另一个客户而收到错误信息。请纠正并致歉。",
    "客户投诉客服在电话中挂断了通话。请真诚致歉。",
    "客户对退货运费承担有异议，认为应由商家承担。请处理。",
    "客户投诉购买的商品缺少配件，无法使用。请补发配件。",
    "客户投诉预售商品延迟发货，影响使用计划。请解释并补偿。",
    "客户因客服使用不礼貌用语而投诉。请严肃处理并致歉。",
]

N_GENERATE = 50  # Generate 50 samples per model


def compute_self_bleu(texts):
    """Compute pairwise Self-BLEU (average BLEU-2 between all pairs)"""
    from collections import Counter
    import math

    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def bleu2(ref_tokens, hyp_tokens):
        ref_bigrams = Counter(get_ngrams(ref_tokens, 2))
        hyp_bigrams = Counter(get_ngrams(hyp_tokens, 2))
        if not hyp_bigrams:
            return 0.0
        overlap = sum((ref_bigrams & hyp_bigrams).values())
        total = sum(hyp_bigrams.values())
        return overlap / total if total > 0 else 0.0

    # Tokenize simply by character for Chinese
    tokenized = [list(t) for t in texts]
    n = len(tokenized)
    if n < 2:
        return 0.0

    scores = []
    # Sample pairs to keep computation manageable
    import random
    random.seed(42)
    pairs = []
    for i in range(min(n, 30)):
        for j in range(i+1, min(n, 30)):
            pairs.append((i, j))
    if len(pairs) > 500:
        pairs = random.sample(pairs, 500)

    for i, j in pairs:
        scores.append(bleu2(tokenized[i], tokenized[j]))

    return float(np.mean(scores))


def compute_empathy(text):
    """Rule-based empathy scoring (0-1)"""
    empathy_positive = ["理解", "抱歉", "体谅", "关心", "感谢您的反馈",
                         "为您带来不便", "歉意", "真诚地", "希望您能",
                         "全力", "耐心", "重视", "诚恳"]
    empathy_negative = ["不关我事", "你自己看", "无可奉告", "随便",
                         "不知道", "不归我管"]
    pos = sum(1 for w in empathy_positive if w in text)
    neg = sum(1 for w in empathy_negative if w in text)
    score = min(pos / 6.0, 1.0) - 0.2 * neg
    return max(0.0, min(1.0, score))


def detect_strategies(text):
    """Detect conflict resolution strategies in generated text"""
    strategy_keywords = {
        "S1": ["道歉", "对不起", "抱歉", "致歉"],
        "S2": ["解释", "说明", "原因是", "由于"],
        "S3": ["补偿", "赔偿", "退款", "返还"],
        "S4": ["倾听", "了解", "明白您的", "理解您的"],
        "S5": ["安抚", "请您放心", "别担心", "不要着急"],
        "S6": ["建议", "推荐", "可以考虑", "建议您"],
        "S7": ["理解", "共情", "感同身受", "体会"],
        "S8": ["感谢", "谢谢", "感谢您", "多谢"],
        "S9": ["承诺", "保证", "一定会", "确保"],
        "S10": ["转接", "帮您联系", "专员", "相关负责人"],
        "S11": ["记录", "已记录", "登记", "备案"],
        "S12": ["跟进", "跟踪", "持续关注", "后续"],
        "S13": ["协商", "商量", "共同", "达成一致"],
        "S14": ["授权", "特殊批准", "破例", "特批"],
        "S15": ["教育", "指导", "教您", "教程"],
        "S16": ["预防", "避免再次", "改进", "优化"],
        "S17": ["确认", "核实", "查证", "确认一下"],
        "S18": ["升级", "上级", "主管", "经理"],
    }
    found = set()
    for strat, keywords in strategy_keywords.items():
        if any(kw in text for kw in keywords):
            found.add(strat)
    return found


def compute_vocab_diversity(texts):
    """Unique tokens / total tokens"""
    all_chars = []
    for t in texts:
        all_chars.extend(list(t))
    if not all_chars:
        return 0.0
    return len(set(all_chars)) / len(all_chars)


def evaluate_model(model_name, gpu_id):
    """Full evaluation of one model"""
    print(f"\n=== Evaluating {model_name} on GPU {gpu_id} ===")

    device = f"cuda:0"
    adapter_path = RESULTS_DIR / f"model_{model_name}" / "lora"

    if not (adapter_path / "adapter_model.safetensors").exists():
        print(f"ERROR: No adapter found at {adapter_path}")
        return None

    # Load base model + LoRA
    print(f"Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    print(f"Model loaded on {device}")

    # Generate samples
    generated_texts = []
    prompts_to_use = TEST_PROMPTS[:N_GENERATE]

    for i, prompt in enumerate(prompts_to_use):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        generated_texts.append(response)

        if (i+1) % 10 == 0:
            print(f"  {model_name}: Generated {i+1}/{len(prompts_to_use)}")

    # Compute metrics
    print(f"Computing metrics for {model_name}...")

    # 1. Self-BLEU
    self_bleu = compute_self_bleu(generated_texts)

    # 2. Empathy scores
    empathy_scores = [compute_empathy(t) for t in generated_texts]
    avg_empathy = float(np.mean(empathy_scores))

    # 3. Strategy coverage
    all_strategies = set()
    for text in generated_texts:
        all_strategies.update(detect_strategies(text))
    strategy_coverage = len(all_strategies) / 18.0

    # 4. Vocab diversity
    vocab_div = compute_vocab_diversity(generated_texts)

    # 5. Average length
    avg_length = float(np.mean([len(t) for t in generated_texts]))

    # 6. Conflict distribution (rule-based)
    conflict_dist = {"高": 0, "中": 0, "低": 0}
    for t in generated_texts:
        if any(w in t for w in ["愤怒", "曝光", "投诉", "赔偿", "律师"]):
            conflict_dist["高"] += 1
        elif any(w in t for w in ["不满", "不便", "问题", "困扰"]):
            conflict_dist["中"] += 1
        else:
            conflict_dist["低"] += 1

    results = {
        "strategy_coverage": strategy_coverage,
        "strategies_found": sorted(list(all_strategies)),
        "n_strategies": len(all_strategies),
        "self_bleu": round(self_bleu, 4),
        "avg_empathy": round(avg_empathy, 4),
        "vocab_diversity": round(vocab_div, 4),
        "avg_length": round(avg_length, 2),
        "n_generated": len(generated_texts),
        "conflict_dist": conflict_dist,
        "train_data": model_name,
    }

    print(f"\n{model_name} Results:")
    print(f"  Strategy Coverage: {strategy_coverage:.2%} ({len(all_strategies)}/18)")
    print(f"  Self-BLEU: {self_bleu:.4f}")
    print(f"  Avg Empathy: {avg_empathy:.4f}")
    print(f"  Vocab Diversity: {vocab_div:.4f}")
    print(f"  Avg Length: {avg_length:.1f}")

    # Save individual result
    with open(RESULTS_DIR / f"eval_{model_name}.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Clean up
    del model
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python full_eval_parallel.py <model_name>")
        print("Models: greedy_57, qd_57, random_57, full")
        sys.exit(1)

    model_name = sys.argv[1]
    assert model_name in ["greedy_57", "qd_57", "random_57", "full"]

    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    result = evaluate_model(model_name, gpu_id)

    if result:
        print(f"\nSaved results to eval_{model_name}.json")
    else:
        print(f"\nEvaluation failed for {model_name}")
        sys.exit(1)
