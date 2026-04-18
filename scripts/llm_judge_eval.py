"""
LLM-as-Judge downstream evaluation.
Generates responses from fine-tuned models, then uses Qwen3.5-122B API
to evaluate quality, diversity, and empathy of generated responses.
"""
import sys, os, json, torch, numpy as np, random, time
from pathlib import Path
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from openai import OpenAI

BASE_MODEL = "/home/zcz/.cache/modelscope/hub/Qwen/Qwen2___5-1___5B-Instruct/"
RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/downstream")
OUTPUT_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/llm_judge")

# DashScope API config
API_KEY = "sk-bf04f622fcd94499833c70e98dac0803"
client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible_mode/v1"
)
JUDGE_MODEL = "qwen3.5-122b-a10b"

EVAL_PROMPTS = [
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
    "客户说产品用了一周就坏了，要退货但已过7天无理由期限。请处理。",
    "客户投诉在线客服机器人无法理解问题，要求转人工。请处理。",
    "客户说之前被客服挂断电话，非常愤怒。请安抚并解决。",
    "客户对产品功能有很高期望但实际体验一般。请管理预期。",
    "客户说朋友推荐来购买但自己体验很差，感觉被骗。请挽回信任。",
]


def generate_responses(model_name, gpu_id, seed=42):
    """Generate responses from fine-tuned model."""
    device = f"cuda:{gpu_id}"
    adapter_path = RESULTS_DIR / f"model_{model_name}" / "lora"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    responses = []
    for prompt in EVAL_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True,
                                     temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        responses.append({"prompt": prompt, "response": response})

    del model
    torch.cuda.empty_cache()
    return responses


def judge_single(response_text, prompt_text):
    """Use LLM judge to evaluate a single response."""
    judge_prompt = f"""请评估以下客服回复的质量。按1-5分制打分，给出分数和简短理由。

客户问题：{prompt_text}
客服回复：{response_text}

请从以下维度打分（每项1-5分）：
1. 专业性：回复是否专业、有条理
2. 共情能力：是否理解并回应客户的情感需求
3. 问题解决：是否提供具体的解决方案
4. 语言质量：表达是否流畅、得体

请用以下JSON格式回复：
{{"professionalism": X, "empathy": X, "problem_solving": X, "language_quality": X, "overall": X, "reason": "..."}}"""

    try:
        completion = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.1,
            max_tokens=300,
        )
        text = completion.choices[0].message.content.strip()
        # Extract JSON from response
        if "{" in text and "}" in text:
            json_str = text[text.index("{"):text.rindex("}")+1]
            scores = json.loads(json_str)
            return scores
        return None
    except Exception as e:
        print(f"  Judge API error: {e}")
        return None


def judge_diversity(responses_list):
    """Use LLM to judge diversity of a set of responses."""
    # Pick 5 random pairs and ask about similarity
    random.seed(42)
    pairs = []
    indices = list(range(len(responses_list)))
    random.shuffle(indices)
    for i in range(min(5, len(indices)//2)):
        r1 = responses_list[indices[2*i]]["response"][:200]
        r2 = responses_list[indices[2*i+1]]["response"][:200]
        pairs.append((r1, r2))

    similarity_scores = []
    for r1, r2 in pairs:
        prompt = f"""请评估以下两个客服回复的相似度（1-5分）：
回复1：{r1}
回复2：{r2}
1分=完全不同，5分=几乎相同。只回复数字。"""
        try:
            completion = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10,
            )
            text = completion.choices[0].message.content.strip()
            score = int(''.join(c for c in text if c.isdigit()))
            if 1 <= score <= 5:
                similarity_scores.append(score)
        except:
            pass
        time.sleep(0.5)

    return np.mean(similarity_scores) if similarity_scores else None


if __name__ == "__main__":
    model_name = sys.argv[1]  # greedy_57, qd_57, random_57, full
    gpu_id = int(sys.argv[2])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate responses
    print(f"Generating responses for {model_name} on GPU {gpu_id}...")
    responses = generate_responses(model_name, gpu_id, seed=42)

    # Save raw responses
    raw_path = OUTPUT_DIR / f"responses_{model_name}.json"
    with open(raw_path, "w") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(responses)} responses to {raw_path}")

    # Step 2: LLM-Judge evaluation
    print(f"Running LLM-Judge evaluation for {model_name}...")
    all_scores = []
    for i, item in enumerate(responses):
        print(f"  Judging response {i+1}/{len(responses)}...", end=" ")
        scores = judge_single(item["response"], item["prompt"])
        if scores:
            all_scores.append(scores)
            print(f"overall={scores.get('overall', '?')}")
        else:
            print("FAILED")
        time.sleep(0.3)  # Rate limit

    # Step 3: Diversity judgment
    print("Judging response diversity...")
    div_score = judge_diversity(responses)

    # Aggregate results
    if all_scores:
        metrics = ["professionalism", "empathy", "problem_solving", "language_quality", "overall"]
        result = {"model": model_name, "n_judged": len(all_scores)}
        for m in metrics:
            vals = [s[m] for s in all_scores if m in s and isinstance(s[m], (int, float))]
            if vals:
                result[f"{m}_mean"] = float(np.mean(vals))
                result[f"{m}_std"] = float(np.std(vals))
                result[f"{m}_median"] = float(np.median(vals))

        if div_score is not None:
            result["diversity_similarity"] = float(div_score)  # lower = more diverse

        out_path = OUTPUT_DIR / f"judge_{model_name}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nResults for {model_name}:")
        print(json.dumps(result, indent=2))
        print(f"Saved to {out_path}")
    else:
        print(f"ERROR: No successful judgments for {model_name}")
