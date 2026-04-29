"""
LLM-as-Judge using DashScope native HTTP API.
Reads pre-generated responses and evaluates with Qwen3.5-122B.
"""
import sys, json, time, requests, numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = "sk-bf04f622fcd94499833c70e98dac0803"
API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
JUDGE_MODEL = "qwen-plus"  # Use qwen-plus (works with native API)
INPUT_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/llm_judge")
OUTPUT_DIR = INPUT_DIR

HEADERS = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}


def call_dashscope(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            data = {
                'model': JUDGE_MODEL,
                'input': {'messages': [{'role': 'user', 'content': prompt}]},
                'parameters': {'temperature': 0.1, 'max_tokens': 300, 'result_format': 'message'}
            }
            r = requests.post(API_URL, headers=HEADERS, json=data, timeout=30)
            if r.status_code == 200:
                resp = r.json()
                # Handle both text and message format
                if 'output' in resp:
                    if 'text' in resp['output']:
                        return resp['output']['text']
                    elif 'choices' in resp['output']:
                        return resp['output']['choices'][0]['message']['content']
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"    Retry {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    return None


def judge_response(prompt_text, response_text):
    """Judge a single response."""
    judge_prompt = f"""请评估以下客服回复的质量。按1-5分制打分。

客户问题：{prompt_text}
客服回复：{response_text}

请从以下维度打分（每项1-5分）：
1. 专业性(professionalism)
2. 共情能力(empathy)
3. 问题解决(problem_solving)
4. 语言质量(language_quality)

请严格用以下JSON格式回复（不要添加其他文字）：
{{"professionalism": X, "empathy": X, "problem_solving": X, "language_quality": X, "overall": X}}"""

    result = call_dashscope(judge_prompt)
    if result:
        try:
            # Extract JSON
            if "{" in result and "}" in result:
                json_str = result[result.index("{"):result.rindex("}")+1]
                return json.loads(json_str)
        except:
            pass
    return None


def judge_diversity_pair(r1, r2):
    """Judge similarity between two responses."""
    prompt = f"""请评估以下两个客服回复的相似度（1-5分）：
回复1：{r1[:300]}
回复2：{r2[:300]}
1分=完全不同，5分=几乎相同。只回复一个数字。"""
    result = call_dashscope(prompt)
    if result:
        digits = ''.join(c for c in result if c.isdigit())
        if digits:
            score = int(digits[0])
            if 1 <= score <= 5:
                return score
    return None


def process_model(model_name):
    """Judge all responses for one model."""
    input_path = INPUT_DIR / f"responses_{model_name}.json"
    if not input_path.exists():
        print(f"  {model_name}: No responses file found")
        return

    with open(input_path) as f:
        responses = json.load(f)

    print(f"\nJudging {model_name} ({len(responses)} responses)...")

    # Judge individual responses (batch with ThreadPoolExecutor)
    all_scores = []
    for i, item in enumerate(responses):
        if i % 5 == 0:
            print(f"  {model_name}: {i}/{len(responses)} done")
        scores = judge_response(item["prompt"], item["response"])
        if scores:
            all_scores.append(scores)
        time.sleep(0.3)  # Rate limit

    # Judge diversity (5 random pairs)
    np.random.seed(42)
    indices = np.random.permutation(len(responses))
    div_scores = []
    for i in range(min(5, len(indices)//2)):
        r1 = responses[indices[2*i]]["response"]
        r2 = responses[indices[2*i+1]]["response"]
        score = judge_diversity_pair(r1, r2)
        if score is not None:
            div_scores.append(score)
        time.sleep(0.3)

    # Aggregate
    metrics = ["professionalism", "empathy", "problem_solving", "language_quality", "overall"]
    result = {"model": model_name, "n_judged": len(all_scores), "judge_model": JUDGE_MODEL}
    for m in metrics:
        vals = [s[m] for s in all_scores if m in s and isinstance(s[m], (int, float))]
        if vals:
            result[f"{m}_mean"] = round(float(np.mean(vals)), 2)
            result[f"{m}_std"] = round(float(np.std(vals)), 2)

    if div_scores:
        result["diversity_similarity_mean"] = round(float(np.mean(div_scores)), 2)
        result["diversity_similarity_note"] = "lower=more diverse (1-5 scale)"

    out_path = OUTPUT_DIR / f"judge_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n{model_name} Results:")
    print(json.dumps(result, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    models = sys.argv[1:] if len(sys.argv) > 1 else ["greedy_57", "qd_57", "random_57", "full"]
    for model in models:
        process_model(model)
    print("\n=== All done ===")
