"""
Head-to-head win-rate comparison using Qwen-Plus as judge.
For each prompt, compare QD-57 response vs Greedy-57 response head-to-head.
"""
import json, time, requests, numpy as np
from pathlib import Path

API_KEY = "sk-bf04f622fcd94499833c70e98dac0803"
API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
JUDGE_MODEL = "qwen-plus"
RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/llm_judge")
OUTPUT_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/llm_judge")

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
                'parameters': {'temperature': 0.1, 'max_tokens': 100, 'result_format': 'message'}
            }
            r = requests.post(API_URL, headers=HEADERS, json=data, timeout=30)
            if r.status_code == 200:
                resp = r.json()
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

def compare_pair(prompt_text, response_a, response_b, swap=True):
    """Compare two responses head-to-head. Randomize order to reduce position bias."""
    if swap:
        # Swap order randomly for fairness
        import random
        if random.random() > 0.5:
            first, second, label_a, label_b = response_a, response_b, "A", "B"
        else:
            first, second, label_a, label_b = response_b, response_a, "B", "A"
    else:
        first, second, label_a, label_b = response_a, response_b, "A", "B"

    judge_prompt = f"""请比较以下两个客服回复的质量。只需要考虑：专业性、共情能力、问题解决效果、语言自然度。

客户问题：{prompt_text}

回复{label_a}：{first[:400]}
回复{label_b}：{second[:400]}

哪个回复更好？只回复一个字母：A或B。不要解释。"""

    result = call_dashscope(judge_prompt)
    if result:
        result = result.strip()
        if 'A' in result and 'B' not in result:
            return "A"
        elif 'B' in result and 'A' not in result:
            return "B"
    return None

def main():
    # Load responses
    with open(RESULTS_DIR / "responses_qd_57.json") as f:
        qd_responses = json.load(f)
    with open(RESULTS_DIR / "responses_greedy_57.json") as f:
        greedy_responses = json.load(f)
    with open(RESULTS_DIR / "responses_random_57.json") as f:
        random_responses = json.load(f)

    n = min(len(qd_responses), len(greedy_responses), len(random_responses))
    print(f"Comparing {n} pairs...")

    # QD vs Greedy
    qd_wins_greedy = 0
    greedy_wins = 0
    ties = 0
    for i in range(n):
        if i % 5 == 0:
            print(f"  QD vs Greedy: {i}/{n}")
        prompt = qd_responses[i]["prompt"]
        result = compare_pair(prompt, qd_responses[i]["response"], greedy_responses[i]["response"])
        if result == "A":
            qd_wins_greedy += 1
        elif result == "B":
            greedy_wins += 1
        else:
            ties += 1
        time.sleep(0.3)

    print(f"\n  QD vs Greedy: QD wins={qd_wins_greedy}, Greedy wins={greedy_wins}, Ties={ties}")

    # QD vs Random
    qd_wins_random = 0
    random_wins = 0
    ties2 = 0
    for i in range(n):
        if i % 5 == 0:
            print(f"  QD vs Random: {i}/{n}")
        prompt = qd_responses[i]["prompt"]
        result = compare_pair(prompt, qd_responses[i]["response"], random_responses[i]["response"])
        if result == "A":
            qd_wins_random += 1
        elif result == "B":
            random_wins += 1
        else:
            ties2 += 1
        time.sleep(0.3)

    print(f"\n  QD vs Random: QD wins={qd_wins_random}, Random wins={random_wins}, Ties={ties2}")

    # Save results
    output = {
        "qd_vs_greedy": {
            "qd_wins": qd_wins_greedy,
            "opponent_wins": greedy_wins,
            "ties": ties,
            "n_comparisons": n,
            "qd_win_rate": round(qd_wins_greedy / max(n, 1), 3)
        },
        "qd_vs_random": {
            "qd_wins": qd_wins_random,
            "opponent_wins": random_wins,
            "ties": ties2,
            "n_comparisons": n,
            "qd_win_rate": round(qd_wins_random / max(n, 1), 3)
        },
        "judge_model": JUDGE_MODEL
    }

    out_path = OUTPUT_DIR / "winrate_comparison.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")
    print(f"QD win rate vs Greedy: {output['qd_vs_greedy']['qd_win_rate']}")
    print(f"QD win rate vs Random: {output['qd_vs_random']['qd_win_rate']}")

if __name__ == "__main__":
    main()
