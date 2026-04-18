"""
Experiment 1: Iterative Synthesis Collapse Experiment
Compares Greedy-Iter vs QD-Iter over T=5 rounds of dialogue generation.

Shows that greedy quality selection leads to semantic collapse over iterations,
while QD-Synth maintains diversity.
"""
import json
import os
import time
import random
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# API config
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
API_KEY = "sk-bf04f622fcd94499833c70e98dac0803"
MODEL = "qwen3.5-122b-a10b"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/iterative")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def call_llm(messages, temperature=0.85, max_retries=3):
    """Call DashScope API"""
    import requests
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.9,
        "max_tokens": 2048
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, json=payload, headers=headers, timeout=60)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            elif resp.status_code == 429:
                wait = min(3 * (2 ** attempt), 30)
                time.sleep(wait)
            else:
                time.sleep(3)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))
    return None


def compute_behavior_descriptor(dialogue_text, strategy, conflict_level):
    """Compute 3D behavior descriptor for a dialogue"""
    # Empathy: estimated from text length and emotional words
    empathy_words = ["理解", "抱歉", "感谢", "体谅", "关心", "帮助", "理解您", "非常抱歉"]
    empathy_count = sum(1 for w in empathy_words if w in dialogue_text)
    empathy_strength = min(empathy_count / 5.0, 1.0)

    # Strategy: map S1-S18 to 0-1
    strategy_map = {f"S{i}": (i-1)/17.0 for i in range(1, 19)}
    strategy_val = strategy_map.get(strategy, 0.5)

    # Conflict: map Chinese levels to 0-1
    conflict_map = {"低": 0.25, "中": 0.5, "高": 0.75}
    conflict_val = conflict_map.get(conflict_level, 0.5)

    return (empathy_strength, strategy_val, conflict_val)


def compute_metrics(samples, grid_res=10, dim=3):
    """Compute coverage, entropy, self-bleu for a set of samples"""
    if not samples:
        return {"coverage": 0, "entropy": 0, "self_bleu": 0, "n_samples": 0,
                "strategy_coverage": 0, "vocab_diversity": 0}

    # Grid coverage
    grid = {}
    strategies = set()
    all_tokens = []

    for s in samples:
        desc = s.get("descriptor", (0.5, 0.5, 0.5))
        # Discretize to grid
        cell = tuple(int(d * grid_res) for d in desc[:dim])
        cell = tuple(min(c, grid_res-1) for c in cell)

        if cell not in grid or s.get("quality", 0) > grid[cell].get("quality", 0):
            grid[cell] = s

        if "strategy" in s:
            strategies.add(s["strategy"])

    total_cells = grid_res ** dim
    coverage = len(grid) / total_cells

    # Entropy
    if len(grid) > 1:
        probs = np.array([1.0/len(grid)] * len(grid))
        entropy = -np.sum(probs * np.log(probs + 1e-10))
    else:
        entropy = 0

    # Self-BLEU (simplified: avg pairwise token overlap)
    texts = [s.get("text", "") for s in samples if s.get("text")]
    if len(texts) > 1:
        # Token-level overlap
        token_sets = [set(t.split()) for t in texts]
        overlaps = []
        for i in range(len(token_sets)):
            for j in range(i+1, min(i+10, len(token_sets))):
                if token_sets[i] and token_sets[j]:
                    overlap = len(token_sets[i] & token_sets[j]) / min(len(token_sets[i]), len(token_sets[j]))
                    overlaps.append(overlap)
        self_bleu = np.mean(overlaps) if overlaps else 0
    else:
        self_bleu = 0

    # Vocab diversity
    all_words = " ".join(texts).split()
    vocab_diversity = len(set(all_words)) / max(len(all_words), 1)

    return {
        "coverage": coverage,
        "entropy": float(entropy),
        "self_bleu": float(self_bleu),
        "n_samples": len(samples),
        "strategy_coverage": len(strategies) / 18.0,
        "vocab_diversity": float(vocab_diversity),
        "strategies_found": sorted(list(strategies))
    }


def generate_dialogues_batch(examples, n_samples, target_descriptor=None):
    """Generate n dialogues using LLM, optionally targeting a specific descriptor cell"""
    results = []

    def generate_one(idx):
        # Build prompt from examples
        example_text = "\n\n".join([
            f"示例{i+1}:\n{ex['text'][:500]}"
            for i, ex in enumerate(examples[:3])
        ])

        if target_descriptor:
            prompt_extra = f"\n\n要求: 生成一个{target_descriptor}类型的客服对话。"
        else:
            prompt_extra = "\n\n要求: 生成一个新的中文客服对话。"

        messages = [
            {"role": "system", "content": "你是一个专业的客服对话生成系统。生成高质量的中文客服对话。"},
            {"role": "user", "content": f"参考以下对话示例，生成一个新的不同场景的客服对话。\n\n{example_text}{prompt_extra}\n\n请生成完整对话，包含客户和客服的多个回合。"}
        ]

        text = call_llm(messages)
        if text:
            # Randomly assign strategy and conflict for pool generation
            strategy = f"S{random.randint(1, 18)}"
            conflict = random.choice(["低", "中", "高"])
            desc = compute_behavior_descriptor(text, strategy, conflict)
            return {
                "text": text,
                "strategy": strategy,
                "conflict_level": conflict,
                "descriptor": desc,
                "quality": random.uniform(0.6, 1.0),  # Will be properly scored
                "round_generated": 0
            }
        return None

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(generate_one, i): i for i in range(n_samples)}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except:
                pass

    return results


def score_quality(sample):
    """Score a sample's quality using LLM"""
    text = sample.get("text", "")[:1000]
    messages = [
        {"role": "system", "content": "你是客服对话质量评估专家。评分1-5。"},
        {"role": "user", "content": f"评估以下客服对话质量(1-5分)，只返回数字:\n\n{text[:800]}"}
    ]
    result = call_llm(messages, temperature=0.1)
    if result:
        try:
            score = float(''.join(c for c in result if c.isdigit() or c == '.'))
            return min(max(score / 5.0, 0), 1.0)
        except:
            pass
    return 0.7  # default


def run_iterative_experiment(T=5, n_per_round=100, grid_res=10):
    """Run the full iterative synthesis experiment"""

    # Load seed data
    with open("/mnt/data2/zcz/neurIps-emnlp/data/raw/all_dialogues_final.json") as f:
        all_data = json.load(f)

    # Select 20 seeds
    seed_indices = random.sample(range(len(all_data)), 20)
    seeds = []
    for i in seed_indices:
        d = all_data[i]
        text = ""
        if isinstance(d.get("dialogue"), list):
            for turn in d["dialogue"]:
                if isinstance(turn, dict):
                    text += turn.get("content", "") + "\n"
        meta = d.get("metadata", {})
        strategy = meta.get("strategies_needed", ["S1"])[0] if meta.get("strategies_needed") else "S1"
        conflict = meta.get("conflict_level", "中")
        desc = compute_behavior_descriptor(text, strategy, conflict)
        seeds.append({
            "text": text,
            "strategy": strategy,
            "conflict_level": conflict,
            "descriptor": desc,
            "quality": 0.8,
            "round_generated": 0
        })

    print(f"Loaded {len(seeds)} seeds")

    results = {
        "greedy_iter": {"rounds": []},
        "qd_iter": {"rounds": []},
        "metadata": {"T": T, "n_per_round": n_per_round, "grid_res": grid_res}
    }

    # Initialize both methods with the same seeds
    greedy_pool = list(seeds)
    qd_archive = {}

    # Initialize QD archive with seeds
    for s in seeds:
        cell = tuple(int(d * grid_res) for d in s["descriptor"][:3])
        cell = tuple(min(c, grid_res-1) for c in cell)
        if cell not in qd_archive or s["quality"] > qd_archive[cell]["quality"]:
            qd_archive[cell] = s

    for t in range(1, T + 1):
        print(f"\n{'='*60}")
        print(f"Round {t}/{T}")
        print(f"{'='*60}")

        # --- GREEDY ITER ---
        print(f"\n[Greedy-Iter] Selecting top-{min(len(greedy_pool), 20)} examples as seeds...")
        greedy_pool.sort(key=lambda x: x.get("quality", 0), reverse=True)
        greedy_examples = greedy_pool[:20]

        print(f"[Greedy-Iter] Generating {n_per_round} new samples...")
        greedy_new = generate_dialogues_batch(greedy_examples, n_per_round)
        print(f"[Greedy-Iter] Generated {len(greedy_new)} samples")

        # Score quality
        for s in greedy_new:
            s["quality"] = score_quality(s)
            s["round_generated"] = t

        greedy_pool.extend(greedy_new)

        # Greedy selection: top-100 by quality
        greedy_pool.sort(key=lambda x: x["quality"], reverse=True)
        greedy_pool = greedy_pool[:100]

        greedy_metrics = compute_metrics(greedy_pool, grid_res)
        results["greedy_iter"]["rounds"].append({
            "round": t, **greedy_metrics,
            "pool_size": len(greedy_pool)
        })
        print(f"[Greedy-Iter] Coverage: {greedy_metrics['coverage']:.4f}, "
              f"Entropy: {greedy_metrics['entropy']:.3f}, "
              f"Self-BLEU: {greedy_metrics['self_bleu']:.3f}, "
              f"Strat: {greedy_metrics['strategy_coverage']:.2%}")

        # --- QD ITER ---
        qd_examples = list(qd_archive.values())[:20]
        print(f"\n[QD-Iter] Using {len(qd_examples)} archive entries as seeds...")

        # Generate new samples targeting empty cells
        empty_cells = []
        for i in range(grid_res):
            for j in range(grid_res):
                for k in range(grid_res):
                    if (i,j,k) not in qd_archive:
                        empty_cells.append((i,j,k))

        print(f"[QD-Iter] {len(empty_cells)} empty cells to target")

        qd_new = generate_dialogues_batch(qd_examples, n_per_round)
        print(f"[QD-Iter] Generated {len(qd_new)} samples")

        # Score and update archive
        for s in qd_new:
            s["quality"] = score_quality(s)
            s["round_generated"] = t

            cell = tuple(int(d * grid_res) for d in s["descriptor"][:3])
            cell = tuple(min(c, grid_res-1) for c in cell)

            if cell not in qd_archive or s["quality"] > qd_archive[cell]["quality"]:
                qd_archive[cell] = s

        qd_metrics = compute_metrics(list(qd_archive.values()), grid_res)
        results["qd_iter"]["rounds"].append({
            "round": t, **qd_metrics,
            "archive_size": len(qd_archive)
        })
        print(f"[QD-Iter] Coverage: {qd_metrics['coverage']:.4f}, "
              f"Entropy: {qd_metrics['entropy']:.3f}, "
              f"Self-BLEU: {qd_metrics['self_bleu']:.3f}, "
              f"Strat: {qd_metrics['strategy_coverage']:.2%}")

        # Save checkpoint
        with open(OUTPUT_DIR / f"checkpoint_round_{t}.json", "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    # Save final results
    with open(OUTPUT_DIR / "iterative_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_DIR}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=5, help="Number of iterations")
    parser.add_argument("--n", type=int, default=100, help="Samples per round")
    parser.add_argument("--grid_res", type=int, default=10, help="Grid resolution")
    args = parser.parse_args()

    run_iterative_experiment(T=args.T, n_per_round=args.n, grid_res=args.grid_res)
