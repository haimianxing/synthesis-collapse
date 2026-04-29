"""
Experiment 7: Optimized Iterative Collapse Experiment
- Uses rule-based quality scoring (no LLM quality API calls)
- Reduced to T=5 rounds, 50 samples per round (not 100)
- 4 parallel API workers
"""
import json, os, sys, time, random, numpy as np
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
API_KEY = "sk-bf04f622fcd94499833c70e98dac0803"
MODEL = "qwen3.5-122b-a10b"

SEED_SIZE = 20
N_PER_ROUND = 50  # Reduced from 100 for speed
N_ROUNDS = 5
GRID_RES = 10

DATA_PATH = "/mnt/data2/zcz/neurIps-emnlp/data/raw/all_dialogues_final.json"
OUTPUT_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/iterative_collapse")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def call_api(messages, temperature=0.85, max_retries=3):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL, "messages": messages, "temperature": temperature,
               "top_p": 0.9, "max_tokens": 1500}
    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, json=payload, headers=headers, timeout=60)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            elif resp.status_code == 429:
                time.sleep(min(3 * (2 ** attempt), 30))
            else:
                time.sleep(3)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))
    return None


def compute_descriptor(text, strategy, conflict):
    empathy_words = ["理解", "抱歉", "感谢", "体谅", "关心", "帮助"]
    empathy = min(sum(1 for w in empathy_words if w in text) / 5.0, 1.0)
    strat_map = {f"S{i}": (i-1)/17.0 for i in range(1, 19)}
    conflict_map = {"低": 0.25, "中": 0.5, "高": 0.75}
    return (empathy, strat_map.get(strategy, 0.5), conflict_map.get(conflict, 0.5))


def rule_quality(text):
    """Rule-based quality scoring (fast, no API)"""
    keywords = ["道歉", "解释", "补偿", "倾听", "安抚", "建议", "理解", "共情", "感谢", "承诺"]
    kw_score = sum(1 for k in keywords if k in text) / len(keywords)
    len_score = min(len(text) / 1500.0, 1.0)
    structure_score = min(text.count("\n") / 5.0, 1.0)
    return 0.4 * kw_score + 0.3 * len_score + 0.3 * structure_score


def detect_strategy(text):
    strategy_keywords = {
        "S1": ["道歉", "对不起", "抱歉"], "S2": ["解释", "说明", "原因"],
        "S3": ["补偿", "赔偿", "退款"], "S4": ["倾听", "了解", "明白"],
        "S5": ["安抚", "放心", "别担心"], "S6": ["建议", "推荐", "可以考虑"],
        "S7": ["理解", "共情", "感同身受"], "S8": ["感谢", "谢谢"],
        "S9": ["承诺", "保证", "确保"], "S10": ["转接", "专员", "负责人"],
        "S11": ["记录", "登记", "备案"], "S12": ["跟进", "跟踪", "后续"],
        "S13": ["协商", "商量", "共同"], "S14": ["特殊", "破例", "特批"],
        "S15": ["指导", "教您", "教程"], "S16": ["预防", "避免", "改进"],
        "S17": ["确认", "核实", "查证"], "S18": ["升级", "上级", "主管"],
    }
    for strat, kws in strategy_keywords.items():
        if any(kw in text for kw in kws):
            return strat
    return f"S{random.randint(1, 18)}"


def detect_conflict(text):
    if any(w in text for w in ["愤怒", "曝光", "律师", "消协", "投诉"]):
        return "高"
    elif any(w in text for w in ["满意", "感谢", "放心"]):
        return "低"
    return "中"


def compute_metrics(samples, grid_res=10):
    if not samples:
        return {"coverage": 0, "entropy": 0, "self_bleu": 0, "strategy_count": 0, "vocab_diversity": 0}

    cells = set()
    cell_counts = Counter()
    strategies = set()
    all_chars = []

    for s in samples:
        desc = s.get("descriptor", (0.5, 0.5, 0.5))
        cell = tuple(int(min(d * grid_res, grid_res-1)) for d in desc[:3])
        cells.add(cell)
        cell_counts[cell] += 1
        strategies.add(s.get("strategy", "S1"))
        all_chars.extend(list(s.get("text", "")))

    coverage = len(cells) / (grid_res ** 3)

    total = sum(cell_counts.values())
    probs = [c/total for c in cell_counts.values()]
    entropy = -sum(p * np.log2(p) for p in probs if p > 0) if len(probs) > 1 else 0

    # Self-BLEU on last batch
    texts = [s.get("text", "") for s in samples[-N_PER_ROUND:] if s.get("text")]
    if len(texts) >= 2:
        tokenized = [set(t) for t in texts]
        overlaps = []
        for i in range(min(len(tokenized), 20)):
            for j in range(i+1, min(len(tokenized), 20)):
                if tokenized[i] and tokenized[j]:
                    overlaps.append(len(tokenized[i] & tokenized[j]) / min(len(tokenized[i]), len(tokenized[j])))
        self_bleu = float(np.mean(overlaps)) if overlaps else 0
    else:
        self_bleu = 0

    vocab_div = len(set(all_chars)) / max(len(all_chars), 1)

    return {
        "coverage": coverage,
        "entropy": float(entropy),
        "self_bleu": self_bleu,
        "strategy_count": len(strategies),
        "strategies": sorted(list(strategies)),
        "vocab_diversity": vocab_div,
        "n_samples": len(samples),
        "n_cells": len(cells),
    }


def generate_batch(examples, n_samples):
    """Generate n samples in parallel"""
    results = []
    example_text = "\n\n".join([f"示例{i+1}:\n{ex['text'][:400]}" for i, ex in enumerate(examples[:5])])

    def gen_one(idx):
        messages = [
            {"role": "system", "content": "你是专业的客服对话生成系统。生成真实、多样化的中文客服对话。"},
            {"role": "user", "content": f"参考以下示例，生成一个完全不同场景的客服对话（第{idx+1}条）。确保使用不同的策略和冲突级别。\n\n{example_text}"}
        ]
        text = call_api(messages)
        if text:
            strategy = detect_strategy(text)
            conflict = detect_conflict(text)
            desc = compute_descriptor(text, strategy, conflict)
            quality = rule_quality(text)
            return {"text": text, "strategy": strategy, "conflict": conflict,
                    "descriptor": desc, "quality": quality}
        return None

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(gen_one, i): i for i in range(n_samples)}
        for future in as_completed(futures):
            try:
                r = future.result()
                if r:
                    results.append(r)
            except:
                pass

    return results


def run_experiment():
    print("Loading seed data...")
    with open(DATA_PATH) as f:
        all_data = json.load(f)

    seeds = []
    indices = random.sample(range(len(all_data)), SEED_SIZE)
    for i in indices:
        d = all_data[i]
        text = ""
        if isinstance(d.get("dialogue"), list):
            for turn in d["dialogue"]:
                if isinstance(turn, dict):
                    text += turn.get("content", turn.get("text", "")) + "\n"
        meta = d.get("metadata", {})
        strategy = meta.get("strategies_needed", ["S1"])[0] if meta.get("strategies_needed") else "S1"
        conflict = meta.get("conflict_level", "中")
        desc = compute_descriptor(text, strategy, conflict)
        seeds.append({"text": text, "strategy": strategy, "conflict": conflict,
                      "descriptor": desc, "quality": rule_quality(text)})

    print(f"Loaded {len(seeds)} seeds")

    # Run both methods
    all_results = {"greedy_iter": [], "qd_iter": [], "metadata": {
        "T": N_ROUNDS, "n_per_round": N_PER_ROUND, "grid_res": GRID_RES}}

    for method in ["greedy", "qd"]:
        print(f"\n{'='*60}")
        print(f"  {method.upper()}-ITER")
        print(f"{'='*60}")

        pool = list(seeds)
        archive = {}

        # Init archive
        for s in pool:
            cell = tuple(int(min(d * GRID_RES, GRID_RES-1)) for d in s["descriptor"][:3])
            if cell not in archive or s["quality"] > archive[cell]["quality"]:
                archive[cell] = s

        # Record Round 0
        metrics = compute_metrics(pool)
        all_results[f"{method}_iter"].append({"round": 0, **metrics})
        print(f"  Round 0: cov={metrics['coverage']:.4f}, ent={metrics['entropy']:.3f}, "
              f"bleu={metrics['self_bleu']:.3f}, strat={metrics['strategy_count']}")

        for t in range(1, N_ROUNDS + 1):
            # Select examples for few-shot
            if method == "greedy":
                examples = sorted(pool, key=lambda x: x["quality"], reverse=True)[:10]
            else:
                examples = list(archive.values())[:10]

            # Generate new samples
            print(f"  Round {t}: Generating {N_PER_ROUND} samples...")
            new_samples = generate_batch(examples, N_PER_ROUND)
            print(f"  Round {t}: Generated {len(new_samples)} samples")

            # Update pool and archive
            pool.extend(new_samples)
            for s in new_samples:
                cell = tuple(int(min(d * GRID_RES, GRID_RES-1)) for d in s["descriptor"][:3])
                if cell not in archive or s["quality"] > archive[cell]["quality"]:
                    archive[cell] = s

            # Trim pool to keep manageable
            if method == "greedy":
                pool = sorted(pool, key=lambda x: x["quality"], reverse=True)[:100]

            metrics = compute_metrics(list(archive.values()) if method == "qd" else pool)
            all_results[f"{method}_iter"].append({"round": t, **metrics})
            print(f"  Round {t}: cov={metrics['coverage']:.4f}, ent={metrics['entropy']:.3f}, "
                  f"bleu={metrics['self_bleu']:.3f}, strat={metrics['strategy_count']}, "
                  f"cells={metrics.get('n_cells', 'N/A')}")

            # Save checkpoint
            with open(OUTPUT_DIR / "collapse_all.json", "w") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Generate figure
    generate_figure(all_results)

    print(f"\nResults saved to {OUTPUT_DIR}")
    return all_results


def generate_figure(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIG_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/figures")

    greedy = results["greedy_iter"]
    qd = results["qd_iter"]
    rounds = [r["round"] for r in greedy]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    metrics = [
        ("coverage", "Grid Coverage", axes[0, 0]),
        ("entropy", "Archive Entropy", axes[0, 1]),
        ("self_bleu", "Self-BLEU ($\\uparrow$ = more collapse)", axes[1, 0]),
        ("strategy_count", "Strategy Count", axes[1, 1]),
    ]

    for metric, ylabel, ax in metrics:
        ax.plot(rounds, [r[metric] for r in greedy], 's-', color='#e74c3c',
                label='Greedy-Iter', linewidth=2, markersize=7)
        ax.plot(rounds, [r[metric] for r in qd], 'o-', color='#9b59b6',
                label='QD-Iter', linewidth=2, markersize=7)
        ax.set_xlabel('Iteration Round $t$', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[0, 0].set_title('(a) Grid Coverage', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('(b) Archive Entropy', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('(c) Self-BLEU', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('(d) Strategy Count', fontsize=12, fontweight='bold')

    plt.suptitle('Iterative Synthesis Collapse: Greedy vs QD-Synth',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig4_collapse_dynamics.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIG_DIR / 'fig4_collapse_dynamics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Fig 4 (collapse dynamics) saved")


if __name__ == "__main__":
    start = time.time()
    results = run_experiment()
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
