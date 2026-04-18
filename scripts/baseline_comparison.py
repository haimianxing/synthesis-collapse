"""
Baseline comparison: Implement DPP-based and DEITA-style selection
on our Dialogue dataset for fair comparison with QD-Synth.
No GPU needed - runs on existing data.
"""
import json, random, numpy as np
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = "/mnt/data2/zcz/neurIps-emnlp/data/raw/all_dialogues_final.json"
GRID_RES = 10
OUTPUT_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/baselines")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_samples():
    with open(DATA_PATH) as f:
        all_data = json.load(f)
    samples = []
    for d in all_data:
        text = ""
        if isinstance(d.get("dialogue"), list):
            for turn in d["dialogue"]:
                if isinstance(turn, dict):
                    role = turn.get("role", turn.get("speaker", ""))
                    content = turn.get("content", turn.get("text", ""))
                    text += f"{role}: {content}\n"
        if not text.strip():
            continue
        meta = d.get("metadata", {})
        strategy = meta.get("strategies_needed", ["S1"])[0] if meta.get("strategies_needed") else "S1"
        conflict = meta.get("conflict_level", "中")
        quality = min(len(text) / 2000.0, 1.0)
        empathy_words = ["理解", "抱歉", "感谢", "体谅", "关心"]
        empathy = min(sum(1 for w in empathy_words if w in text) / 5.0, 1.0)
        strat_map = {f"S{i}": (i-1)/17.0 for i in range(1, 19)}
        conflict_map = {"低": 0.25, "中": 0.5, "高": 0.75}
        desc = (empathy, strat_map.get(strategy, 0.5), conflict_map.get(conflict, 0.5))
        samples.append({"text": text, "strategy": strategy, "conflict": conflict,
                        "quality": quality, "descriptor": desc})
    return samples


def compute_coverage(selected, grid_res=10):
    cells = set()
    strategies = set()
    for s in selected:
        cell = tuple(int(min(d * grid_res, grid_res-1)) for d in s["descriptor"][:3])
        cells.add(cell)
        strategies.add(s["strategy"])
    return {
        "coverage": len(cells) / (grid_res ** 3),
        "n_cells": len(cells),
        "strategy_coverage": len(strategies) / 18.0,
        "n_strategies": len(strategies),
        "n_samples": len(selected),
    }


def compute_self_bleu(texts):
    tokenized = [set(list(t)) for t in texts]
    if len(tokenized) < 2:
        return 0.0
    overlaps = []
    for i in range(min(len(tokenized), 30)):
        for j in range(i+1, min(len(tokenized), 30)):
            if tokenized[i] and tokenized[j]:
                overlaps.append(len(tokenized[i] & tokenized[j]) / min(len(tokenized[i]), len(tokenized[j])))
    return float(np.mean(overlaps)) if overlaps else 0.0


# === Method 1: QD-Synth (our method) ===
def qd_select(samples, k, grid_res=10):
    archive = {}
    for s in samples:
        cell = tuple(int(min(d * grid_res, grid_res-1)) for d in s["descriptor"][:3])
        if cell not in archive or s["quality"] > archive[cell]["quality"]:
            archive[cell] = s
    sorted_archive = sorted(archive.values(), key=lambda x: x["quality"], reverse=True)
    return sorted_archive[:k]


# === Method 2: Greedy Quality ===
def greedy_select(samples, k):
    return sorted(samples, key=lambda x: x["quality"], reverse=True)[:k]


# === Method 3: Random ===
def random_select(samples, k):
    return random.sample(samples, min(k, len(samples)))


# === Method 4: DPP-based Selection (k-DPP greedy) ===
def dpp_select(samples, k):
    """Greedy k-DPP: iteratively select item that maximizes determinantal diversity"""
    texts = [s["text"][:500] for s in samples]
    vectorizer = TfidfVectorizer(max_features=500, token_pattern=r'(?u)\b\w+\b')
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Quality-weighted similarity kernel
    qualities = np.array([s["quality"] for s in samples])
    sim_matrix = cosine_similarity(tfidf_matrix)

    selected_indices = []
    # First item: highest quality
    selected_indices.append(int(np.argmax(qualities)))

    for _ in range(k - 1):
        best_idx = -1
        best_score = -1

        for i in range(len(samples)):
            if i in selected_indices:
                continue
            # DPP marginal gain: quality * (1 - max similarity to already selected)
            max_sim = max(sim_matrix[i][j] for j in selected_indices)
            score = samples[i]["quality"] * (1 - max_sim)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx >= 0:
            selected_indices.append(best_idx)

    return [samples[i] for i in selected_indices]


# === Method 5: DEITA-style (Complexity + Quality + Diversity) ===
def deita_select(samples, k):
    """Simulate DEITA: score on complexity, quality, diversity dimensions"""
    # Step 1: Score on quality + complexity
    for s in samples:
        strategy_keywords = ["道歉", "解释", "补偿", "倾听", "安抚", "建议", "理解", "感谢", "承诺"]
        complexity = min(len(s["text"]) / 2000.0, 1.0) * 0.5 + sum(1 for kw in strategy_keywords if kw in s["text"]) / len(strategy_keywords) * 0.5
        s["deita_score"] = 0.4 * s["quality"] + 0.3 * complexity + 0.3 * s.get("descriptor", (0.5,0.5,0.5))[0]

    # Step 2: Take top-2k by score, then apply TF-IDF diversity filter
    ranked = sorted(samples, key=lambda x: x["deita_score"], reverse=True)[:2*k]

    # Use TF-IDF for diversity
    texts = [s["text"][:500] for s in ranked]
    vectorizer = TfidfVectorizer(max_features=300, token_pattern=r'(?u)\b\w+\b')
    tfidf_matrix = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(tfidf_matrix)

    selected_indices = [0]  # Start with top-scored
    for i in range(1, len(ranked)):
        if len(selected_indices) >= k:
            break
        max_sim = max(sim_matrix[i][j] for j in selected_indices)
        if max_sim < 0.95:  # Only reject near-duplicates
            selected_indices.append(i)

    return [ranked[i] for i in selected_indices[:k]]


# === Method 6: Dedup + Random (post-hoc baseline) ===
def dedup_random_select(samples, k, threshold=0.95):
    """Near-duplicate removal using TF-IDF + random selection"""
    # TF-IDF based dedup
    texts = [s["text"][:500] for s in samples]
    vectorizer = TfidfVectorizer(max_features=300, token_pattern=r'(?u)\b\w+\b')
    tfidf_matrix = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(tfidf_matrix)

    # Keep items that aren't too similar to any earlier item
    keep_indices = [0]
    for i in range(1, len(samples)):
        too_similar = any(sim_matrix[i][j] > threshold for j in keep_indices)
        if not too_similar:
            keep_indices.append(i)

    deduped = [samples[i] for i in keep_indices]
    print(f"  Dedup: {len(samples)} -> {len(deduped)} after removing near-duplicates (threshold={threshold})")

    # Then randomly sample k from deduplicated
    return random.sample(deduped, min(k, len(deduped)))


def run_comparison():
    print("Loading data...")
    samples = load_samples()
    print(f"Total samples: {len(samples)}")

    k = 57
    results = {}

    methods = [
        ("QD-Synth", lambda s: qd_select(s, k)),
        ("Greedy", lambda s: greedy_select(s, k)),
        ("Random", lambda s: random_select(s, k)),
        ("DPP", lambda s: dpp_select(s, k)),
        ("DEITA-style", lambda s: deita_select(s, k)),
        ("Dedup+Random", lambda s: dedup_random_select(s, k)),
    ]

    for name, method in methods:
        print(f"\nRunning {name}...")
        t0 = __import__("time").time()
        selected = method(samples)
        elapsed = __import__("time").time() - t0

        coverage = compute_coverage(selected)
        self_bleu = compute_self_bleu([s["text"] for s in selected])
        avg_quality = float(np.mean([s["quality"] for s in selected]))

        results[name] = {
            **coverage,
            "self_bleu": round(self_bleu, 4),
            "avg_quality": round(avg_quality, 4),
            "time_seconds": round(elapsed, 2),
        }

        print(f"  {name}: Coverage={coverage['coverage']:.4f}, StratCov={coverage['strategy_coverage']:.2%}, "
              f"SelfBLEU={self_bleu:.4f}, Quality={avg_quality:.4f}, Cells={coverage['n_cells']}, Time={elapsed:.1f}s")

    # Save results
    with open(OUTPUT_DIR / "baseline_comparison.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Generate comparison table
    print("\n" + "=" * 80)
    print(f"{'Method':<15} {'Coverage':>10} {'StratCov':>10} {'SelfBLEU':>10} {'Quality':>10} {'N_Cells':>10} {'Time':>10}")
    print("-" * 75)
    for name, r in results.items():
        print(f"{name:<15} {r['coverage']:>10.4f} {r['strategy_coverage']:>10.2%} "
              f"{r['self_bleu']:>10.4f} {r['avg_quality']:>10.4f} {r['n_cells']:>10} {r['time_seconds']:>10.1f}s")

    return results


def generate_figure(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIG_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/figures")

    methods = list(results.keys())
    coverages = [results[m]["coverage"] * 100 for m in methods]
    strat_covs = [results[m]["strategy_coverage"] * 100 for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    colors = ['#9b59b6', '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#1abc9c']

    bars = ax1.bar(methods, coverages, color=colors[:len(methods)], alpha=0.85, edgecolor='white')
    ax1.set_ylabel('Grid Coverage (%)', fontsize=11)
    ax1.set_title('(a) Grid Coverage at $k{=}57$', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, coverages):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}%', ha='center', fontsize=9)
    ax1.tick_params(axis='x', rotation=15)
    ax1.grid(axis='y', alpha=0.3)

    bars = ax2.bar(methods, strat_covs, color=colors[:len(methods)], alpha=0.85, edgecolor='white')
    ax2.set_ylabel('Strategy Coverage (%)', fontsize=11)
    ax2.set_title('(b) Strategy Coverage at $k{=}57$', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, strat_covs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}%', ha='center', fontsize=9)
    ax2.tick_params(axis='x', rotation=15)
    ax2.set_ylim(40, 85)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig17_baseline_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIG_DIR / 'fig17_baseline_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Fig 17 (baseline comparison) saved")


if __name__ == "__main__":
    results = run_comparison()
    generate_figure(results)
