"""
Semantic Collapse Analysis Across Iterative Rounds (v2)
Uses REAL per-round archives from code_iterative_v2 and math_iterative_v2.
Computes TF-IDF cosine similarity + vocab diversity to show:
1. Greedy: semantic diversity DECREASES (codes converge to similar patterns)
2. QD: semantic diversity MAINTAINED/INCREASES (codes stay diverse)
"""
import os, sys, json, re, ast, numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist

GRID_RES = 10

RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/semantic_analysis")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CODE_ITER_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/code_iterative_v2")
MATH_ITER_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/math_iterative_v2")

print("=== Semantic Collapse Analysis v2 (Real Per-Round Archives) ===", flush=True)

def preprocess_code(code):
    if not code:
        return ""
    code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    tokens = re.findall(r'[a-zA-Z_]\w+', code)
    return ' '.join(tokens)

def preprocess_math(answer):
    if not answer:
        return ""
    answer = re.sub(r'<<.*?>>', 'STEP', answer)
    tokens = re.findall(r'[a-zA-Z_]\w+|\d+\.?\d*', answer)
    return ' '.join(tokens)

def compute_semantic_metrics(texts):
    if len(texts) < 2:
        return {'avg_pairwise_sim': 0, 'avg_pairwise_dist': 0, 'centroid_dist': 0,
                'effective_dim': 0, 'vocab_diversity': 0, 'n_texts': len(texts)}

    vectorizer = TfidfVectorizer(max_features=500, token_pattern=r'[a-zA-Z_]\w+|\d+\.?\d*')
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except:
        return {'avg_pairwise_sim': 0, 'avg_pairwise_dist': 0, 'centroid_dist': 0,
                'effective_dim': 0, 'vocab_diversity': 0, 'n_texts': len(texts)}

    sim_matrix = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(sim_matrix, 0)
    avg_sim = float(np.mean(sim_matrix[sim_matrix > 0])) if np.any(sim_matrix > 0) else 0

    dense = tfidf_matrix.toarray()
    distances = pdist(dense, metric='cosine')
    avg_dist = float(np.mean(distances))

    centroid = np.mean(dense, axis=0)
    centroid_dists = [1 - np.dot(dense[i], centroid) / (np.linalg.norm(dense[i]) * np.linalg.norm(centroid) + 1e-10)
                      for i in range(len(dense))]

    cov = np.cov(dense.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-10]
    eff_dim = float(np.sum(eigvals)**2 / np.sum(eigvals**2)) if len(eigvals) > 0 else 0

    all_tokens = ' '.join(texts).split()
    vocab_div = len(set(all_tokens)) / max(len(all_tokens), 1)

    return {
        'avg_pairwise_sim': round(avg_sim, 4),
        'avg_pairwise_dist': round(avg_dist, 4),
        'centroid_dist': round(float(np.mean(centroid_dists)), 4),
        'effective_dim': round(eff_dim, 2),
        'vocab_diversity': round(vocab_div, 4),
        'n_texts': len(texts)
    }


all_results = {}

# ============ Code Domain (Per-Round Archives) ============
print("\n--- Code Domain (Per-Round Archives) ---", flush=True)
code_results = {}

for strategy in ["greedy", "qd"]:
    round_metrics = []
    for rnd in range(10):
        archive_path = CODE_ITER_DIR / f"{strategy}_archive_r{rnd}.json"
        if not archive_path.exists():
            continue
        with open(archive_path) as f:
            items = json.load(f)

        texts = [preprocess_code(item.get('code', '')) for item in items]
        texts = [t for t in texts if len(t) > 10]

        metrics = compute_semantic_metrics(texts)
        metrics['round'] = rnd
        metrics['n_archive_items'] = len(items)
        round_metrics.append(metrics)

        print(f"  {strategy.upper()} R{rnd}: n={len(items)}, sim={metrics['avg_pairwise_sim']:.3f}, "
              f"dist={metrics['avg_pairwise_dist']:.3f}, eff_dim={metrics['effective_dim']:.1f}, "
              f"vocab_div={metrics['vocab_diversity']:.3f}", flush=True)

    code_results[strategy] = round_metrics

    # Compute trend
    if len(round_metrics) >= 2:
        first, last = round_metrics[0], round_metrics[-1]
        sim_delta = last['avg_pairwise_sim'] - first['avg_pairwise_sim']
        div_delta = last['vocab_diversity'] - first['vocab_diversity']
        print(f"  {strategy.upper()} trend: sim {first['avg_pairwise_sim']:.3f}→{last['avg_pairwise_sim']:.3f} ({sim_delta:+.3f}), "
              f"vocab_div {first['vocab_diversity']:.3f}→{last['vocab_diversity']:.3f} ({div_delta:+.3f})", flush=True)

all_results['code'] = code_results

# ============ Math Domain (Per-Round Archives) ============
print("\n--- Math Domain (Per-Round Archives) ---", flush=True)
math_results = {}

for strategy in ["greedy", "qd"]:
    round_metrics = []
    for rnd in range(10):
        archive_path = MATH_ITER_DIR / f"{strategy}_archive_r{rnd}.json"
        if not archive_path.exists():
            continue
        with open(archive_path) as f:
            items = json.load(f)

        texts = [preprocess_math(item.get('answer', '')) for item in items]
        texts = [t for t in texts if len(t) > 10]

        metrics = compute_semantic_metrics(texts)
        metrics['round'] = rnd
        metrics['n_archive_items'] = len(items)
        round_metrics.append(metrics)

        print(f"  {strategy.upper()} R{rnd}: n={len(items)}, sim={metrics['avg_pairwise_sim']:.3f}, "
              f"dist={metrics['avg_pairwise_dist']:.3f}, eff_dim={metrics['effective_dim']:.1f}, "
              f"vocab_div={metrics['vocab_diversity']:.3f}", flush=True)

    math_results[strategy] = round_metrics

    if len(round_metrics) >= 2:
        first, last = round_metrics[0], round_metrics[-1]
        sim_delta = last['avg_pairwise_sim'] - first['avg_pairwise_sim']
        div_delta = last['vocab_diversity'] - first['vocab_diversity']
        print(f"  {strategy.upper()} trend: sim {first['avg_pairwise_sim']:.3f}→{last['avg_pairwise_sim']:.3f} ({sim_delta:+.3f}), "
              f"vocab_div {first['vocab_diversity']:.3f}→{last['vocab_diversity']:.3f} ({div_delta:+.3f})", flush=True)

all_results['math'] = math_results

# ============ Cross-Round Novelty ============
print(f"\n--- Cross-Round Novelty ---", flush=True)
for domain, iter_dir in [("code", CODE_ITER_DIR), ("math", MATH_ITER_DIR)]:
    for strategy in ["greedy", "qd"]:
        archives = {}
        for rnd in range(10):
            path = iter_dir / f"{strategy}_archive_r{rnd}.json"
            if path.exists():
                with open(path) as f:
                    archives[rnd] = json.load(f)
        if len(archives) < 2:
            continue

        sorted_rounds = sorted(archives.keys())
        print(f"\n  {domain.upper()} {strategy.upper()}:", flush=True)
        for i in range(1, len(sorted_rounds)):
            prev_rnd = sorted_rounds[i-1]
            curr_rnd = sorted_rounds[i]
            if domain == "code":
                prev_codes = set(preprocess_code(item.get('code', '')) for item in archives[prev_rnd])
                curr_codes = set(preprocess_code(item.get('code', '')) for item in archives[curr_rnd])
            else:
                prev_codes = set(preprocess_math(item.get('answer', '')) for item in archives[prev_rnd])
                curr_codes = set(preprocess_math(item.get('answer', '')) for item in archives[curr_rnd])

            novel = curr_codes - prev_codes
            novelty_rate = len(novel) / max(len(curr_codes), 1)
            print(f"    R{prev_rnd}→R{curr_rnd}: {len(curr_codes)} items, {len(novel)} novel ({novelty_rate:.1%})", flush=True)

# ============ Save ============
with open(RESULTS_DIR / "semantic_collapse_v2.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\nResults saved to {RESULTS_DIR}", flush=True)
