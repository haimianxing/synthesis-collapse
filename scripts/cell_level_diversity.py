"""
Cell-level diversity analysis for QD-Synth.
Uses strategy × conflict as grid cells (Dialogue domain).
Compares intra-cell vs inter-cell diversity between QD and Greedy.
"""
import json, numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/downstream")

def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

print("Loading data...", flush=True)
qd_data = load_jsonl(RESULTS_DIR / "data_qd_57.jsonl")
greedy_data = load_jsonl(RESULTS_DIR / "data_greedy_57.jsonl")
random_data = load_jsonl(RESULTS_DIR / "data_random_57.jsonl")
print(f"QD: {len(qd_data)}, Greedy: {len(greedy_data)}, Random: {len(random_data)}", flush=True)

# Cell = (strategy, conflict) pair
def get_cell(item):
    s = item.get('strategy', 'unknown')
    c = item.get('conflict', 'unknown')
    return (s, c)

# Build cell maps
qd_cells = {}
for i, item in enumerate(qd_data):
    cell = get_cell(item)
    qd_cells.setdefault(cell, []).append(i)

greedy_cells = {}
for i, item in enumerate(greedy_data):
    cell = get_cell(item)
    greedy_cells.setdefault(cell, []).append(i)

random_cells = {}
for i, item in enumerate(random_data):
    cell = get_cell(item)
    random_cells.setdefault(cell, []).append(i)

qd_cell_list = list(qd_cells.keys())
greedy_cell_list = list(greedy_cells.keys())
random_cell_list = list(random_cells.keys())

print(f"\nQD cells: {len(qd_cell_list)}", flush=True)
print(f"Greedy cells: {len(greedy_cell_list)}", flush=True)
print(f"Random cells: {len(random_cell_list)}", flush=True)

# TF-IDF embeddings
all_data = qd_data + greedy_data + random_data
all_texts = [item.get('text', '') for item in all_data]

print(f"\nComputing TF-IDF for {len(all_texts)} items...", flush=True)
vectorizer = TfidfVectorizer(max_features=500, analyzer='char', ngram_range=(2, 4))
embeddings = vectorizer.fit_transform(all_texts)

n_qd = len(qd_data)
n_greedy = len(greedy_data)
qd_embeds = embeddings[:n_qd]
greedy_embeds = embeddings[n_qd:n_qd + n_greedy]
random_embeds = embeddings[n_qd + n_greedy:]

def avg_pairwise_distance(embeds):
    if embeds.shape[0] < 2:
        return 0.0
    sim = cosine_similarity(embeds)
    n = sim.shape[0]
    total = sum(1.0 - sim[i, j] for i in range(n) for j in range(i + 1, n))
    count = n * (n - 1) // 2
    return total / count if count > 0 else 0.0

# 1. Overall pairwise diversity
print("\n=== Overall Pairwise Diversity (TF-IDF cosine distance) ===", flush=True)
qd_dist = avg_pairwise_distance(qd_embeds)
greedy_dist = avg_pairwise_distance(greedy_embeds)
random_dist = avg_pairwise_distance(random_embeds)
print(f"QD-57:     {qd_dist:.4f}", flush=True)
print(f"Greedy-57: {greedy_dist:.4f}", flush=True)
print(f"Random-57: {random_dist:.4f}", flush=True)

# 2. Cell-level analysis
print("\n=== Cell-Level Analysis ===", flush=True)

# Greedy redundancy: samples per cell
greedy_cell_sizes = [len(v) for v in greedy_cells.values()]
qd_cell_sizes = [len(v) for v in qd_cells.values()]

print(f"\nGreedy max samples/cell: {max(greedy_cell_sizes)}, avg: {np.mean(greedy_cell_sizes):.1f}", flush=True)
print(f"QD max samples/cell: {max(qd_cell_sizes)}, avg: {np.mean(qd_cell_sizes):.1f}", flush=True)

# Greedy cells with multiple samples = redundancy
greedy_multi_cells = sum(1 for s in greedy_cell_sizes if s > 1)
print(f"Greedy cells with 2+ samples: {greedy_multi_cells}/{len(greedy_cell_list)}", flush=True)

# 3. Intra-cell similarity for Greedy (multi-sample cells)
# High intra-cell similarity = redundant samples
print("\n=== Greedy Intra-Cell Redundancy ===", flush=True)
greedy_intra_dists = []
for cell, indices in greedy_cells.items():
    if len(indices) >= 2:
        cell_embeds = greedy_embeds[indices]
        dist = avg_pairwise_distance(cell_embeds)
        greedy_intra_dists.append((cell, dist, len(indices)))

if greedy_intra_dists:
    for cell, dist, count in sorted(greedy_intra_dists, key=lambda x: -x[2])[:5]:
        print(f"  Cell {cell}: {count} samples, intra-cell dist = {dist:.4f}", flush=True)
    avg_intra = np.mean([d for _, d, _ in greedy_intra_dists])
    print(f"  Avg intra-cell distance (multi-sample cells): {avg_intra:.4f}", flush=True)

# 4. Inter-cell distance: how different are cells from each other?
print("\n=== Inter-Cell Diversity ===", flush=True)

def compute_inter_cell_distance(cells, embeds):
    """Average distance between cell representatives."""
    from scipy.sparse import vstack as sparse_vstack
    reps = []
    for cell in cells:
        idx = cells[cell][0]  # Take first sample as representative
        reps.append(embeds[idx])
    if len(reps) < 2:
        return 0.0
    reps_matrix = sparse_vstack(reps)
    sim = cosine_similarity(reps_matrix)
    n = sim.shape[0]
    total = sum(1.0 - sim[i, j] for i in range(n) for j in range(i + 1, n))
    count = n * (n - 1) // 2
    return total / count if count > 0 else 0.0

qd_inter = compute_inter_cell_distance(qd_cells, qd_embeds)
greedy_inter = compute_inter_cell_distance(greedy_cells, greedy_embeds)
random_inter = compute_inter_cell_distance(random_cells, random_embeds)

print(f"QD inter-cell distance:     {qd_inter:.4f} ({len(qd_cell_list)} cells)", flush=True)
print(f"Greedy inter-cell distance: {greedy_inter:.4f} ({len(greedy_cell_list)} cells)", flush=True)
print(f"Random inter-cell distance: {random_inter:.4f} ({len(random_cell_list)} cells)", flush=True)

# 5. Strategy coverage
qd_strategies = set(item.get('strategy', '') for item in qd_data)
greedy_strategies = set(item.get('strategy', '') for item in greedy_data)
random_strategies = set(item.get('strategy', '') for item in random_data)

print(f"\n=== Strategy Coverage ===", flush=True)
print(f"QD strategies:     {sorted(qd_strategies)} ({len(qd_strategies)})", flush=True)
print(f"Greedy strategies: {sorted(greedy_strategies)} ({len(greedy_strategies)})", flush=True)
print(f"Random strategies: {sorted(random_strategies)} ({len(random_strategies)})", flush=True)

# 6. Conflict balance
qd_conflicts = [item.get('conflict', '') for item in qd_data]
greedy_conflicts = [item.get('conflict', '') for item in greedy_data]

print(f"\n=== Conflict Distribution ===", flush=True)
for name, conflicts in [("QD", qd_conflicts), ("Greedy", greedy_conflicts)]:
    from collections import Counter
    counts = Counter(conflicts)
    print(f"  {name}: {dict(counts)}", flush=True)

# Summary
results = {
    "pairwise_diversity": {
        "qd_57": round(float(qd_dist), 4),
        "greedy_57": round(float(greedy_dist), 4),
        "random_57": round(float(random_dist), 4)
    },
    "cell_occupation": {
        "qd_unique_cells": len(qd_cell_list),
        "greedy_unique_cells": len(greedy_cell_list),
        "random_unique_cells": len(random_cell_list),
        "greedy_max_per_cell": int(max(greedy_cell_sizes)),
        "greedy_avg_per_cell": round(float(np.mean(greedy_cell_sizes)), 1),
        "greedy_redundant_cells": greedy_multi_cells
    },
    "inter_cell_diversity": {
        "qd": round(float(qd_inter), 4),
        "greedy": round(float(greedy_inter), 4),
        "random": round(float(random_inter), 4)
    },
    "strategy_coverage": {
        "qd": len(qd_strategies),
        "greedy": len(greedy_strategies),
        "random": len(random_strategies)
    },
    "redundancy_ratio": {
        "qd": round(len(qd_data) / max(len(qd_cell_list), 1), 2),
        "greedy": round(len(greedy_data) / max(len(greedy_cell_list), 1), 2),
        "random": round(len(random_data) / max(len(random_cell_list), 1), 2)
    }
}

if greedy_intra_dists:
    results["greedy_intra_cell_distance"] = round(float(avg_intra), 4)

out_path = RESULTS_DIR / "cell_level_diversity.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n=== Summary ===", flush=True)
print(json.dumps(results, indent=2), flush=True)
print(f"\nSaved to {out_path}", flush=True)
