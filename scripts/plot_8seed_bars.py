"""Regenerate fig16 downstream bars with 8-seed statistics + significance markers."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

STATS = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/downstream/8seed_statistics.json")
FIGDIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/figures")

with open(STATS) as f:
    data = json.load(f)

models = ["greedy_57", "random_57", "qd_57", "full"]
labels = ["Greedy-57", "Random-57", "QD-57", "Full-542"]
colors = ["#e74c3c", "#95a5a6", "#2ecc71", "#3498db"]

metrics = [
    ("self_bleu", "Self-BLEU (↓ = more diverse)", False),
    ("avg_empathy", "Empathy Score (↑)", True),
    ("strategy_coverage", "Strategy Coverage (↑)", True),
    ("vocab_diversity", "Vocabulary Diversity (↑)", True),
]

fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
qd_vals_cache = {}

for ax, (metric, title, higher_better) in zip(axes, metrics):
    means, stds = [], []
    for m in models:
        d = data["per_model"][m][metric]
        means.append(d["mean"])
        stds.append(d["std"])
    qd_vals_cache[metric] = (means[2], stds[2])

    x = np.arange(len(models))
    bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor='black', linewidth=0.5, width=0.6)

    # Add significance markers for QD-57 vs Greedy-57 and vs Random-57
    if metric in data.get("wilcoxon", {}):
        for i_bl, bl in enumerate(["greedy_57", "random_57"]):
            if bl in data["wilcoxon"][metric]:
                p = data["wilcoxon"][metric][bl]["p"]
                if p < 0.01:
                    marker = "**"
                elif p < 0.05:
                    marker = "*"
                else:
                    marker = ""
                if marker:
                    y_max = max(means[i_bl], means[2]) + max(stds[i_bl], stds[2]) + 0.01
                    ax.plot([i_bl, i_bl, 2, 2], [y_max, y_max+0.005, y_max+0.005, y_max],
                            'k-', linewidth=0.8)
                    ax.text((i_bl + 2) / 2, y_max + 0.006, marker, ha='center', fontsize=9)

    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha='right')

plt.tight_layout()
plt.savefig(FIGDIR / "fig16_downstream_bars.pdf", bbox_inches='tight', dpi=300)
plt.savefig(FIGDIR / "fig16_downstream_bars.png", bbox_inches='tight', dpi=150)
print("Fig 16 saved (8-seed version)")
