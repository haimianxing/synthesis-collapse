"""Generate radar chart and case study for downstream evaluation."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

FIG_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/figures")
RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/downstream")

# Load results
models = ["greedy_57", "qd_57", "random_57", "full"]
results = {}
for m in models:
    with open(RESULTS_DIR / f"eval_{m}.json") as f:
        results[m] = json.load(f)

# Also load old vocab diversity (per-text computation, more meaningful)
with open(RESULTS_DIR / "downstream_results.json") as f:
    old_results = json.load(f)

# === 1. Radar Chart ===
# Normalize metrics to 0-1 scale for radar chart
metrics = {
    "Strategy\nCoverage": {m: results[m]["strategy_coverage"] for m in models},
    "Empathy\nScore": {m: results[m]["avg_empathy"] for m in models},
    "Vocab\nDiversity": {m: old_results[m]["vocab_diversity"] for m in models},
    "Diversity\n(1-SelfBLEU)": {m: 1 - results[m]["self_bleu"] for m in models},
    "Response\nLength": {m: min(results[m]["avg_length"] / 500, 1.0) for m in models},
}

labels = list(metrics.keys())
n_metrics = len(labels)
angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
angles += angles[:1]  # close the polygon

colors = {'greedy_57': '#e74c3c', 'qd_57': '#9b59b6', 'random_57': '#3498db', 'full': '#2ecc71'}
display_names = {'greedy_57': 'Greedy-57', 'qd_57': 'QD-57', 'random_57': 'Random-57', 'full': 'Full-542'}

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for model in models:
    values = [metrics[lab][model] for lab in labels]
    values += values[:1]  # close polygon
    ax.plot(angles, values, 'o-', color=colors[model], label=display_names[model], linewidth=2, markersize=6)
    ax.fill(angles, values, alpha=0.1, color=colors[model])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0, 1)
ax.set_title('Downstream Evaluation: Multi-Dimensional Comparison', fontsize=13, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig15_downstream_radar.pdf', dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / 'fig15_downstream_radar.png', dpi=300, bbox_inches='tight')
plt.close()
print("Fig 15 (radar chart) saved")

# === 2. Bar Chart: Key Metrics Comparison ===
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# (a) Self-BLEU (lower is better)
ax = axes[0]
x = np.arange(len(models))
self_bleus = [results[m]["self_bleu"] for m in models]
bars = ax.bar(x, self_bleus, color=[colors[m] for m in models], alpha=0.8, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels([display_names[m] for m in models], fontsize=9)
ax.set_ylabel('Self-BLEU (lower = more diverse)', fontsize=10)
ax.set_title('(a) Response Diversity', fontsize=11, fontweight='bold')
for bar, val in zip(bars, self_bleus):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{val:.4f}', ha='center', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# (b) Empathy Score
ax = axes[1]
empathies = [results[m]["avg_empathy"] for m in models]
bars = ax.bar(x, empathies, color=[colors[m] for m in models], alpha=0.8, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels([display_names[m] for m in models], fontsize=9)
ax.set_ylabel('Empathy Score', fontsize=10)
ax.set_title('(b) Response Empathy', fontsize=11, fontweight='bold')
for bar, val in zip(bars, empathies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# (c) Strategy Coverage
ax = axes[2]
strat_covs = [results[m]["strategy_coverage"] for m in models]
bars = ax.bar(x, strat_covs, color=[colors[m] for m in models], alpha=0.8, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels([display_names[m] for m in models], fontsize=9)
ax.set_ylabel('Strategy Coverage', fontsize=10)
ax.set_title('(c) Strategy Coverage', fontsize=11, fontweight='bold')
for bar, val in zip(bars, strat_covs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.1%}', ha='center', fontsize=9)
ax.set_ylim(0.7, 1.0)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig16_downstream_bars.pdf', dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / 'fig16_downstream_bars.png', dpi=300, bbox_inches='tight')
plt.close()
print("Fig 16 (bar charts) saved")

# === 3. Print summary ===
print("\n=== Downstream Evaluation Summary ===")
print(f"{'Model':<15} {'StratCov':>10} {'SelfBLEU':>10} {'Empathy':>10} {'VocabDiv':>10} {'AvgLen':>10}")
print("-" * 65)
for m in models:
    r = results[m]
    vd = old_results[m]["vocab_diversity"]
    print(f"{display_names[m]:<15} {r['strategy_coverage']:>10.2%} {r['self_bleu']:>10.4f} {r['avg_empathy']:>10.4f} {vd:>10.4f} {r['avg_length']:>10.1f}")
