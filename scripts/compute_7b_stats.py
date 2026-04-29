"""
Compute multi-seed statistics for 7B downstream evaluation.
Reads eval files, computes mean±std, Wilcoxon tests, Cohen's d.
"""
import json, numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/downstream_7b")
MODELS = ["greedy_57", "qd_57", "random_57", "full"]
SEEDS = [42, 123, 456, 789, 2024]
METRICS = ["strategy_coverage", "self_bleu", "avg_empathy", "vocab_diversity"]

# Collect all results
data = {}
for model in MODELS:
    data[model] = {m: [] for m in METRICS}
    for seed in SEEDS:
        path = RESULTS_DIR / f"eval_{model}_seed{seed}.json"
        if path.exists():
            with open(path) as f:
                r = json.load(f)
            for m in METRICS:
                if m in r:
                    data[model][m].append(r[m])
        else:
            print(f"  Missing: {path.name}")

print("\n=== 7B Multi-Seed Statistics ===\n")

# Descriptive stats
for model in MODELS:
    print(f"\n{model}:")
    for m in METRICS:
        vals = data[model][m]
        if vals:
            print(f"  {m}: {np.mean(vals):.3f} ± {np.std(vals):.3f} (n={len(vals)})")
        else:
            print(f"  {m}: NO DATA")

# Wilcoxon tests: QD vs Greedy, QD vs Random
print("\n=== Wilcoxon Signed-Rank Tests (QD-57 vs Baselines) ===\n")
pairs = [("qd_57", "greedy_57"), ("qd_57", "random_57")]
for m in METRICS:
    print(f"\n{m}:")
    for qd, baseline in pairs:
        qd_vals = data[qd][m]
        base_vals = data[baseline][m]
        n = min(len(qd_vals), len(base_vals))
        if n >= 5:
            qd_arr = np.array(qd_vals[:n])
            base_arr = np.array(base_vals[:n])
            diff = qd_arr - base_arr
            if np.all(diff == 0):
                print(f"  QD vs {baseline}: all differences are zero")
                continue
            try:
                stat, p = wilcoxon(qd_arr, base_arr)
                cohens_d = float(np.mean(diff)) / float(np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0
                sig = "*" if p < 0.05 else "**" if p < 0.01 else "ns"
                print(f"  QD vs {baseline}: p={p:.4f}, d={cohens_d:.2f} ({sig}, n={n})")
            except Exception as e:
                print(f"  QD vs {baseline}: test failed ({e})")
        else:
            print(f"  QD vs {baseline}: insufficient data (n={n})")

# Save results
output = {"model_size": "7B", "seeds": SEEDS, "models": {}}
for model in MODELS:
    output["models"][model] = {}
    for m in METRICS:
        vals = data[model][m]
        if vals:
            output["models"][model][m] = {
                "mean": round(float(np.mean(vals)), 3),
                "std": round(float(np.std(vals)), 3),
                "n": len(vals),
                "values": [round(v, 3) for v in vals]
            }

out_path = RESULTS_DIR / "7b_multiseed_statistics.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {out_path}")
