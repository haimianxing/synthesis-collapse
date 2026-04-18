"""
Compute 8-seed statistics for downstream evaluation.
Uses Wilcoxon signed-rank test (proper for paired n=8 comparisons).
"""
import json, numpy as np
from pathlib import Path
from scipy import stats

RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/downstream")
SEEDS = [42, 123, 456, 789, 2024, 314, 159, 271]
MODELS = ["qd_57", "greedy_57", "random_57", "full"]
METRICS = ["strategy_coverage", "avg_empathy", "self_bleu", "vocab_diversity", "avg_length"]

def load_results():
    data = {}
    for model in MODELS:
        data[model] = {}
        for seed in SEEDS:
            path = RESULTS_DIR / f"stat_{model}_seed{seed}.json"
            if path.exists():
                with open(path) as f:
                    data[model][seed] = json.load(f)
            else:
                print(f"  WARNING: Missing {model} seed {seed}")
    return data

def compute_stats(data):
    print("=" * 80)
    print("8-SEED DOWNSTREAM EVALUATION STATISTICS")
    print("=" * 80)

    # Check completeness
    complete_seeds = {}
    for model in MODELS:
        complete_seeds[model] = sorted([s for s in SEEDS if s in data.get(model, {})])
        print(f"\n{model}: {len(complete_seeds[model])} seeds: {complete_seeds[model]}")

    # Find common seeds across all models
    common = set(complete_seeds[MODELS[0]])
    for model in MODELS[1:]:
        common &= set(complete_seeds[model])
    common = sorted(common)
    print(f"\nCommon seeds: {common} (n={len(common)})")

    if len(common) < 3:
        print("ERROR: Too few common seeds for statistics")
        return

    n = len(common)
    print(f"\n{'='*80}")
    print(f"DESCRIPTIVE STATISTICS (n={n} seeds)")
    print(f"{'='*80}")

    results_table = {}
    for metric in METRICS:
        print(f"\n--- {metric} ---")
        for model in MODELS:
            vals = [data[model][s][metric] for s in common]
            mean = np.mean(vals)
            std = np.std(vals, ddof=1)
            results_table[(model, metric)] = {"mean": mean, "std": std, "vals": vals}
            print(f"  {model:12s}: {mean:.4f} ± {std:.4f}")

    # Wilcoxon signed-rank test: QD vs each baseline
    print(f"\n{'='*80}")
    print(f"WILCOXON SIGNED-RANK TEST: QD-57 vs Baselines (n={n})")
    print(f"{'='*80}")

    baselines = ["greedy_57", "random_57", "full"]
    for metric in METRICS:
        print(f"\n--- {metric} ---")
        qd_vals = results_table[("qd_57", metric)]["vals"]
        for bl in baselines:
            bl_vals = results_table[(bl, metric)]["vals"]
            diff = np.array(qd_vals) - np.array(bl_vals)
            if np.all(diff == 0):
                print(f"  vs {bl:12s}: identical (all diffs = 0)")
                continue
            try:
                stat, p = stats.wilcoxon(qd_vals, bl_vals)
                # Effect size: r = Z / sqrt(N)
                N = len(qd_vals)
                # Z approximation from Wilcoxon
                r_effect = stat / (N * (N + 1) / 2)  # normalized W
                direction = "↑" if np.mean(diff) > 0 else "↓"
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                print(f"  vs {bl:12s}: W={stat:.1f}, p={p:.4f} {sig} {direction} mean_diff={np.mean(diff):+.4f}")
            except Exception as e:
                print(f"  vs {bl:12s}: {e}")

    # Cohen's d effect sizes
    print(f"\n{'='*80}")
    print(f"COHEN'S d EFFECT SIZES: QD-57 vs Baselines")
    print(f"{'='*80}")

    for metric in METRICS:
        print(f"\n--- {metric} ---")
        qd_vals = np.array(results_table[("qd_57", metric)]["vals"])
        for bl in baselines:
            bl_vals = np.array(results_table[(bl, metric)]["vals"])
            # Paired Cohen's d
            diff = qd_vals - bl_vals
            d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)
            magnitude = "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small"
            direction = "↑" if d > 0 else "↓"
            print(f"  vs {bl:12s}: d={d:+.3f} ({magnitude}) {direction}")

    # Bootstrap 95% CI for QD-57
    print(f"\n{'='*80}")
    print(f"BOOTSTRAP 95% CI for QD-57 (10000 resamples)")
    print(f"{'='*80}")

    np.random.seed(42)
    for metric in METRICS:
        vals = np.array(results_table[("qd_57", metric)]["vals"])
        bootstrap_means = []
        for _ in range(10000):
            sample = np.random.choice(vals, size=len(vals), replace=True)
            bootstrap_means.append(np.mean(sample))
        ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
        print(f"  {metric:20s}: {np.mean(vals):.4f} [{ci_low:.4f}, {ci_high:.4f}]")

    # LaTeX table format
    print(f"\n{'='*80}")
    print(f"LATEX TABLE ROWS (copy-paste ready)")
    print(f"{'='*80}")

    # Model display names
    display_names = {
        "qd_57": "QD-Synth-57",
        "greedy_57": "Greedy-57",
        "random_57": "Random-57",
        "full": "Full-542"
    }

    for model in MODELS:
        row = f"{display_names[model]}"
        for metric in METRICS:
            m = results_table[(model, metric)]["mean"]
            s = results_table[(model, metric)]["std"]
            row += f" & {m:.3f}$\\pm${s:.3f}"
        row += " \\\\"
        print(row)

    # Save full results
    output = {
        "n_seeds": n,
        "seeds": common,
        "per_model": {},
        "wilcoxon": {},
        "cohens_d": {}
    }
    for model in MODELS:
        output["per_model"][model] = {}
        for metric in METRICS:
            m = results_table[(model, metric)]["mean"]
            s = results_table[(model, metric)]["std"]
            output["per_model"][model][metric] = {
                "mean": float(m), "std": float(s),
                "values": [float(v) for v in results_table[(model, metric)]["vals"]]
            }

    for metric in METRICS:
        output["wilcoxon"][metric] = {}
        output["cohens_d"][metric] = {}
        qd_vals = results_table[("qd_57", metric)]["vals"]
        for bl in baselines:
            bl_vals = results_table[(bl, metric)]["vals"]
            diff = np.array(qd_vals) - np.array(bl_vals)
            d_val = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-10))
            output["cohens_d"][metric][bl] = d_val
            if not np.all(diff == 0):
                try:
                    w, p = stats.wilcoxon(qd_vals, bl_vals)
                    output["wilcoxon"][metric][bl] = {"W": float(w), "p": float(p)}
                except:
                    pass

    out_path = RESULTS_DIR / "8seed_statistics.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to {out_path}")

if __name__ == "__main__":
    data = load_results()
    compute_stats(data)
