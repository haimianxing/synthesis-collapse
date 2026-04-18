#!/usr/bin/env python3
"""
V7 Code Self-Synthesis Results Analysis.
Reads V7 JSON results and generates comparison tables + statistics.

Usage: python3 analyze_v7_results.py
"""
import json, os
import numpy as np
from collections import defaultdict

RESULTS_DIR = "/mnt/data2/zcz/neurIps-emnlp/neurips/results/self_synthesis_v7_code"
STRATEGIES = ["qd", "greedy", "simple_dedup", "random"]
SEEDS = [42, 123]
N_ROUNDS = 4

def load_results():
    """Load all V7 results."""
    all_results = {}
    for strategy in STRATEGIES:
        for seed in SEEDS:
            name = f"{strategy}_s{seed}"
            result_file = os.path.join(RESULTS_DIR, name, f"{name}_v7.json")
            if os.path.exists(result_file):
                with open(result_file) as f:
                    all_results[name] = json.load(f)
                print(f"Loaded {name}: {len(all_results[name])} rounds")
            else:
                print(f"Missing: {name}")
    return all_results

def analyze_accuracy(all_results):
    """Compare accuracy across strategies and rounds."""
    print("\n" + "=" * 70)
    print("ACCURACY COMPARISON (HumanEval pass@1)")
    print("=" * 70)

    # Group by strategy
    strat_accs = defaultdict(lambda: defaultdict(list))
    for name, data in all_results.items():
        parts = name.rsplit("_s", 1)
        strategy = parts[0]
        seed = int(parts[1])
        for rnd in range(N_ROUNDS + 1):
            key = f"{name}_r{rnd}"
            if key in data:
                strat_accs[strategy][rnd].append(data[key]['accuracy'])

    # Print table
    header = f"{'Strategy':<15}"
    for rnd in range(N_ROUNDS + 1):
        header += f" {'R'+str(rnd):>10}"
    header += f" {'Peak':>8} {'Drift':>8}"
    print(header)
    print("-" * len(header))

    for strategy in STRATEGIES:
        row = f"{strategy:<15}"
        accs = []
        for rnd in range(N_ROUNDS + 1):
            vals = strat_accs[strategy].get(rnd, [])
            if vals:
                mean = np.mean(vals)
                std = np.std(vals) if len(vals) > 1 else 0
                accs.append(mean)
                row += f" {mean:>7.1%}"
                if len(vals) > 1:
                    row += f"±{std:.1%}"
                else:
                    row += "  "
            else:
                row += f" {'N/A':>10}"
        if accs:
            peak = max(accs)
            drift = peak - accs[-1] if accs else 0
            row += f" {peak:>7.1%} {drift:>7.1%}"
        print(row)

def analyze_cells(all_results):
    """Compare cell coverage and entropy across strategies."""
    print("\n" + "=" * 70)
    print("CELL COVERAGE AND ENTROPY")
    print("=" * 70)

    strat_metrics = defaultdict(lambda: defaultdict(list))
    for name, data in all_results.items():
        parts = name.rsplit("_s", 1)
        strategy = parts[0]
        for rnd in range(N_ROUNDS + 1):
            key = f"{name}_r{rnd}"
            if key in data:
                r = data[key]
                strat_metrics[strategy][rnd].append({
                    'n_cells': r.get('n_cells_selected', 0),
                    'entropy': r.get('sel_entropy', 0),
                })

    for metric, label in [('n_cells', 'Selected Cells'), ('entropy', 'Sel. Entropy')]:
        print(f"\n  {label}:")
        header = f"    {'Strategy':<15}"
        for rnd in range(N_ROUNDS + 1):
            header += f" {'R'+str(rnd):>8}"
        print(header)
        print("    " + "-" * (len(header) - 4))

        for strategy in STRATEGIES:
            row = f"    {strategy:<15}"
            for rnd in range(N_ROUNDS + 1):
                vals = [m[metric] for m in strat_metrics[strategy].get(rnd, [])]
                if vals:
                    mean = np.mean(vals)
                    row += f" {mean:>8.2f}"
                else:
                    row += f" {'N/A':>8}"
            print(row)

def analyze_mechanism(all_results):
    """Test dedup mechanism: QD vs SimpleDedup vs Greedy."""
    print("\n" + "=" * 70)
    print("MECHANISM ANALYSIS: QD vs Simple-Dedup vs Greedy")
    print("=" * 70)

    # For each seed, compare strategies
    for seed in SEEDS:
        print(f"\n  Seed {seed}:")
        qd_name = f"qd_s{seed}"
        greedy_name = f"greedy_s{seed}"
        dedup_name = f"simple_dedup_s{seed}"

        for rnd in range(N_ROUNDS + 1):
            results = {}
            for name, label in [(qd_name, 'QD'), (greedy_name, 'Greedy'), (dedup_name, 'Dedup')]:
                if name in all_results:
                    key = f"{name}_r{rnd}"
                    if key in all_results[name]:
                        r = all_results[name][key]
                        results[label] = {
                            'acc': r['accuracy'],
                            'cells': r.get('n_cells_selected', 0),
                            'entropy': r.get('sel_entropy', 0),
                        }

            if len(results) == 3:
                print(f"    R{rnd}: QD={results['QD']['acc']:.1%} (H={results['QD']['entropy']:.2f}, c={results['QD']['cells']})"
                      f"  Greedy={results['Greedy']['acc']:.1%} (H={results['Greedy']['entropy']:.2f}, c={results['Greedy']['cells']})"
                      f"  Dedup={results['Dedup']['acc']:.1%} (H={results['Dedup']['entropy']:.2f}, c={results['Dedup']['cells']})")

                # Check if QD ≈ Dedup >> Greedy or QD > Dedup > Greedy
                qd_g = results['QD']['acc'] - results['Greedy']['acc']
                dedup_g = results['Dedup']['acc'] - results['Greedy']['acc']
                qd_dedup = results['QD']['acc'] - results['Dedup']['acc']
                print(f"         QD-Greedy={qd_g:+.1%}  Dedup-Greedy={dedup_g:+.1%}  QD-Dedup={qd_dedup:+.1%}")

def main():
    all_results = load_results()
    if not all_results:
        print("\nNo results found yet. V7 experiments still running.")
        return

    print(f"\nLoaded {len(all_results)} strategy-seed combinations")
    analyze_accuracy(all_results)
    analyze_cells(all_results)
    analyze_mechanism(all_results)

if __name__ == "__main__":
    main()
