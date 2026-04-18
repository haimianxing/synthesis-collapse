#!/usr/bin/env python3
"""
Retrospective analysis: Simple Dedup baseline vs QD vs Greedy.
Uses existing V3 base-reset generation data to simulate simple dedup selection.

Key question: Does "1 sample per cell + greedy quality" achieve the same
diversity/quality as QD MAP-Elites selection?

This is a CPU-only analysis - no GPU or training needed.
"""
import json, os, sys
from collections import Counter, defaultdict
import numpy as np

# === Config ===
V3_RESULTS_DIR = "/mnt/data2/zcz/neurIps-emnlp/neurips/results/self_synthesis_v3_base_reset"
STRATEGIES = ["qd_s42", "greedy_s42"]
OUTPUT_DIR = "/mnt/data2/zcz/neurIps-emnlp/neurips/results/simple_dedup_analysis"
GRID_RES = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_entropy(cell_counts):
    """Compute Shannon entropy from cell count distribution."""
    total = sum(cell_counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in cell_counts.values() if c > 0]
    return -sum(p * np.log2(p) for p in probs)

def simple_dedup_select(generated_data, n_select, grid_res):
    """
    Simple Dedup selection:
    1. For each cell, keep the highest quality sample
    2. Fill remaining quota with highest quality samples from all data
    """
    # Group by cell
    cell_groups = defaultdict(list)
    no_cell = []
    for item in generated_data:
        cell = item.get('cell')
        if cell is not None:
            cell_groups[cell].append(item)
        else:
            no_cell.append(item)

    # Step 1: Best per cell
    selected = []
    used_cells = set()
    for cell, items in cell_groups.items():
        best = max(items, key=lambda x: x.get('quality', 0))
        selected.append(best)
        used_cells.add(cell)

    # Step 2: Fill remaining with highest quality from ALL data
    remaining_quota = n_select - len(selected)
    if remaining_quota > 0:
        # Get all items not already selected (by index/identity)
        selected_ids = set(id(s) for s in selected)
        remaining = [item for item in generated_data if id(item) not in selected_ids]
        remaining.sort(key=lambda x: x.get('quality', 0), reverse=True)
        selected.extend(remaining[:remaining_quota])

    return selected[:n_select]

def analyze_round_data(strategy_name, round_key, round_data):
    """Extract selection statistics from a round."""
    selected = round_data.get('selected_data', [])
    if not selected:
        return None

    # Count cells
    cells = [item.get('cell') for item in selected if item.get('cell') is not None]
    cell_counts = Counter(cells)
    n_cells = len(cell_counts)
    entropy = compute_entropy(cell_counts)

    # Quality stats
    qualities = [item.get('quality', 0) for item in selected]
    avg_quality = np.mean(qualities) if qualities else 0

    # Correctness
    correct = sum(1 for item in selected if item.get('correct', False))

    return {
        'n_selected': len(selected),
        'n_cells': n_cells,
        'entropy': entropy,
        'avg_quality': avg_quality,
        'n_correct': correct,
        'cell_counts': dict(cell_counts),
    }

def main():
    all_results = {}

    for strategy in STRATEGIES:
        result_file = os.path.join(V3_RESULTS_DIR, strategy, f"{strategy}_v3.json")
        if not os.path.exists(result_file):
            print(f"SKIP: {result_file} not found")
            continue

        with open(result_file) as f:
            data = json.load(f)

        print(f"\n{'='*60}")
        print(f"Strategy: {strategy}")
        print(f"{'='*60}")

        strategy_results = {}

        for round_key in sorted(data.keys()):
            if not round_key.startswith('round_'):
                continue

            round_data = data[round_key]
            generated = round_data.get('generated_data', [])
            selected_actual = round_data.get('selected_data', [])

            if not generated or not selected_actual:
                continue

            round_num = round_data.get('round', '?')
            n_select = len(selected_actual)

            # 1. Actual selection stats
            actual_stats = analyze_round_data(strategy, round_key, round_data)

            # 2. Simple Dedup selection from SAME generated pool
            dedup_selected = simple_dedup_select(generated, n_select, GRID_RES)
            dedup_stats = {
                'n_selected': len(dedup_selected),
                'n_cells': len(set(item.get('cell') for item in dedup_selected if item.get('cell'))),
                'entropy': compute_entropy(Counter(item.get('cell') for item in dedup_selected if item.get('cell'))),
                'avg_quality': np.mean([item.get('quality', 0) for item in dedup_selected]),
                'n_correct': sum(1 for item in dedup_selected if item.get('correct', False)),
            }

            # 3. Overlap analysis
            actual_questions = set(item.get('question', '')[:200] for item in selected_actual)
            dedup_questions = set(item.get('question', '')[:200] for item in dedup_selected)
            overlap = actual_questions & dedup_questions
            overlap_pct = len(overlap) / len(actual_questions) * 100 if actual_questions else 0

            # 4. Exclusive cells
            actual_cells = set(item.get('cell') for item in selected_actual if item.get('cell'))
            dedup_cells = set(item.get('cell') for item in dedup_selected if item.get('cell'))
            exclusive_actual = actual_cells - dedup_cells
            exclusive_dedup = dedup_cells - actual_cells

            result = {
                'round': round_num,
                'n_generated': len(generated),
                'n_select': n_select,
                'actual': actual_stats,
                'simple_dedup': dedup_stats,
                'overlap_pct': overlap_pct,
                'n_overlap': len(overlap),
                'exclusive_cells_actual': len(exclusive_actual),
                'exclusive_cells_dedup': len(exclusive_dedup),
            }
            strategy_results[round_key] = result

            print(f"\n  Round {round_num} (gen={len(generated)}, sel={n_select}):")
            print(f"    {'Method':<20} {'Cells':>6} {'H':>6} {'AvgQ':>6} {'Correct':>8}")
            print(f"    {'-'*50}")
            if actual_stats:
                print(f"    {'Actual':<20} {actual_stats['n_cells']:>6} {actual_stats['entropy']:>6.2f} {actual_stats['avg_quality']:>6.3f} {actual_stats['n_correct']:>8}")
            print(f"    {'Simple Dedup':<20} {dedup_stats['n_cells']:>6} {dedup_stats['entropy']:>6.2f} {dedup_stats['avg_quality']:>6.3f} {dedup_stats['n_correct']:>8}")
            print(f"    Overlap: {overlap_pct:.1f}% ({len(overlap)}/{len(actual_questions)})")
            print(f"    Exclusive cells: actual={len(exclusive_actual)}, dedup={len(exclusive_dedup)}")

        all_results[strategy] = strategy_results

    # Save results
    output_file = os.path.join(OUTPUT_DIR, "simple_dedup_vs_qd.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")

    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY: Simple Dedup vs QD vs Greedy")
    print(f"{'='*60}")
    print(f"{'Strategy':<20} {'Round':>6} {'Method':<20} {'Cells':>6} {'H':>6} {'Overlap':>8}")
    print("-" * 70)
    for strategy, rounds in all_results.items():
        for round_key, r in sorted(rounds.items()):
            if r.get('actual'):
                print(f"{strategy:<20} {r['round']:>6} {'Actual':<20} {r['actual']['n_cells']:>6} {r['actual']['entropy']:>6.2f} {'':>8}")
            print(f"{'':<20} {r['round']:>6} {'Simple Dedup':<20} {r['simple_dedup']['n_cells']:>6} {r['simple_dedup']['entropy']:>6.2f} {r['overlap_pct']:>7.1f}%")

if __name__ == "__main__":
    main()
