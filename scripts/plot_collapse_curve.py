"""
Plot Self-Synthesis Collapse Curves (v1 + v2)
==============================================
Reads results from self_synthesis_7b/ (v1) and self_synthesis_v2/ (v2)
Generates: collapse curves, cell coverage, entropy, diversity radar
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

RESULTS_BASE = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results")
FIGURES_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/figures")
FIGURES_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'figure.autolayout': True
})

COLORS = {'greedy': '#e74c3c', 'qd': '#2ecc71', 'random': '#3498db', 'qd_no_surprisal': '#f39c12'}
LABELS = {'greedy': 'Greedy', 'qd': 'QD-Synth (Ours)', 'random': 'Random', 'qd_no_surprisal': 'QD w/o Surprisal'}

def load_v1_results():
    """Load v1 7B results."""
    base = RESULTS_BASE / "self_synthesis_7b"
    data = {}
    for strategy in ['greedy', 'qd', 'random']:
        fpath = base / strategy / f"{strategy}_self_synthesis_7b.json"
        if fpath.exists():
            raw = json.load(open(fpath))
            rounds = []
            for k in sorted(raw.keys()):
                v = raw[k]
                if v.get('status') == 'completed':
                    rounds.append(v)
            data[strategy] = rounds
    return data

def load_v2_results():
    """Load v2 multi-seed results."""
    base = RESULTS_BASE / "self_synthesis_v2"
    data = defaultdict(list)  # strategy -> list of (seed, rounds)
    if not base.exists():
        return {}
    for d in base.iterdir():
        if not d.is_dir(): continue
        json_files = list(d.glob("*_v2.json"))
        for jf in json_files:
            try:
                raw = json.load(open(jf))
                for k, v in sorted(raw.items()):
                    if v.get('status') == 'completed':
                        strategy = v['strategy']
                        seed = v['seed']
                        data[strategy].append(v)
            except: pass

    # Group by strategy
    result = {}
    for strategy, entries in data.items():
        # Group by seed
        seed_data = defaultdict(list)
        for e in entries:
            seed_data[e['seed']].append(e)
        result[strategy] = dict(seed_data)
    return result

def plot_v1_collapse(data):
    """Plot v1 collapse curve."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for strategy in ['greedy', 'qd', 'random']:
        if strategy in data:
            rounds = data[strategy]
            r_nums = [r['round'] for r in rounds]
            accs = [r['accuracy'] * 100 for r in rounds]
            ax.plot(r_nums, accs, marker='o', color=COLORS[strategy],
                   label=LABELS[strategy], linewidth=2, markersize=8)
    # Add base line
    ax.axhline(y=84.6, color='gray', linestyle='--', alpha=0.5, label='Base (84.6%)')
    ax.set_xlabel('Self-Synthesis Round')
    ax.set_ylabel('GSM8K Accuracy (%)')
    ax.set_title('7B Self-Synthesis v1: Accuracy Dynamics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(FIGURES_DIR / 'v1_collapse.pdf')
    plt.savefig(FIGURES_DIR / 'v1_collapse.png')
    print("Saved: v1_collapse.pdf/png")
    plt.close()

def plot_v2_collapse(v2_data):
    """Plot v2 collapse curve with mean±std."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Accuracy by round (mean ± std)
    ax = axes[0]
    for strategy in ['greedy', 'qd', 'random', 'qd_no_surprisal']:
        if strategy not in v2_data:
            continue
        seed_data = v2_data[strategy]
        # Collect (round -> [accuracies across seeds])
        round_accs = defaultdict(list)
        for seed, rounds in seed_data.items():
            for r in rounds:
                round_accs[r['round']].append(r['accuracy'] * 100)

        if not round_accs: continue
        r_nums = sorted(round_accs.keys())
        means = [np.mean(round_accs[r]) for r in r_nums]
        stds = [np.std(round_accs[r]) for r in r_nums]

        ax.errorbar(r_nums, means, yerr=stds, marker='o', color=COLORS[strategy],
                   label=f"{LABELS[strategy]} (n={len(seed_data)})",
                   linewidth=2, markersize=8, capsize=5)

    ax.axhline(y=84.6, color='gray', linestyle='--', alpha=0.5, label='Base')
    ax.set_xlabel('Self-Synthesis Round')
    ax.set_ylabel('GSM8K Accuracy (%)')
    ax.set_title('v2: Accuracy (Mean ± Std)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Cell coverage by round
    ax = axes[1]
    for strategy in ['greedy', 'qd', 'random', 'qd_no_surprisal']:
        if strategy not in v2_data:
            continue
        seed_data = v2_data[strategy]
        round_cells = defaultdict(list)
        for seed, rounds in seed_data.items():
            for r in rounds:
                round_cells[r['round']].append(r.get('n_cells_generated', 0))

        if not round_cells: continue
        r_nums = sorted(round_cells.keys())
        means = [np.mean(round_cells[r]) for r in r_nums]
        stds = [np.std(round_cells[r]) for r in r_nums]

        ax.errorbar(r_nums, means, yerr=stds, marker='s', color=COLORS[strategy],
                   label=LABELS[strategy], linewidth=2, markersize=8, capsize=5)

    ax.set_xlabel('Self-Synthesis Round')
    ax.set_ylabel('Unique Cells Generated')
    ax.set_title('v2: Cell Coverage (Mean ± Std)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(FIGURES_DIR / 'v2_collapse.pdf')
    plt.savefig(FIGURES_DIR / 'v2_collapse.png')
    print("Saved: v2_collapse.pdf/png")
    plt.close()

def plot_v2_diversity(v2_data):
    """Plot v2 diversity metrics."""
    metrics = [
        ('gen_entropy', 'Shannon Entropy', 'Entropy of Cell Distribution'),
        ('gen_strategies', 'Unique Strategies', 'Solution Strategy Diversity'),
        ('gen_vocab_diversity', 'Vocab Diversity', 'Vocabulary Diversity Ratio'),
        ('sel_entropy', 'Selection Entropy', 'Entropy of Selected Data'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (metric_key, ylabel, title) in enumerate(metrics):
        ax = axes[idx]
        for strategy in ['greedy', 'qd', 'random', 'qd_no_surprisal']:
            if strategy not in v2_data:
                continue
            seed_data = v2_data[strategy]
            round_vals = defaultdict(list)
            for seed, rounds in seed_data.items():
                for r in rounds:
                    val = r.get(metric_key, 0)
                    if val:
                        round_vals[r['round']].append(val)

            if not round_vals: continue
            r_nums = sorted(round_vals.keys())
            means = [np.mean(round_vals[r]) for r in r_nums]
            stds = [np.std(round_vals[r]) for r in r_nums]

            ax.errorbar(r_nums, means, yerr=stds, marker='o', color=COLORS[strategy],
                       label=LABELS[strategy], linewidth=2, markersize=6, capsize=3)

        ax.set_xlabel('Round')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.savefig(FIGURES_DIR / 'v2_diversity.pdf')
    plt.savefig(FIGURES_DIR / 'v2_diversity.png')
    print("Saved: v2_diversity.pdf/png")
    plt.close()

def plot_surprisal_ablation(v2_data):
    """Compare QD vs QD-no-surprisal."""
    if 'qd' not in v2_data or 'qd_no_surprisal' not in v2_data:
        print("Skipping surprisal ablation plot (missing data)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax_idx, (metric, ylabel, title) in enumerate([
        ('accuracy', 'Accuracy (%)', 'GSM8K Accuracy'),
        ('n_cells_generated', 'Unique Cells', 'Cell Coverage'),
        ('gen_entropy', 'Entropy', 'Distribution Entropy'),
    ]):
        ax = axes[ax_idx]
        for strategy, label, color in [('qd', 'QD (w/ Surprisal)', '#2ecc71'),
                                        ('qd_no_surprisal', 'QD (w/o Surprisal)', '#f39c12')]:
            if strategy not in v2_data: continue
            seed_data = v2_data[strategy]
            round_vals = defaultdict(list)
            for seed, rounds in seed_data.items():
                for r in rounds:
                    val = r.get(metric, 0)
                    if metric == 'accuracy': val *= 100
                    round_vals[r['round']].append(val)

            if not round_vals: continue
            r_nums = sorted(round_vals.keys())
            means = [np.mean(round_vals[r]) for r in r_nums]

            ax.plot(r_nums, means, marker='o', color=color,
                   label=label, linewidth=2, markersize=8)

        ax.set_xlabel('Round')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.savefig(FIGURES_DIR / 'v2_surprisal_ablation.pdf')
    plt.savefig(FIGURES_DIR / 'v2_surprisal_ablation.png')
    print("Saved: v2_surprisal_ablation.pdf/png")
    plt.close()

def print_v2_summary(v2_data):
    """Print text summary."""
    print(f"\n{'='*70}")
    print(f"  v2 Self-Synthesis Summary (N=1000/300, 7B)")
    print(f"{'='*70}")
    for strategy in ['greedy', 'qd', 'random', 'qd_no_surprisal']:
        if strategy not in v2_data:
            print(f"  {strategy.upper():20s}: NO DATA")
            continue
        seed_data = v2_data[strategy]
        all_rounds = []
        for seed, rounds in seed_data.items():
            all_rounds.extend(rounds)
        if not all_rounds:
            print(f"  {strategy.upper():20s}: NO COMPLETED ROUNDS")
            continue
        accs = [r['accuracy']*100 for r in all_rounds]
        cells = [r.get('n_cells_generated',0) for r in all_rounds]
        entropy = [r.get('gen_entropy',0) for r in all_rounds]
        print(f"  {strategy.upper():20s}: n_seeds={len(seed_data)}, "
              f"acc={np.mean(accs):.1f}±{np.std(accs):.1f}%, "
              f"peak={max(accs):.1f}%, "
              f"cells={np.mean(cells):.0f}±{np.std(cells):.0f}, "
              f"H={np.mean(entropy):.2f}")
    print(f"{'='*70}")

if __name__ == "__main__":
    v1_data = load_v1_results()
    v2_data = load_v2_results()

    if any(v1_data.values()):
        plot_v1_collapse(v1_data)
    if v2_data:
        print_v2_summary(v2_data)
        plot_v2_collapse(v2_data)
        plot_v2_diversity(v2_data)
        plot_surprisal_ablation(v2_data)

    if not any(v1_data.values()) and not v2_data:
        print("No data found yet. Experiments may still be running.")

    print("\nDone!")
