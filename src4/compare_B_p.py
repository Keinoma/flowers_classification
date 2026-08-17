#!/usr/bin/env python3
"""
compare_B_p.py
Compare le Groupe B' (Transfer Learning sans augmentation) au Groupe B,
a la Baseline (src1) et au Groupe A (src2) sur les 3 runs.
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import argparse


def load_metrics(src_dir, seeds, report_name="evaluation_report"):
    results = []
    for seed in seeds:
        candidates = [
            f"{report_name}.json",
            f"{report_name}_A.json",
            f"{report_name}_B.json",
            f"{report_name}_B_p.json",
        ]
        path = None
        for candidate in candidates:
            p = Path(src_dir) / "results" / f"run_seed{seed}" / candidate
            if p.exists():
                path = p
                break
        
        if path is None:
            raise FileNotFoundError(f"Rapport introuvable pour {src_dir}, seed {seed}")
        
        with open(path) as f:
            data = json.load(f)
        metrics = data.get('classification_metrics', {})
        results.append({
            'seed': seed,
            'accuracy': metrics.get('accuracy_percent', 0),
            'f1_macro': metrics.get('f1_score', {}).get('macro', 0) * 100,
            'precision_macro': metrics.get('precision', {}).get('macro', 0) * 100,
            'recall_macro': metrics.get('recall', {}).get('macro', 0) * 100,
        })
    return results


def t_test_summary(group1, group2, metric_name, name1="G1", name2="G2"):
    x = [r[metric_name] for r in group1]
    y = [r[metric_name] for r in group2]
    t_stat, p_value = stats.ttest_ind(x, y)
    return {
        f'{name1}_mean': np.mean(x),
        f'{name1}_std': np.std(x, ddof=1),
        f'{name2}_mean': np.mean(y),
        f'{name2}_std': np.std(y, ddof=1),
        't_stat': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def main():
    parser = argparse.ArgumentParser(description="Comparaison Groupe B'")
    parser.add_argument('--src1', default='src1', help='Baseline')
    parser.add_argument('--src2', default='src2', help='Groupe A')
    parser.add_argument('--src3', default='src3', help='Groupe B')
    parser.add_argument('--src4', default='src4', help="Groupe B'")
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 999])
    parser.add_argument('--output', default='comparison_B_p_report.png')
    args = parser.parse_args()

    seeds = args.seeds
    baseline = load_metrics(args.src1, seeds)
    groupe_a = load_metrics(args.src2, seeds)
    groupe_b = load_metrics(args.src3, seeds)
    groupe_bp = load_metrics(args.src4, seeds)

    print("=" * 70)
    print("  COMPARAISON GROUPE B' — TRANSFER LEARNING SANS AUGMENTATION")
    print("=" * 70)

    metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

    # B' vs Baseline
    print("\n--- GROUPE B' vs BASELINE ---")
    results_bp_vs_base = {}
    for metric in metrics:
        res = t_test_summary(baseline, groupe_bp, metric, "Baseline", "Groupe_Bp")
        results_bp_vs_base[metric] = res
        print(f"{metric:12s}: Baseline {res['Baseline_mean']:.2f}% | B' {res['Groupe_Bp_mean']:.2f}% | p={res['p_value']:.4f}")

    # B' vs A
    print("\n--- GROUPE B' vs GROUPE A ---")
    results_bp_vs_a = {}
    for metric in metrics:
        res = t_test_summary(groupe_a, groupe_bp, metric, "Groupe_A", "Groupe_Bp")
        results_bp_vs_a[metric] = res
        print(f"{metric:12s}: A {res['Groupe_A_mean']:.2f}% | B' {res['Groupe_Bp_mean']:.2f}% | p={res['p_value']:.4f}")

    # B' vs B
    print("\n--- GROUPE B' vs GROUPE B ---")
    results_bp_vs_b = {}
    for metric in metrics:
        res = t_test_summary(groupe_b, groupe_bp, metric, "Groupe_B", "Groupe_Bp")
        results_bp_vs_b[metric] = res
        print(f"{metric:12s}: B {res['Groupe_B_mean']:.2f}% | B' {res['Groupe_Bp_mean']:.2f}% | p={res['p_value']:.4f}")

    # Tableau récap
    print("\n" + "=" * 70)
    print("TABLEAU RECAPITULATIF")
    print("=" * 70)
    print(f"{'Metrique':<20s} {'Baseline':>10s} {'Groupe A':>10s} {'Groupe B':>10s} {'Groupe Bp':>10s}")
    print("-" * 60)
    for metric in metrics:
        b = np.mean([r[metric] for r in baseline])
        a = np.mean([r[metric] for r in groupe_a])
        b2 = np.mean([r[metric] for r in groupe_b])
        bp = np.mean([r[metric] for r in groupe_bp])
        print(f"{metric:<20s} {b:>9.2f}% {a:>9.2f}% {b2:>9.2f}% {bp:>9.2f}%")

    # Graphique
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.2

    baseline_means = [np.mean([r[m] for r in baseline]) for m in metrics]
    baseline_stds  = [np.std([r[m] for r in baseline], ddof=1) for m in metrics]
    groupe_a_means = [np.mean([r[m] for r in groupe_a]) for m in metrics]
    groupe_a_stds  = [np.std([r[m] for r in groupe_a], ddof=1) for m in metrics]
    groupe_b_means = [np.mean([r[m] for r in groupe_b]) for m in metrics]
    groupe_b_stds  = [np.std([r[m] for r in groupe_b], ddof=1) for m in metrics]
    groupe_bp_means = [np.mean([r[m] for r in groupe_bp]) for m in metrics]
    groupe_bp_stds  = [np.std([r[m] for r in groupe_bp], ddof=1) for m in metrics]

    ax.bar(x - 1.5*width, baseline_means, width, yerr=baseline_stds, label='Baseline', capsize=3, color='steelblue', edgecolor='black')
    ax.bar(x - 0.5*width, groupe_a_means, width, yerr=groupe_a_stds, label='Groupe A', capsize=3, color='coral', edgecolor='black')
    ax.bar(x + 0.5*width, groupe_b_means, width, yerr=groupe_b_stds, label='Groupe B', capsize=3, color='mediumseagreen', edgecolor='black')
    ax.bar(x + 1.5*width, groupe_bp_means, width, yerr=groupe_bp_stds, label="Groupe B'", capsize=3, color='mediumpurple', edgecolor='black')

    ax.set_ylabel('Score (%)')
    ax.set_title("Comparaison des 4 groupes (moyenne ± ecart-type, 3 runs)")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\n📊 Graphique sauvegarde : {args.output}")


if __name__ == "__main__":
    main()