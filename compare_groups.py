#!/usr/bin/env python3
"""
compare_groups.py
Compare les résultats de la Baseline (src1) et du Groupe A (src2)
sur les 3 runs (seeds 42, 123, 999).
Génère un rapport statistique avec t-test de Student.
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import argparse


def load_metrics(src_dir, seeds):
    """Charge les métriques des runs pour un groupe."""
    results = []
    for seed in seeds:
        # Baseline = src1, Groupe A = src2
        path = Path(src_dir) / "results" / f"run_seed{seed}" / "evaluation_report.json"
        if not path.exists():
            # Fallback pour Groupe A si la structure diffère
            alt_path = Path(src_dir) / "results" / f"run_seed{seed}" / "evaluation_report_A.json"
            if alt_path.exists():
                path = alt_path
            else:
                raise FileNotFoundError(f"Rapport introuvable : {path} (ou {alt_path})")
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


def t_test_summary(baseline, groupe_a, metric_name):
    """Effectue un t-test de Student entre deux groupes."""
    x = [r[metric_name] for r in baseline]
    y = [r[metric_name] for r in groupe_a]
    t_stat, p_value = stats.ttest_ind(x, y)
    return {
        'baseline_mean': np.mean(x),
        'baseline_std': np.std(x, ddof=1),
        'groupe_a_mean': np.mean(y),
        'groupe_a_std': np.std(y, ddof=1),
        't_stat': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

def save_text_report(results, baseline, groupe_a, seeds, output_path="comparison_report.txt"):
    """Sauvegarde le rapport texte dans un fichier."""
    lines = []
    lines.append("=" * 70)
    lines.append("  COMPARAISON BASELINE vs GROUPE A")
    lines.append("=" * 70)
    lines.append(f"Seeds : {seeds}")
    lines.append("")

    metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    for metric in metrics:
        res = results[metric]
        lines.append(f"\n📊 {metric.upper()}")
        lines.append(f"   Baseline : {res['baseline_mean']:.2f}% ± {res['baseline_std']:.2f}%")
        lines.append(f"   Groupe A : {res['groupe_a_mean']:.2f}% ± {res['groupe_a_std']:.2f}%")
        lines.append(f"   t-stat   : {res['t_stat']:.3f}")
        sig = 'SIGNIFICATIF' if res['significant'] else 'Non significatif'
        lines.append(f"   p-value  : {res['p_value']:.4f} {sig} (alpha=0.05)")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"\n📝 Rapport texte sauvegarde : {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Comparaison Baseline vs Groupe A')
    parser.add_argument('--src1', default='src1', help='Dossier baseline')
    parser.add_argument('--src2', default='src2', help='Dossier groupe A')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 999])
    parser.add_argument('--output', default='comparison_report.png')
    args = parser.parse_args()

    seeds = args.seeds

    baseline = load_metrics(args.src1, seeds)
    groupe_a = load_metrics(args.src2, seeds)

    print("=" * 70)
    print("  COMPARAISON BASELINE vs GROUPE A")
    print("=" * 70)

    metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    results = {}

    for metric in metrics:
        res = t_test_summary(baseline, groupe_a, metric)
        results[metric] = res
        print(f"\n📊 {metric.upper()}")
        print(f"   Baseline : {res['baseline_mean']:.2f}% ± {res['baseline_std']:.2f}%")
        print(f"   Groupe A : {res['groupe_a_mean']:.2f}% ± {res['groupe_a_std']:.2f}%")
        print(f"   t-stat   : {res['t_stat']:.3f}")
        print(f"   p-value  : {res['p_value']:.4f} {'✅ SIGNIFICATIF' if res['significant'] else '❌ Non significatif'} (alpha=0.05)")

    save_text_report(results, baseline, groupe_a, seeds, output_path="comparison_report.txt")
    
    # Graphique comparatif
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35

    baseline_means = [results[m]['baseline_mean'] for m in metrics]
    baseline_stds = [results[m]['baseline_std'] for m in metrics]
    groupe_a_means = [results[m]['groupe_a_mean'] for m in metrics]
    groupe_a_stds = [results[m]['groupe_a_std'] for m in metrics]

    ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds, label='Baseline', capsize=5, color='steelblue')
    ax.bar(x + width/2, groupe_a_means, width, yerr=groupe_a_stds, label='Groupe A', capsize=5, color='coral')

    ax.set_ylabel('Score (%)')
    ax.set_title('Comparaison Baseline vs Groupe A (moyenne ± écart-type, 3 runs)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\n📊 Graphique sauvegardé : {args.output}")


if __name__ == "__main__":
    main()