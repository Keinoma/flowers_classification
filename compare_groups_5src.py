#!/usr/bin/env python3
"""
compare_groups_5src.py
Compare src5 aux 4 autres dossiers (src1, src2, src3, src4) sur les 3 runs
(seeds 42, 123, 999).
Genere un rapport statistique avec t-test de Student pour chaque paire
(src5 vs src1, src5 vs src2, src5 vs src3, src5 vs src4).
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import argparse


def load_metrics(src_dir, seeds, report_name="evaluation_report"):
    """Charge les metriques des runs pour un dossier src."""
    results = []
    for seed in seeds:
        # Essayer plusieurs noms de rapport possibles
        candidates = [
            f"{report_name}.json",
            f"{report_name}_A.json",
            f"{report_name}_B.json",
            f"{report_name}_C.json",
        ]
        path = None
        for candidate in candidates:
            p = Path(src_dir) / "results" / f"run_seed{seed}" / candidate
            if p.exists():
                path = p
                break

        if path is None:
            raise FileNotFoundError(
                f"Rapport introuvable pour {src_dir}, seed {seed}. "
                f"Chemins testes : {[str(Path(src_dir) / 'results' / f'run_seed{seed}' / c) for c in candidates]}"
            )

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


def t_test_summary(group1, group2, metric_name, name1="Src1", name2="Src2"):
    """Effectue un t-test de Student entre deux dossiers src."""
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


def print_comparison(results, metric, name1, name2):
    """Affiche les resultats d'une comparaison."""
    res = results[metric]
    print(f"\n📊 {metric.upper()}")
    print(f"   {name1:12s} : {res[f'{name1}_mean']:.2f}% ± {res[f'{name1}_std']:.2f}%")
    print(f"   {name2:12s} : {res[f'{name2}_mean']:.2f}% ± {res[f'{name2}_std']:.2f}%")
    print(f"   t-stat       : {res['t_stat']:.3f}")
    sig = '✅ SIGNIFICATIF' if res['significant'] else '❌ Non significatif'
    print(f"   p-value      : {res['p_value']:.4f} {sig} (alpha=0.05)")


def save_text_report(all_comparisons, all_groups, group_names, seeds,
                     ref_name, output_path="comparison_5src_report.txt"):
    """Sauvegarde le rapport texte complet dans un fichier."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"  COMPARAISON {ref_name} vs {', '.join(n for n in group_names if n != ref_name)}")
    lines.append("=" * 70)
    lines.append(f"Seeds : {seeds}")
    lines.append("")

    metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

    for other_name, results in all_comparisons.items():
        lines.append("\n" + "-" * 70)
        lines.append(f"  {ref_name} vs {other_name}")
        lines.append("-" * 70)
        for metric in metrics:
            res = results[metric]
            lines.append(f"\n{metric.upper()}")
            lines.append(f"   {other_name:8s} : {res[f'{other_name}_mean']:.2f}% ± {res[f'{other_name}_std']:.2f}%")
            lines.append(f"   {ref_name:8s} : {res[f'{ref_name}_mean']:.2f}% ± {res[f'{ref_name}_std']:.2f}%")
            lines.append(f"   t-stat   : {res['t_stat']:.3f}")
            sig = 'SIGNIFICATIF' if res['significant'] else 'Non significatif'
            lines.append(f"   p-value  : {res['p_value']:.4f} {sig} (α=0.05)")

    # Tableau recapitulatif
    lines.append("\n" + "=" * 70)
    lines.append("  TABLEAU RECAPITULATIF")
    lines.append("=" * 70)
    header = "".join(f"{name:>12s}" for name in group_names)
    lines.append(f"{'Metrique':<20s}{header}")
    lines.append("-" * (20 + 12 * len(group_names)))
    for metric in metrics:
        row = "".join(f"{np.mean([r[metric] for r in all_groups[name]]):>11.2f}%" for name in group_names)
        lines.append(f"{metric:<20s}{row}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"\n📝 Rapport texte sauvegarde : {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Comparaison src5 vs src1, src2, src3, src4')
    parser.add_argument('--src1', default='src1', help='Dossier src1')
    parser.add_argument('--src2', default='src2', help='Dossier src2')
    parser.add_argument('--src3', default='src3', help='Dossier src3')
    parser.add_argument('--src4', default='src4', help='Dossier src4')
    parser.add_argument('--src5', default='src5', help='Dossier src5 (reference de la comparaison)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 999])
    parser.add_argument('--output', default='comparison_5src_report.png')
    args = parser.parse_args()

    seeds = args.seeds

    # Dossiers src et leurs noms, dans l'ordre
    src_paths = {
        'Src1': args.src1,
        'Src2': args.src2,
        'Src3': args.src3,
        'Src4': args.src4,
        'Src5': args.src5,
    }
    group_names = list(src_paths.keys())
    ref_name = 'Src5'  # dossier de reference, compare a tous les autres

    # Chargement des metriques
    all_groups = {name: load_metrics(path, seeds) for name, path in src_paths.items()}

    print("=" * 70)
    print(f"  COMPARAISON {ref_name} vs {', '.join(n for n in group_names if n != ref_name)}")
    print("=" * 70)

    metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

    # ============================================================
    # 1. Src5 vs chacun des autres dossiers
    # ============================================================
    all_comparisons = {}
    for i, other_name in enumerate([n for n in group_names if n != ref_name], start=1):
        print("\n" + "─" * 70)
        print(f"  {i}️⃣  {ref_name} vs {other_name}")
        print("─" * 70)

        results = {}
        for metric in metrics:
            res = t_test_summary(all_groups[other_name], all_groups[ref_name], metric, other_name, ref_name)
            results[metric] = res
            print_comparison(results, metric, other_name, ref_name)
        all_comparisons[other_name] = results

    # ============================================================
    # 2. TABLEAU RECAPITULATIF
    # ============================================================
    print("\n" + "=" * 70)
    print("  TABLEAU RECAPITULATIF")
    print("=" * 70)
    header = "".join(f"{name:>12s}" for name in group_names)
    print(f"\n{'Metrique':<20s}{header}")
    print("─" * (20 + 12 * len(group_names)))
    for metric in metrics:
        row = "".join(f"{np.mean([r[metric] for r in all_groups[name]]):>11.2f}%" for name in group_names)
        print(f"{metric:<20s}{row}")

    # ============================================================
    # 3. INTERPRETATION
    # ============================================================
    print("\n" + "=" * 70)
    print("  INTERPRETATION")
    print("=" * 70)

    for other_name in [n for n in group_names if n != ref_name]:
        res = all_comparisons[other_name]['accuracy']
        if res['significant'] and res[f'{ref_name}_mean'] > res[f'{other_name}_mean']:
            print(f"✅ {ref_name} >> {other_name} : gain SIGNIFICATIF en accuracy.")
        elif res['significant'] and res[f'{ref_name}_mean'] < res[f'{other_name}_mean']:
            print(f"⚠️  {ref_name} << {other_name} : {other_name} surpasse significativement {ref_name}.")
        else:
            print(f"❌ {ref_name} ≈ {other_name} : pas de difference significative.")

    # ============================================================
    # 4. GRAPHIQUE COMPARATIF (5 dossiers)
    # ============================================================
    fig, ax = plt.subplots(figsize=(15, 6))
    x = np.arange(len(metrics))
    n_groups = len(group_names)
    width = 0.8 / n_groups
    offsets = [(-((n_groups - 1) / 2) + i) * width for i in range(n_groups)]
    colors = ['steelblue', 'coral', 'mediumseagreen', 'goldenrod', 'mediumpurple']

    means_by_group = {name: [np.mean([r[m] for r in all_groups[name]]) for m in metrics] for name in group_names}
    stds_by_group = {name: [np.std([r[m] for r in all_groups[name]], ddof=1) for m in metrics] for name in group_names}

    for offset, name, color in zip(offsets, group_names, colors):
        ax.bar(x + offset, means_by_group[name], width, yerr=stds_by_group[name],
               label=name, capsize=3, color=color, edgecolor='black', linewidth=0.5)
        for i in range(len(metrics)):
            ax.text(x[i] + offset, means_by_group[name][i] + stds_by_group[name][i] + 1,
                    f'{means_by_group[name][i]:.1f}', ha='center', fontsize=8)

    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title(f'Comparison {", ".join(group_names)}\n(mean ± standard deviation, 3 runs)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    save_text_report(all_comparisons, all_groups, group_names, seeds, ref_name,
                     output_path="comparison_5src_report.txt")

    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\n📊 Graphique sauvegarde : {args.output}")


if __name__ == "__main__":
    main()