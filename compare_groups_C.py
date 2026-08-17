#!/usr/bin/env python3
"""
compare_groups_C.py
Compare le Groupe C (Fine-tuning) a la Baseline (src1), au Groupe A (src2)
et au Groupe B (src5) sur les 3 runs (seeds 42, 123, 999).
Genere un rapport statistique avec t-test de Student pour chaque paire.
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import argparse


def load_metrics(src_dir, seeds, report_name="evaluation_report"):
    """Charge les metriques des runs pour un groupe."""
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


def t_test_summary(group1, group2, metric_name, name1="Groupe1", name2="Groupe2"):
    """Effectue un t-test de Student entre deux groupes."""
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


def save_text_report(results_c_vs_base, results_c_vs_a, results_c_vs_b,
                     baseline, groupe_a, groupe_b, groupe_c, seeds,
                     output_path="comparison_C_report.txt"):
    """Sauvegarde le rapport texte complet dans un fichier."""
    lines = []
    lines.append("=" * 70)
    lines.append("  COMPARAISON GROUPE C — FINE-TUNING")
    lines.append("=" * 70)
    lines.append(f"Seeds : {seeds}")
    lines.append("")

    metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

    # C vs Baseline
    lines.append("\n" + "-" * 70)
    lines.append("  GROUPE C vs BASELINE")
    lines.append("-" * 70)
    for metric in metrics:
        res = results_c_vs_base[metric]
        lines.append(f"\n{metric.upper()}")
        lines.append(f"   Baseline : {res['Baseline_mean']:.2f}% ± {res['Baseline_std']:.2f}%")
        lines.append(f"   Groupe C : {res['Groupe_C_mean']:.2f}% ± {res['Groupe_C_std']:.2f}%")
        lines.append(f"   t-stat   : {res['t_stat']:.3f}")
        sig = 'SIGNIFICATIF' if res['significant'] else 'Non significatif'
        lines.append(f"   p-value  : {res['p_value']:.4f} {sig} (α=0.05)")

    # C vs A
    lines.append("\n" + "-" * 70)
    lines.append("  GROUPE C vs GROUPE A")
    lines.append("-" * 70)
    for metric in metrics:
        res = results_c_vs_a[metric]
        lines.append(f"\n{metric.upper()}")
        lines.append(f"   Groupe A : {res['Groupe_A_mean']:.2f}% ± {res['Groupe_A_std']:.2f}%")
        lines.append(f"   Groupe C : {res['Groupe_C_mean']:.2f}% ± {res['Groupe_C_std']:.2f}%")
        lines.append(f"   t-stat   : {res['t_stat']:.3f}")
        sig = 'SIGNIFICATIF' if res['significant'] else 'Non significatif'
        lines.append(f"   p-value  : {res['p_value']:.4f} {sig} (α=0.05)")

    # C vs B
    lines.append("\n" + "-" * 70)
    lines.append("  GROUPE C vs GROUPE B")
    lines.append("-" * 70)
    for metric in metrics:
        res = results_c_vs_b[metric]
        lines.append(f"\n{metric.upper()}")
        lines.append(f"   Groupe B : {res['Groupe_B_mean']:.2f}% ± {res['Groupe_B_std']:.2f}%")
        lines.append(f"   Groupe C : {res['Groupe_C_mean']:.2f}% ± {res['Groupe_C_std']:.2f}%")
        lines.append(f"   t-stat   : {res['t_stat']:.3f}")
        sig = 'SIGNIFICATIF' if res['significant'] else 'Non significatif'
        lines.append(f"   p-value  : {res['p_value']:.4f} {sig} (α=0.05)")

    # Tableau recapitulatif
    lines.append("\n" + "=" * 70)
    lines.append("  TABLEAU RECAPITULATIF")
    lines.append("=" * 70)
    lines.append(f"{'Metrique':<20s} {'Baseline':>12s} {'Groupe A':>12s} {'Groupe B':>12s} {'Groupe C':>12s}")
    lines.append("-" * 72)
    for metric in metrics:
        b_mean = np.mean([r[metric] for r in baseline])
        a_mean = np.mean([r[metric] for r in groupe_a])
        b2_mean = np.mean([r[metric] for r in groupe_b])
        c_mean = np.mean([r[metric] for r in groupe_c])
        lines.append(f"{metric:<20s} {b_mean:>11.2f}% {a_mean:>11.2f}% {b2_mean:>11.2f}% {c_mean:>11.2f}%")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"\n📝 Rapport texte sauvegarde : {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Comparaison Groupe C vs Baseline, A et B')
    parser.add_argument('--src1', default='src1', help='Dossier baseline')
    parser.add_argument('--src2', default='src2', help='Dossier groupe A')
    parser.add_argument('--src3', default='src5', help='Dossier groupe B')
    parser.add_argument('--src4', default='src4', help='Dossier groupe C')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 999])
    parser.add_argument('--output', default='comparison_C_report.png')
    args = parser.parse_args()

    seeds = args.seeds

    # Chargement des metriques
    baseline = load_metrics(args.src1, seeds)
    groupe_a = load_metrics(args.src2, seeds)
    groupe_b = load_metrics(args.src3, seeds)
    groupe_c = load_metrics(args.src4, seeds)

    print("=" * 70)
    print("  COMPARAISON GROUPE C — FINE-TUNING")
    print("=" * 70)

    metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

    # ============================================================
    # 1. C vs BASELINE
    # ============================================================
    print("\n" + "─" * 70)
    print("  1️⃣  GROUPE C vs BASELINE")
    print("  Hypothese : le fine-tuning ameliore-t-il significativement ?")
    print("─" * 70)

    results_c_vs_base = {}
    for metric in metrics:
        res = t_test_summary(baseline, groupe_c, metric, "Baseline", "Groupe_C")
        results_c_vs_base[metric] = res
        print_comparison(results_c_vs_base, metric, "Baseline", "Groupe_C")

    # ============================================================
    # 2. C vs A
    # ============================================================
    print("\n" + "─" * 70)
    print("  2️⃣  GROUPE C vs GROUPE A")
    print("  Hypothese : le fine-tuning surpasse-t-il le CNN from scratch + aug ?")
    print("─" * 70)

    results_c_vs_a = {}
    for metric in metrics:
        res = t_test_summary(groupe_a, groupe_c, metric, "Groupe_A", "Groupe_C")
        results_c_vs_a[metric] = res
        print_comparison(results_c_vs_a, metric, "Groupe_A", "Groupe_C")

    # ============================================================
    # 3. C vs B
    # ============================================================
    print("\n" + "─" * 70)
    print("  3️⃣  GROUPE C vs GROUPE B")
    print("  Hypothese : adapter les features apporte-t-il un gain par rapport aux features figees ?")
    print("─" * 70)

    results_c_vs_b = {}
    for metric in metrics:
        res = t_test_summary(groupe_b, groupe_c, metric, "Groupe_B", "Groupe_C")
        results_c_vs_b[metric] = res
        print_comparison(results_c_vs_b, metric, "Groupe_B", "Groupe_C")

    # ============================================================
    # 4. TABLEAU RECAPITULATIF
    # ============================================================
    print("\n" + "=" * 70)
    print("  TABLEAU RECAPITULATIF")
    print("=" * 70)
    print(f"\n{'Metrique':<20s} {'Baseline':>12s} {'Groupe A':>12s} {'Groupe B':>12s} {'Groupe C':>12s}")
    print("─" * 72)
    for metric in metrics:
        b_mean = np.mean([r[metric] for r in baseline])
        a_mean = np.mean([r[metric] for r in groupe_a])
        b2_mean = np.mean([r[metric] for r in groupe_b])
        c_mean = np.mean([r[metric] for r in groupe_c])
        print(f"{metric:<20s} {b_mean:>11.2f}% {a_mean:>11.2f}% {b2_mean:>11.2f}% {c_mean:>11.2f}%")

    # ============================================================
    # 5. INTERPRETATION
    # ============================================================
    print("\n" + "=" * 70)
    print("  INTERPRETATION")
    print("=" * 70)

    acc_c_vs_base = results_c_vs_base['accuracy']
    acc_c_vs_a = results_c_vs_a['accuracy']
    acc_c_vs_b = results_c_vs_b['accuracy']

    if acc_c_vs_base['significant'] and acc_c_vs_base['Groupe_C_mean'] > acc_c_vs_base['Baseline_mean']:
        print("✅ C >> Baseline : Le fine-tuning apporte un gain SIGNIFICATIF.")
    else:
        print("❌ C ≈ Baseline  : Le fine-tuning n'apporte PAS de gain significatif.")

    if acc_c_vs_a['significant'] and acc_c_vs_a['Groupe_C_mean'] > acc_c_vs_a['Groupe_A_mean']:
        print("✅ C >> A        : Le fine-tuning surpasse significativement le CNN from scratch + aug.")
    elif acc_c_vs_a['significant'] and acc_c_vs_a['Groupe_C_mean'] < acc_c_vs_a['Groupe_A_mean']:
        print("⚠️  C << A        : Le CNN from scratch + aug surpasse le fine-tuning (etrange, a verifier).")
    else:
        print("❌ C ≈ A         : Pas de difference significative entre C et A.")

    if acc_c_vs_b['significant'] and acc_c_vs_b['Groupe_C_mean'] > acc_c_vs_b['Groupe_B_mean']:
        print("✅ C >> B        : Adapter les features apporte un gain SIGNIFICATIF par rapport aux features figees.")
    elif acc_c_vs_b['significant'] and acc_c_vs_b['Groupe_C_mean'] < acc_c_vs_b['Groupe_B_mean']:
        print("⚠️  C << B        : Les features figees surpassent le fine-tuning (possible overfitting de C).")
    else:
        print("❌ C ≈ B         : Pas de difference significative. Les features generiques suffisent.")

    # ============================================================
    # 6. GRAPHIQUE COMPARATIF (4 groupes)
    # ============================================================
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(metrics))
    width = 0.20

    baseline_means = [np.mean([r[m] for r in baseline]) for m in metrics]
    baseline_stds  = [np.std([r[m] for r in baseline], ddof=1) for m in metrics]
    groupe_a_means = [np.mean([r[m] for r in groupe_a]) for m in metrics]
    groupe_a_stds  = [np.std([r[m] for r in groupe_a], ddof=1) for m in metrics]
    groupe_b_means = [np.mean([r[m] for r in groupe_b]) for m in metrics]
    groupe_b_stds  = [np.std([r[m] for r in groupe_b], ddof=1) for m in metrics]
    groupe_c_means = [np.mean([r[m] for r in groupe_c]) for m in metrics]
    groupe_c_stds  = [np.std([r[m] for r in groupe_c], ddof=1) for m in metrics]

    ax.bar(x - 1.5*width, baseline_means, width, yerr=baseline_stds,
           label='Baseline', capsize=3, color='steelblue', edgecolor='black', linewidth=0.5)
    ax.bar(x - 0.5*width, groupe_a_means, width, yerr=groupe_a_stds,
           label='Groupe A (CNN + Aug)', capsize=3, color='coral', edgecolor='black', linewidth=0.5)
    ax.bar(x + 0.5*width, groupe_b_means, width, yerr=groupe_b_stds,
           label='Groupe B (Transfer Learning)', capsize=3, color='mediumseagreen', edgecolor='black', linewidth=0.5)
    ax.bar(x + 1.5*width, groupe_c_means, width, yerr=groupe_c_stds,
           label='Groupe C (Fine-tuning)', capsize=3, color='mediumpurple', edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Comparaison Baseline vs Groupe A vs Groupe B vs Groupe C\n(moyenne ± ecart-type, 3 runs)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(bottom=0)

    # Ajouter les valeurs au-dessus des barres
    for i, (b, a, b2, c) in enumerate(zip(baseline_means, groupe_a_means, groupe_b_means, groupe_c_means)):
        ax.text(i - 1.5*width, b + baseline_stds[i] + 1, f'{b:.1f}', ha='center', fontsize=8)
        ax.text(i - 0.5*width, a + groupe_a_stds[i] + 1, f'{a:.1f}', ha='center', fontsize=8)
        ax.text(i + 0.5*width, b2 + groupe_b_stds[i] + 1, f'{b2:.1f}', ha='center', fontsize=8)
        ax.text(i + 1.5*width, c + groupe_c_stds[i] + 1, f'{c:.1f}', ha='center', fontsize=8)

    plt.tight_layout()

    save_text_report(results_c_vs_base, results_c_vs_a, results_c_vs_b,
                     baseline, groupe_a, groupe_b, groupe_c, seeds,
                     output_path="comparison_C_report.txt")

    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\n📊 Graphique sauvegarde : {args.output}")


if __name__ == "__main__":
    main()
