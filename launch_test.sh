#!/bin/bash
set -euo pipefail

# Se placer dans le dossier parent de src1/ et src2/
cd "$(dirname "$0")"


# POUR LE GROUPE C : Fine tuning
for seed in 42 123 999; do
    echo "=== train baseline C seed $seed ==="
    python src5/main_C.py --seed $seed
done

for seed in 42 123 999; do
    echo "=== evaluate baseline C seed $seed ==="
    python src5/evaluate_model_C.py --seed $seed
done
echo "=== Entrainements terminés ==="

python compare_groups_C.py
echo "=== Comparaisons terminés ==="



# POUR LE GROUPE B' : ResNet mais sans data-augm
for seed in 42 123 999; do
    echo "=== train baseline B_p seed $seed ==="
    python src4/main_B_p.py --seed $seed
done

for seed in 42 123 999; do
    echo "=== evaluate baseline B_p seed $seed ==="
    python src4/evaluate_model_B_p.py --seed $seed
done
echo "=== Entrainements terminés ==="

python compare_groups_B_p.py
echo "=== Comparaisons terminés ==="
