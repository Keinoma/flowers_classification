#!/bin/bash
set -euo pipefail

# Se placer dans le dossier parent de src1/ et src2/
cd "$(dirname "$0")"

for seed in 42 123 999; do
    echo "=== train baseline A seed $seed ==="
    python src1/main.py --seed $seed
done

for seed in 42 123 999; do
    echo "=== evaluate baseline A seed $seed ==="
    python src1/evaluate_model.py --seed $seed
done


for seed in 42 123 999; do
    echo "=== Train group A seed $seed ==="
    python src2/main_A.py --seed $seed
done

for seed in 42 123 999; do
    echo "=== Evaluate group A seed $seed ==="
    python src2/evaluate_model_A.py --seed $seed
done


for seed in 42 123 999; do
    echo "=== Train group B seed $seed ==="
    python src3/main_B.py --seed $seed
done

for seed in 42 123 999; do
    echo "=== Evaluate group B seed $seed ==="
    python src3/evaluate_model_B.py --seed $seed
done

echo "=== Entrainements terminés ==="



python compare_groups.py
python compare_groups_B.py --src1 src1 --src2 src2 --src3 src3 --seeds 42 123 999

echo "=== Comparaisons terminés ==="
