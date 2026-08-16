#!/bin/bash
set -euo pipefail

# Se placer dans le dossier parent de src1/ et src2/
cd "$(dirname "$0")"

#python compare/compare_groups.py
python compare/compare_groups_B.py --src1 src1 --src2 src2 --src3 src3 --seeds 42 123 999
