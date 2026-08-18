"""
Script principal d'orchestration — Groupe C (Fine-tuning).
Version Oxford 102 (src5_oxford102) : dataset téléchargé via torchvision,
split officiel train/val/test. Lance le chargement, l'entraînement,
la sauvegarde et l'évaluation.
"""

import os
import torch

from load_data_ox102 import load_flower_data, LoadDataOxford
from model_ox102 import create_model
from train_ox102 import Trainer, prepare_dataloaders
from pathlib import Path

import random
import numpy as np

# ═══════════════════════════════════════════════════════════════
# SEEDS — Reproductibilité stricte pour l'Ablation Study
# ═══════════════════════════════════════════════════════════════
import argparse
parser = argparse.ArgumentParser(description='Entraînement Groupe C — Fine-tuning')
parser.add_argument('--seed', type=int, default=42, help='Seed pour reproductibilité')
args = parser.parse_args()
SEED = args.seed

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent.parent  # remonte de src4 vers code
SRC_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = SRC_DIR / "checkpoints" / f"run_seed{SEED}"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = BASE_DIR / 'data' / 'oxford102'  # cache torchvision (téléchargement auto au 1er run)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
EPOCHS = 50              # l'early stopping (patience=7) coupera avant si la val stagne
LR_HEAD = 0.001          # LR pour la tête de classification (nouvelle)
LR_BACKBONE = 0.00001    # LR pour le backbone (10-100x plus faible)
NUM_WORKERS = 4  

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  🌸 CLASSIFICATION DE FLEURS - ENTRAÎNEMENT COMPLET 🌸")
    print("  Groupe C — Fine-tuning (ResNet50 dégelé) — Dataset Oxford 102")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"LR tête: {LR_HEAD}")
    print(f"LR backbone: {LR_BACKBONE}")
    print("=" * 70)

    # ── ÉTAPE 1 : Chargement des données ──────────────────────
    print("\n[1/5] 📁 Chargement des images...")
    images, labels, class_names, split_sizes = load_flower_data(DATA_PATH, image_size=(224, 224))
    loader = LoadDataOxford(DATA_PATH, image_size=(224, 224))
    loader.classes = class_names
    loader.plot_class_distribution(labels, save_path=CHECKPOINTS_DIR/"oxford102_distrib_ox102.png")
    print(f"   Classes: {class_names}")

    # ── ÉTAPE 2 : DataLoaders ───────────────────────────────────
    print("\n[2/5] ⚙️  Préparation des DataLoaders...")
    train_loader, val_loader, test_loader = prepare_dataloaders(
        images, labels,
        split_sizes=split_sizes,   # split officiel Oxford 102 (1020 / 1020 / 6149)
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    # ── ÉTAPE 3 : Modèle ────────────────────────────────────────
    print("\n[3/5] 🧠 Création du modèle...")
    model = create_model(
        num_classes=len(class_names),
        device=DEVICE,
        unfreeze=True
    )

    # ── ÉTAPE 4 : Entraînement ──────────────────────────────────
    print("\n[4/5] 🔥 Lancement de l'entraînement...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        lr_head=LR_HEAD,
        lr_backbone=LR_BACKBONE,
        epochs=EPOCHS,
        early_stopping_patience=7
    )

    history = trainer.fit(save_dir=CHECKPOINTS_DIR)

    # ── ÉTAPE 5 : Visualisation + Évaluation ────────────────────
    print("\n[5/5] 📊 Évaluation finale et courbes...")
    model.load_state_dict(torch.load(CHECKPOINTS_DIR/"best_model_ox102.pth", weights_only=True)['model_state_dict'])
    trainer.plot_history(save_path=CHECKPOINTS_DIR/"train_curves_ox102.png")

    # Évaluation sur le test set
    print("\n" + "=" * 70)
    print("  ÉVALUATION SUR LE TEST SET")
    print("=" * 70)
    test_acc = trainer.evaluate(test_loader, class_names, save_path=CHECKPOINTS_DIR/"conf_matrix_ox102.png")

    # Résumé final
    print("\n" + "=" * 70)
    print("  ✅ ENTRAÎNEMENT TERMINÉ !")
    print("=" * 70)
    print(f"  Meilleure val accuracy : {trainer.best_val_acc:.2f}%")
    print(f"  Test accuracy          : {test_acc:.2f}%")
    print(f"  Modèle sauvegardé      : {CHECKPOINTS_DIR}/best_model_ox102.pth")
    print(f"  Courbes sauvegardées   : {CHECKPOINTS_DIR}/train_curves_ox102.png")
    print("=" * 70)

if __name__ == "__main__":
    main()