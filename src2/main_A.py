"""
Script principal d'orchestration.
Lance le chargement, l'entraînement, la sauvegarde et l'évaluation.
"""

import os
import torch

from load_data_A import load_flower_data, LoadData
from model_A import create_model
from train_A import Trainer, prepare_dataloaders
from pathlib import Path

import random
import numpy as np

# ═══════════════════════════════════════════════════════════════
# SEEDS — Reproductibilité stricte pour l'Ablation Study
# ═══════════════════════════════════════════════════════════════
import argparse

parser = argparse.ArgumentParser(description='Entraînement Baseline')
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


BASE_DIR = Path(__file__).resolve().parent.parent  # remonte de src2 vers code
SRC_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = SRC_DIR / "checkpoints" / f"run_seed{SEED}"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = BASE_DIR / 'data' / '5_classes'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
NUM_WORKERS = 4  


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  🌸 CLASSIFICATION DE FLEURS - ENTRAÎNEMENT COMPLET 🌸")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("=" * 70)

    # ── ÉTAPE 1 : Chargement des données ──────────────────────
    print("\n[1/5] 📁 Chargement des images...")
    images, labels, class_names = load_flower_data(DATA_PATH, image_size=(224, 224))
    # Visualisation de la distribution des classes
    loader = LoadData(DATA_PATH, image_size=(224, 224))
    loader.classes = class_names
    loader.plot_class_distribution(labels, save_path=CHECKPOINTS_DIR/"5_classes_distrib_A.png")
    print(f"   Classes: {class_names}")

    # ── ÉTAPE 2 : DataLoaders ───────────────────────────────────
    print("\n[2/5] ⚙️  Préparation des DataLoaders...")
    train_loader, val_loader, test_loader = prepare_dataloaders(
    images, labels,
    batch_size=BATCH_SIZE,
    val_split=0.2,
    test_split=0.1,
    num_workers=NUM_WORKERS
    )

    # ── ÉTAPE 3 : Modèle ────────────────────────────────────────
    print("\n[3/5] 🧠 Création du modèle...")
    model = create_model(
        num_classes=len(class_names),
        device=DEVICE
    )

    # ── ÉTAPE 4 : Entraînement ──────────────────────────────────
    print("\n[4/5] 🔥 Lancement de l'entraînement...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS
    )

    history = trainer.fit(save_dir=CHECKPOINTS_DIR)

    # ── ÉTAPE 5 : Visualisation + Évaluation ────────────────────
    print("\n[5/5] 📊 Évaluation finale et courbes...")
    model.load_state_dict(torch.load(CHECKPOINTS_DIR/"best_model_A.pth", weights_only=True)['model_state_dict'])    # Courbes d'entraînement
    trainer.plot_history(save_path=CHECKPOINTS_DIR/"train_curves_A.png")

    # Évaluation sur le test set (charge le meilleur modèle)
    print("\n" + "=" * 70)
    print("  ÉVALUATION SUR LE TEST SET")
    print("=" * 70)
    test_acc = trainer.evaluate(test_loader, class_names, save_path=CHECKPOINTS_DIR/"conf_matrix_A.png")

    # Résumé final
    print("\n" + "=" * 70)
    print("  ✅ ENTRAÎNEMENT TERMINÉ !")
    print("=" * 70)
    print(f"  Meilleure val accuracy : {trainer.best_val_acc:.2f}%")
    print(f"  Test accuracy          : {test_acc:.2f}%")
    print(f"  Modèle sauvegardé      : {CHECKPOINTS_DIR}/best_model_A.pth")
    print(f"  Courbes sauvegardées   : {CHECKPOINTS_DIR}/train_curves_A.png")
    print("=" * 70)

if __name__ == "__main__":
    main()