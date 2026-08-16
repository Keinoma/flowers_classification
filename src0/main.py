"""
Script principal d'orchestration.
Lance le chargement, l'entraînement, la sauvegarde et l'évaluation.
"""

import os
import torch

from load_data import load_flower_data, LoadData
from model import create_model
from train import Trainer, prepare_dataloaders
from pathlib import Path


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════


BASE_DIR = Path(__file__).resolve().parent.parent  # remonte de src0 vers code
DATA_PATH = BASE_DIR / 'data' / '5_classes'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
EPOCHS = 2
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
    loader.plot_class_distribution(labels, save_path="5_classes_distribution.png")
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
        freeze_backbone=True,
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
    
    history = trainer.fit(save_dir="checkpoints")
    
    # ── ÉTAPE 5 : Visualisation + Évaluation ────────────────────
    print("\n[5/5] 📊 Évaluation finale et courbes...")
    
    # Courbes d'entraînement
    trainer.plot_history(save_path="checkpoints/training_curves.png")
    
    # Évaluation sur le test set (charge le meilleur modèle)
    print("\n" + "=" * 70)
    print("  ÉVALUATION SUR LE TEST SET")
    print("=" * 70)
    test_acc = trainer.evaluate(test_loader, class_names)
    
    # Résumé final
    print("\n" + "=" * 70)
    print("  ✅ ENTRAÎNEMENT TERMINÉ !")
    print("=" * 70)
    print(f"  Meilleure val accuracy : {trainer.best_val_acc:.2f}%")
    print(f"  Test accuracy          : {test_acc:.2f}%")
    print(f"  Modèle sauvegardé      : checkpoints/best_model.pth")
    print(f"  Courbes sauvegardées   : checkpoints/training_curves.png")
    print("=" * 70)

if __name__ == "__main__":
    main()