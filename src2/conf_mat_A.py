"""
reproduce_confusion_matrix.py
Recrée la matrice de confusion à partir du modèle sauvegardé.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from load_data_A import load_flower_data, LoadData
from model_A import create_model
from train_A import prepare_dataloaders, get_transforms, FlowerDataset
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION (même que main.py)
# ═══════════════════════════════════════════════════════════════
import argparse
parser = argparse.ArgumentParser(description='Reconstruction matrice de confusion Groupe A')
parser.add_argument('--seed', type=int, default=42, help='Seed du run')
args = parser.parse_args()

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'data' / '5_classes'
CHECKPOINT_PATH = SRC_DIR / 'checkpoints' / f'run_seed{args.seed}' / 'best_model_A.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
NUM_WORKERS = 4

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  🌸 RECONSTRUCTION DE LA MATRICE DE CONFUSION 🌸")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print("=" * 70)

    # ── ÉTAPE 1 : Charger les données (même ordre que l'entraînement) ──
    print("\n[1/3] 📁 Chargement des images...")
    images, labels, class_names = load_flower_data(DATA_PATH, image_size=(224, 224))
    print(f"   Classes: {class_names}")
    print(f"   Total images: {len(images)}")

    # ── ÉTAPE 2 : Recréer les DataLoaders (seed 42 = même split) ──
    print("\n[2/3] ⚙️  Recréation des DataLoaders (seed=42)...")
    train_loader, val_loader, test_loader = prepare_dataloaders(
        images, labels,
        batch_size=BATCH_SIZE,
        val_split=0.2,
        test_split=0.1,
        num_workers=NUM_WORKERS
    )
    print(f"   Test set: {len(test_loader.dataset)} images")

    # ── ÉTAPE 3 : Charger le modèle ──
    print("\n[3/3] 🧠 Chargement du modèle...")
    model = create_model(num_classes=len(class_names), device=DEVICE)
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\n❌ ERREUR: Le checkpoint n'existe pas: {CHECKPOINT_PATH}")
        print("   Adapte la variable CHECKPOINT_PATH dans le script.")
        return

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"   Modèle chargé (epoch {checkpoint.get('epoch', 'N/A')}, val_acc: {checkpoint.get('val_acc', 'N/A'):.2f}%)")

    # ── ÉTAPE 4 : Inférence sur le test set ──
    print("\n" + "=" * 70)
    print("  INFÉRENCE SUR LE TEST SET")
    print("=" * 70)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images_batch, labels_batch in tqdm(test_loader, desc="Test"):
            images_batch = images_batch.to(DEVICE)
            outputs = model(images_batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.numpy())

    # ── ÉTAPE 5 : Métriques ──
    acc = accuracy_score(all_labels, all_preds) * 100
    print(f"\n{'='*70}")
    print(f"  RÉSULTATS")
    print(f"{'='*70}")
    print(f"  Accuracy: {acc:.2f}%")
    print(f"\n  Rapport de classification:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # ── ÉTAPE 6 : Matrice de confusion ──
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap='Blues')
    plt.title('Matrice de Confusion', fontsize=14, fontweight='bold')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Prédit', fontsize=12)
    plt.ylabel('Réel', fontsize=12)
    
    # Annotations avec couleur adaptative
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', 
                     color=color, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Sauvegarde
    save_path = Path(CHECKPOINT_PATH).parent / 'conf_mat_1.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✅ Matrice de confusion sauvegardée: {save_path}")
    plt.close()

if __name__ == "__main__":
    main()