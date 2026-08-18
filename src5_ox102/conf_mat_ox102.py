"""
conf_mat_ox102.py — Version Oxford 102 (src5_oxford102)
Recrée la matrice de confusion à partir du modèle sauvegardé (Groupe C).
Split officiel Oxford 102, matrice 102×102 sans annotations dans les cases.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from load_data_oxford import load_flower_data
from model_ox102 import create_model
from train_ox102 import prepare_dataloaders, get_transforms, FlowerDataset
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'data' / 'oxford102'  # cache torchvision Oxford 102
CHECKPOINT_PATH = SRC_DIR / 'checkpoints' / 'best_model_ox102.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
NUM_WORKERS = 4

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  🌸 RECONSTRUCTION DE LA MATRICE DE CONFUSION (GROUPE C) 🌸")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print("=" * 70)

    # ── ÉTAPE 1 : Charger les données ──
    print("\n[1/3] 📁 Chargement des images...")
    images, labels, class_names, split_sizes = load_flower_data(DATA_PATH, image_size=(224, 224))
    print(f"   Classes: {class_names}")
    print(f"   Total images: {len(images)}")

    # ── ÉTAPE 2 : Recréer les DataLoaders (seed 42 = même split) ──
    print("\n[2/3] ⚙️  Recréation des DataLoaders (split officiel Oxford 102)...")
    train_loader, val_loader, test_loader = prepare_dataloaders(
        images, labels,
        split_sizes=split_sizes,   # split officiel Oxford 102
        batch_size=BATCH_SIZE,
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
    
    # 102 classes : aucune annotation dans les cases (illisible),
    # ticks espacés tous les 10 indices — usage standard pour grandes matrices
    n_classes = len(class_names)
    plt.figure(figsize=(18, 16))
    plt.imshow(cm, cmap='Blues')
    plt.title('Matrice de Confusion (Groupe C — Oxford 102)', fontsize=14, fontweight='bold')
    plt.colorbar()

    tick_marks = np.arange(0, n_classes, 10)
    plt.xticks(tick_marks, tick_marks, rotation=90, fontsize=7)
    plt.yticks(tick_marks, tick_marks, fontsize=7)
    plt.xlabel('Prédit (indice de classe)', fontsize=12)
    plt.ylabel('Réel (indice de classe)', fontsize=12)

    plt.tight_layout()
    
    # Sauvegarde
    save_path = SRC_DIR / 'checkpoints' / 'conf_mat_ox102.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✅ Matrice de confusion sauvegardée: {save_path}")
    plt.close()

if __name__ == "__main__":
    main()  