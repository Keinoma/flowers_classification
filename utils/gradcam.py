"""
gradcam.py
Grad-CAM pur PyTorch, zéro dépendance externe.
Usage:
    # Une seule image
    python gradcam.py data/5_classes/Orchid/Or1.jpg
    python gradcam.py data/5_classes/Orchid/Or1.jpg --target_class Lilly

    # Tout un dossier (grille)
    python gradcam.py data/5_classes/Orchid
    python gradcam.py data/5_classes/Orchid --max 12
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from torchvision import transforms
from glob import glob
from torch.utils.data import DataLoader, random_split
from train_A import FlowerDataset, get_transforms
from load_data_A import load_flower_data

from model_A import create_model

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASS_NAMES = ['Lilly', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']
IMAGE_SIZE = 224

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = SRC_DIR / 'checkpoints' / 'run_seed42' / 'best_model_A.pth'
# Mêmes transforms que predict.py
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ═══════════════════════════════════════════════════════════════
# CLASSE GRAD-CAM
# ═══════════════════════════════════════════════════════════════

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None
        self.hook = target_layer.register_forward_hook(self.save_features)

    def save_features(self, module, input, output):
        self.features = output
        output.register_hook(self.save_gradients)

    def save_gradients(self, grad):
        self.gradients = grad

    def generate(self, img_tensor, target_class):
        self.model.eval()
        self.gradients = None
        self.features = None

        # FORCER l'activation des gradients pour Grad-CAM
        with torch.enable_grad():
            output = self.model(img_tensor)
            target_score = output[0, target_class]
            self.model.zero_grad()
            target_score.backward()

        if self.gradients is None or self.features is None:
            raise RuntimeError("Gradients ou features non capturés. Vérifie que target_layer est bien la dernière couche conv.")

        # Moyenne des gradients par canal (Global Average Pooling spatial)
        pooled_grads = torch.mean(self.gradients[0], dim=(1, 2))

        # Pondération des feature maps par les gradients
        cam = torch.zeros_like(self.features[0, 0])
        for i, w in enumerate(pooled_grads):
            cam += w * self.features[0, i]

        # ReLU + normalisation
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.detach().cpu().numpy()

    def remove_hook(self):
        if hasattr(self, 'hook') and self.hook is not None:
            self.hook.remove()
            self.hook = None
# ═══════════════════════════════════════════════════════════════
# UTILITAIRES
# ═══════════════════════════════════════════════════════════════

def overlay_heatmap(org_img_pil, cam, alpha=0.5):
    """Superpose la heatmap Grad-CAM sur l'image PIL originale."""
    org_w, org_h = org_img_pil.size
    cam = cv2.resize(cam, (org_w, org_h))
    cam = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    org_img = np.array(org_img_pil)
    org_img = cv2.cvtColor(org_img, cv2.COLOR_RGB2BGR)

    superimposed = cv2.addWeighted(org_img, 1 - alpha, heatmap, alpha, 0)
    return cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)


def predict_single(model, img_path):
    """Prédit la classe d'une image et retourne (pred_class, probs)."""
    org_img = Image.open(img_path).convert('RGB')
    input_tensor = transform(org_img).unsqueeze(0).to(DEVICE)
    outputs = model(input_tensor)
    probs = torch.softmax(outputs, dim=1)[0]
    pred_class = torch.argmax(probs).item()
    return org_img, input_tensor, pred_class, probs


# ═══════════════════════════════════════════════════════════════
# MODE FICHIER UNIQUE (comportement existant)
# ═══════════════════════════════════════════════════════════════

def run_single(image_path, model, target_class_name=None):
    org_img, input_tensor, pred_class, probs = predict_single(model, image_path)

    print(f"\n📷 Image: {image_path}")
    print(f"🔮 Prédite : {CLASS_NAMES[pred_class]} ({probs[pred_class]*100:.1f}%)")
    print("Probabilités:")
    for i, (cls, prob) in enumerate(zip(CLASS_NAMES, probs.cpu().numpy())):
        marker = " ←" if i == pred_class else ""
        print(f"  {cls:12s}: {prob*100:.1f}%{marker}")

    # Déterminer les classes cibles pour Grad-CAM
    targets = [("Prédite", pred_class)]

    true_class_name = os.path.basename(os.path.dirname(image_path))
    if true_class_name in CLASS_NAMES:
        true_idx = CLASS_NAMES.index(true_class_name)
        if true_idx != pred_class:
            targets.append(("Réelle", true_idx))

    if target_class_name:
        if target_class_name not in CLASS_NAMES:
            print(f"❌ Classe inconnue: {target_class_name}. Choix: {CLASS_NAMES}")
            sys.exit(1)
        manual_idx = CLASS_NAMES.index(target_class_name)
        if manual_idx not in [t[1] for t in targets]:
            targets.append((f"Manuelle ({target_class_name})", manual_idx))

    # Générer Grad-CAM
    gradcam = GradCAM(model, model.conv4)

    n_cols = 1 + len(targets)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    if n_cols == 1:
        axes = [axes]

    # Image originale
    axes[0].imshow(org_img)
    axes[0].set_title("Image originale")
    axes[0].axis('off')

    # Heatmaps
    for ax_idx, (title_suffix, target_class) in enumerate(targets, start=1):
        cam = gradcam.generate(input_tensor, target_class)
        superimposed = overlay_heatmap(org_img, cam, alpha=0.5)

        axes[ax_idx].imshow(superimposed)
        axes[ax_idx].set_title(f"Grad-CAM: {CLASS_NAMES[target_class]}\n({title_suffix})")
        axes[ax_idx].axis('off')

    plt.tight_layout()
    save_name = f"gradcam_{Path(image_path).stem}.png"
    save_path = BASE_DIR / save_name
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Grad-CAM sauvegardé: {save_path}")
    plt.close()

    gradcam.remove_hook()


# ═══════════════════════════════════════════════════════════════
# MODE DOSSIER (grille de heatmaps)
# ═══════════════════════════════════════════════════════════════

def run_folder(folder_path, model, max_images=20):
    """Génère une grille Grad-CAM pour toutes les images d'un dossier."""
    folder = Path(folder_path)
    image_paths = set()
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
        image_paths.update(folder.glob(ext))
    image_paths = sorted(image_paths)[:max_images]

    if not image_paths:
        print(f"❌ Aucune image trouvée dans: {folder_path}")
        sys.exit(1)

    print(f"\n📁 Dossier: {folder_path}")
    print(f"🖼️  Images trouvées: {len(image_paths)} (affichage limité à {max_images})")

    # Grille : 4 colonnes
    n = len(image_paths)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1:
        axes = np.array([axes]).reshape(1, -1) if cols > 1 else np.array([[axes]])
    axes = axes.flatten()

    gradcam = GradCAM(model, model.conv4)

    for idx, img_path in enumerate(image_paths):
        ax = axes[idx]

        try:
            org_img, input_tensor, pred_class, probs = predict_single(model, str(img_path))
            cam = gradcam.generate(input_tensor, pred_class)
            superimposed = overlay_heatmap(org_img, cam, alpha=0.5)

            true_name = os.path.basename(os.path.dirname(str(img_path)))
            pred_name = CLASS_NAMES[pred_class]
            conf = probs[pred_class] * 100

            # Titre avec prédiction + confiance
            title = f"Prédit: {pred_name} ({conf:.0f}%)"
            if true_name != pred_name:
                title += f"\nRéel: {true_name}"

            ax.imshow(superimposed)
            ax.set_title(title, fontsize=9, color='darkred' if true_name != pred_name else 'darkgreen')
            ax.axis('off')

        except Exception as e:
            ax.set_title(f"Erreur\n{img_path.name}")
            ax.axis('off')
            print(f"⚠️ Erreur sur {img_path}: {e}")

    # Cacher les axes vides
    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    save_name = f"gradcam_batch_{folder.name}.png"
    save_path = SRC_DIR / save_name
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Grille sauvegardée: {save_path}")
    plt.close()

    gradcam.remove_hook()


def run_test_set(model, max_images=20, misclassified_only=True):
    """
    Génère Grad-CAM sur les images du test set.
    Par défaut, ne montre que les images mal classées.
    """
    print("\n📊 Chargement du test set...")
    DATA_PATH = "C:\Flowers Classification\code\data"
    images, labels, class_names = load_flower_data(DATA_PATH, image_size=(224, 224))

    # Recréer le même split que l'entraînement (seed 42)
    total = len(images)
    test_size = int(total * 0.1)
    val_size = int(total * 0.2)
    train_size = total - val_size - test_size

    train_indices, val_indices, test_indices = random_split(
        range(total),
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    test_images = images[test_indices.indices]
    test_labels = labels[test_indices.indices]

    # Inférence sur le test set
    test_dataset = FlowerDataset(test_images, test_labels, transform=get_transforms(train=False))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for imgs_batch, lbls_batch in test_loader:
            imgs_batch = imgs_batch.to(DEVICE)
            outputs = model(imgs_batch)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls_batch.numpy())
            all_probs.extend(probs.cpu().numpy())

    # Filtrer les indices
    indices = list(range(len(test_labels)))
    if misclassified_only:
        indices = [i for i in indices if all_preds[i] != all_labels[i]]
        print(f"❌ Images mal classées: {len(indices)} / {len(test_labels)}")
    else:
        print(f"🖼️  Images dans le test set: {len(indices)}")

    if not indices:
        print("Rien à afficher.")
        return

    indices = indices[:max_images]
    n = len(indices)

    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1:
        axes = np.array(axes).reshape(1, -1).flatten()
    else:
        axes = axes.flatten()

    gradcam = GradCAM(model, model.conv4)

    for plot_idx, data_idx in enumerate(indices):
        ax = axes[plot_idx]

        # Image originale (non transformée)
        org_img = Image.fromarray(test_images[data_idx])
        input_tensor = transform(org_img).unsqueeze(0).to(DEVICE)

        pred_class = int(all_preds[data_idx])
        true_class = int(all_labels[data_idx])
        conf = all_probs[data_idx][pred_class] * 100

        # Heatmap pour la classe PRÉDITE (pour comprendre l'erreur)
        cam = gradcam.generate(input_tensor, pred_class)
        superimposed = overlay_heatmap(org_img, cam, alpha=0.5)

        pred_name = CLASS_NAMES[pred_class]
        true_name = CLASS_NAMES[true_class]

        title = f"Prédit: {pred_name} ({conf:.0f}%)\nRéel: {true_name}"
        color = 'darkred' if pred_class != true_class else 'darkgreen'

        ax.imshow(superimposed)
        ax.set_title(title, fontsize=9, color=color)
        ax.axis('off')

    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    suffix = "misclassified" if misclassified_only else "testset"
    save_name = f"gradcam_{suffix}.png"
    save_path = BASE_DIR / save_name
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Grille sauvegardée: {save_path}")
    plt.close()

    gradcam.remove_hook()

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python gradcam.py <chemin_image>")
        print("  python gradcam.py <chemin_image> --target_class NomClasse")
        print("  python gradcam.py <chemin_dossier>")
        print("  python gradcam.py <chemin_dossier> --max 12")
        print("  python gradcam.py --test_set")
        print("  python gradcam.py --test_set --misclassified_only --max 20")
        sys.exit(1)

    use_test_set = '--test_set' in sys.argv
    misclassified_only = '--misclassified_only' in sys.argv

    max_images = 20
    if '--max' in sys.argv:
        try:
            idx = sys.argv.index('--max')
            max_images = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            pass

    target_class_name = None
    if '--target_class' in sys.argv:
        idx = sys.argv.index('--target_class')
        target_class_name = sys.argv[idx + 1]

    # Chargement modèle
    if not CHECKPOINT_PATH.exists():
        print(f"❌ Checkpoint introuvable: {CHECKPOINT_PATH}")
        sys.exit(1)

    model = create_model(num_classes=5, device=DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Modèle chargé: {CHECKPOINT_PATH}")

    if use_test_set:
        run_test_set(model, max_images=max_images, misclassified_only=misclassified_only)
    else:
        image_path = f"C:/Flowers Classification/code/{sys.argv[1]}"
        if not os.path.exists(image_path):
            print(f"❌ Chemin introuvable: {image_path}")
            sys.exit(1)
        if os.path.isdir(image_path):
            run_folder(image_path, model, max_images=max_images)
        else:
            run_single(image_path, model, target_class_name=target_class_name)

if __name__ == "__main__":
    main()