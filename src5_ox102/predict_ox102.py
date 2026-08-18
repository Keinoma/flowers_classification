"""
predict_ox102.py — Version Oxford 102 (src5_oxford102)
Inférence CLI sur une image. Affiche le Top-5 des prédictions
(avec 102 classes, afficher toutes les probabilités serait illisible).
"""

import sys
import os
import torch
from PIL import Image
from torchvision import transforms
from model_ox102 import create_model
from load_data_oxford import OXFORD_CLASS_NAMES

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASS_NAMES = OXFORD_CLASS_NAMES
NUM_CLASSES = len(CLASS_NAMES)  # 102
IMAGE_SIZE = 224
TOP_K = 5
CHECKPOINT_PATH = "checkpoints/best_model_ox102.pth"

# Transformations
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Charger le modèle
model = create_model(num_classes=NUM_CLASSES, device=DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


def predict_image(image_path):
    """Prédit la classe d'une image (affichage Top-5)."""
    if not os.path.exists(image_path):
        print(f"❌ Erreur: Le fichier '{image_path}' n'existe pas.")
        return

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    top_probs, top_indices = torch.topk(probabilities, TOP_K)

    print(f"Image: {image_path}")
    print(f"Prédiction: {CLASS_NAMES[top_indices[0].item()]}")
    print(f"Top-{TOP_K} des probabilités:")
    for rank, (idx, prob) in enumerate(zip(top_indices.cpu().numpy(), top_probs.cpu().numpy()), start=1):
        marker = " ←" if rank == 1 else ""
        print(f"  {rank}. {CLASS_NAMES[idx]:30s}: {prob*100:.1f}%{marker}")
    print()


def main():
    """Point d'entrée principal avec argument en ligne de commande."""
    if len(sys.argv) < 2:
        print("Usage: python predict_ox102.py <chemin_vers_image>")
        print("Exemple: python predict_ox102.py data/oxford102/flowers-102/jpg/image_00001.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    predict_image(image_path)


if __name__ == "__main__":
    main()
