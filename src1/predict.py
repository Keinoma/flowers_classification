import sys
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import create_model

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASS_NAMES = ['Lilly', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']
IMAGE_SIZE = 224

# Transformations
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

from pathlib import Path
SRC_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = SRC_DIR / 'checkpoints' / 'run_seed42' / 'best_model.pth'

# Charger le modèle
model = create_model(num_classes=5, device=DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


def predict_image(image_path):
    """Prédit la classe d'une image."""
    if not os.path.exists(image_path):
        print(f"❌ Erreur: Le fichier '{image_path}' n'existe pas.")
        return

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()

    print(f"Image: {image_path}")
    print(f"Prédiction: {CLASS_NAMES[predicted_class]}")
    print("Probabilités:")
    for i, (cls, prob) in enumerate(zip(CLASS_NAMES, probabilities.cpu().numpy())):
        marker = " ←" if i == predicted_class else ""
        print(f"  {cls:12s}: {prob*100:.1f}%{marker}")
    print()


def main():
    """Point d'entrée principal avec argument en ligne de commande."""
    if len(sys.argv) < 2:
        print("Usage: python predict.py <chemin_vers_image>")
        print("Exemple: python predict.py data/5_classes/Lilly/00a7.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    predict_image(image_path)


if __name__ == "__main__":
    main()