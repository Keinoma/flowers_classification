"""
============================================================
EXERCICE ULTRA-BASIQUE : MINI CNN SUR MNIST (PyTorch)
============================================================
Objectif : Construire un CNN simple avec PyTorch pour classer
les chiffres manuscrits MNIST (0 à 9).

Tu vas utiliser EXACTEMENT les mêmes couches que dans ton
FlowerCNN : Conv2d, ReLU, MaxPool2d, Sequential, Flatten,
Linear, Dropout.

Architecture cible :
  Entrée  : (B, 1, 28, 28)   -> image grayscale 28x28
  Bloc 1  : Conv2d(1, 8) + ReLU + MaxPool2d(2)  -> (B, 8, 14, 14)
  Bloc 2  : Conv2d(8, 16) + ReLU + MaxPool2d(2) -> (B, 16, 7, 7)
  Flatten : -> (B, 16*7*7) = (B, 784)
  Dense   : Linear(784, 32) + ReLU + Dropout(0.3)
  Sortie  : Linear(32, 10)   -> 10 classes (chiffres 0-9)

Complète les sections marquées [À COMPLÉTER].
============================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ------------------------------------------------------------
# 1. HYPERPARAMÈTRES
# ------------------------------------------------------------
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 3          # On garde peu d'epochs, c'est un exercice
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device utilisé : {DEVICE}")

# ------------------------------------------------------------
# 2. CHARGEMENT DES DONNÉES MNIST
# ------------------------------------------------------------
# MNIST est un dataset classique d'images 28x28 en niveaux de gris.
# torchvision le télécharge automatiquement.

transform = transforms.Compose([
    transforms.ToTensor(),                      # Convertit PIL -> Tensor [0,1]
    transforms.Normalize((0.1307,), (0.3081,))   # Normalisation standard MNIST
])

train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Images d'entraînement : {len(train_dataset)}")
print(f"Images de test      : {len(test_dataset)}")

# ------------------------------------------------------------
# 3. DÉFINITION DU MODÈLE CNN (À COMPLÉTER)
# ------------------------------------------------------------
class MiniMNISTCNN(nn.Module):
    """
    Mini CNN pour MNIST.
    Architecture à compléter ci-dessous.
    """
    def __init__(self, num_classes: int = 10):
        super(MiniMNISTCNN, self).__init__()

        # === BLOC CONVOLUTIF 1 ===
        # Entrée : 1 canal (grayscale), sortie : 8 canaux
        # kernel=3, padding=1 pour garder la taille 28x28
        # MaxPool2d(2) divise par 2 -> 14x14
        self.conv1 = nn.Sequential(
            # [À COMPLÉTER] : nn.Conv2d(in_channels=..., out_channels=..., kernel_size=..., padding=...)
            # [À COMPLÉTER] : activation ReLU
            # [À COMPLÉTER] : pooling MaxPool2d(kernel_size=..., stride=...)
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # === BLOC CONVOLUTIF 2 ===
        # Entrée : 8 canaux, sortie : 16 canaux
        # 14x14 -> MaxPool2d(2) -> 7x7
        self.conv2 = nn.Sequential(
            # [À COMPLÉTER] : Conv2d(8 -> 16, kernel=3, padding=1)
            # [À COMPLÉTER] : ReLU
            # [À COMPLÉTER] : MaxPool2d(2, stride=2)
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # === CLASSIFIER (FULLY CONNECTED) ===
        # Après 2 MaxPool(2) : 28 / 2 / 2 = 7
        # Taille du flatten : 16 canaux * 7 * 7 = 784
        self.classifier = nn.Sequential(
            # [À COMPLÉTER] : aplatir le tenseur
            # [À COMPLÉTER] : Linear(16*7*7, 32)
            # [À COMPLÉTER] : ReLU
            # [À COMPLÉTER] : Dropout(0.3)
            # [À COMPLÉTER] : Linear(32, num_classes)
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant.
        Args:
            x: Tensor de forme (B, 1, 28, 28)
        Returns:
            Logits de forme (B, 10)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.classifier(x)
        return x


# ------------------------------------------------------------
# 4. INSTANCIATION DU MODÈLE, PERTE ET OPTIMISEUR
# ------------------------------------------------------------
model = MiniMNISTCNN(num_classes=10).to(DEVICE)

# [À COMPLÉTER] : quelle fonction de perte pour une classification multi-classes ?
criterion = nn.CrossEntropyLoss()

# [À COMPLÉTER] : quel optimiseur ? (Adam est un bon choix par défaut)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\nArchitecture du modèle :")
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nParamètres totaux : {total_params:,}")

# ------------------------------------------------------------
# 5. BOUCLE D'ENTRAÎNEMENT (À COMPLÉTER)
# ------------------------------------------------------------
print("\n" + "="*50)
print("DÉBUT DE L'ENTRAÎNEMENT")
print("="*50)

for epoch in range(EPOCHS):
    model.train()           # Mode entraînement (active Dropout, BatchNorm...)
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        # Déplacer les données sur le device (GPU/CPU)
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # [À COMPLÉTER] : remettre les gradients à zéro
        optimizer.zero_grad()

        # [À COMPLÉTER] : forward pass -> obtenir les logits
        outputs = model(images)

        # [À COMPLÉTER] : calculer la perte
        loss = criterion(outputs, labels)

        # [À COMPLÉTER] : backpropagation (calcul des gradients)
        loss.backward()

        # [À COMPLÉTER] : mise à jour des poids
        optimizer.step()

        # Statistiques
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if batch_idx % 300 == 0:
            print(f"  Batch {batch_idx:4d} | Loss : {loss.item():.4f}")

    accuracy = 100 * correct / total
    print(f"\n>>> Epoch {epoch+1}/{EPOCHS} | Loss moyenne : {running_loss/len(train_loader):.4f} | Accuracy : {accuracy:.2f}%")

# ------------------------------------------------------------
# 6. ÉVALUATION SUR LE JEU DE TEST
# ------------------------------------------------------------
print("\n" + "="*50)
print("ÉVALUATION SUR LE JEU DE TEST")
print("="*50)

model.eval()            # Mode évaluation (désactive Dropout)

correct = 0
total = 0

with torch.no_grad():   # Pas besoin de calculer les gradients en test
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Accuracy sur le test : {test_accuracy:.2f}%")

# ------------------------------------------------------------
# 7. PRÉDICTION SUR QUELQUES IMAGES (BONUS)
# ------------------------------------------------------------
print("\n" + "="*50)
print("PRÉDICTIONS SUR 5 IMAGES DU TEST")
print("="*50)

model.eval()
with torch.no_grad():
    images, labels = next(iter(test_loader))
    images = images[:5].to(DEVICE)
    labels = labels[:5]

    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    for i in range(5):
        print(f"  Image {i+1} : prédit = {predicted[i].item()} | réel = {labels[i].item()}")

print("\n✅ Exercice terminé !")