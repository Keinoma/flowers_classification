import torch
import torch.nn as nn
from typing import List


class FlowerCNN(nn.Module):
    """
    CNN from scratch pour la classification de fleurs.

    Architecture explicite avec couches convolutives, pooling et fully connected.
    """

    def __init__(self, num_classes: int = 5):
        """
        Initialise le modèle.

        Args:
            num_classes: Nombre de classes de fleurs
        """
        super(FlowerCNN, self).__init__()

        # === BLOC CONVOLUTIF ===
        # Bloc 1: 3 -> 16 canaux, 224x224 -> 112x112
        # Bloc 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),        # ← AJOUTER
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Bloc 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),        # ← AJOUTER
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Bloc 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),        # ← AJOUTER
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Bloc 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),       # ← AJOUTER
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # === CLASSIFIER ===
        # Après 4 MaxPool(2): 224 / 2^4 = 14
        # Taille du flatten: 128 * 14 * 14 = 25088
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du modèle.

        Args:
            x: Tensor de forme (B, 3, H, W)

        Returns:
            Logits de forme (B, num_classes)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.classifier(x)
        return x

    def get_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retourne les prédictions (classes) pour un batch d'images.
        """
        with torch.no_grad():
            logits = self.forward(x)
            _, predicted = torch.max(logits, 1)
        return predicted

    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retourne les probabilités softmax pour un batch d'images.
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs

    def count_parameters(self) -> dict:
        """
        Compte les paramètres entraînables et totaux.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable,
            'trainable_percent': 100 * trainable / total
        }

    def __repr__(self) -> str:
        stats = self.count_parameters()
        return (f"FlowerCNN(num_classes={self.num_classes}, "
                f"backbone=from_scratch, "
                f"trainable_params={stats['trainable']:,}/{stats['total']:,})")


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def create_model(num_classes: int = 5, device: str = 'cuda') -> FlowerCNN:
    """
    Crée et initialise le modèle sur le device spécifié.

    Args:
        num_classes: Nombre de classes
        device: 'cuda' ou 'cpu'

    Returns:
        Modèle initialisé
    """
    model = FlowerCNN(num_classes=num_classes)
    model = model.to(device)

    # Afficher les statistiques
    stats = model.count_parameters()
    print(f"✅ Modèle créé sur {device}")
    print(f"   Paramètres totaux: {stats['total']:,}")
    print(f"   Paramètres entraînables: {stats['trainable']:,} ({stats['trainable_percent']:.1f}%)")

    return model


def get_model_summary(model: nn.Module, input_size: tuple = (1, 3, 224, 224)):
    """
    Affiche un résumé du modèle.
    """
    print("=" * 60)
    print("RÉSUMÉ DU MODÈLE")
    print("=" * 60)

    # Test forward pass
    x = torch.randn(input_size).to(next(model.parameters()).device)

    print(f"\nInput shape: {x.shape}")

    # Compter les couches
    total_layers = sum(1 for _ in model.modules())
    print(f"Nombre total de couches: {total_layers}")

    # Paramètres
    stats = model.count_parameters()
    print(f"Paramètres totaux: {stats['total']:,}")
    print(f"Paramètres entraînables: {stats['trainable']:,}")

    # Test output
    with torch.no_grad():
        output = model(x)
    print(f"\nOutput shape: {output.shape}")
    print(f"Classes: {output.shape[1]}")

    print("=" * 60)