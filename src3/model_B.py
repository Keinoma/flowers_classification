import torch.nn as nn
from torchvision import models


class FlowerResNet(nn.Module):
    """
    ResNet50 pré-entraîné sur ImageNet avec backbone gelé.
    Seule la tête de classification est entraînée (feature extraction).
    """

    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        # Gel total du backbone — no gradient through ResNet50
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Classifier 
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable,
            'trainable_percent': 100 * trainable / total
        }


def create_model(num_classes: int = 5, device: str = 'cuda'):
    """Crée et initialise le modèle ResNet50 gelé."""
    model = FlowerResNet(num_classes=num_classes)
    model = model.to(device)
    stats = model.count_parameters()
    print(f"✅ Modèle créé sur {device}")
    print(f"   Paramètres totaux: {stats['total']:,}")
    print(f"   Paramètres entraînables: {stats['trainable']:,} ({stats['trainable_percent']:.1f}%)")
    print(f"   Paramètres gelés: {stats['frozen']:,}")
    return model
