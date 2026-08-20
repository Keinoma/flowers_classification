import torch.nn as nn
from torchvision import models


class FlowerResNet(nn.Module):
    """
    ResNet50 pre-entraine sur ImageNet avec backbone DEGELe.
    Toutes les couches sont entrainables (fine-tuning complet).
    La tete de classification est remplacee par un classifier adapte aux fleurs.
    """

    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        ## Here, we would have frozen the backbone in the previous algorithm

        # Remplacement de la tete FC par un classifier adapte aux fleurs
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

    def unfreeze_layers(self, layer_names: list = None):
        """
        Degel progressif des couches du backbone.
        Par defaut, degle tout. Si layer_names est fourni, degle uniquement
        les couches specifiees (ex: ['layer4', 'layer3']).

        Args:
            layer_names: liste des noms de couches a degeler (None = tout)
        """
        if layer_names is None:
            # Degel complet — tous les parametres du backbone
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("   Dégel complet du backbone (toutes les couches)")
        else:
            # Degel selectif — uniquement les couches specifiees
            for name, param in self.backbone.named_parameters():
                if any(layer in name for layer in layer_names):
                    param.requires_grad = True
            print(f"   Dégel sélectif : {layer_names}")


def create_model(num_classes: int = 5, device: str = 'cuda', unfreeze: bool = True):
    """Cree et initialise le modele ResNet50 pour fine-tuning.

    Args:
        num_classes: nombre de classes
        device: 'cuda' ou 'cpu'
        unfreeze: si True, degle tout le backbone (fine-tuning complet)
    """
    model = FlowerResNet(num_classes=num_classes)
    if unfreeze:
        model.unfreeze_layers()  # degel complet par defaut
    model = model.to(device)
    stats = model.count_parameters()
    print(f"✅ Modèle créé sur {device}")
    print(f"   Paramètres totaux: {stats['total']:,}")
    print(f"   Paramètres entraînables: {stats['trainable']:,} ({stats['trainable_percent']:.1f}%)")
    print(f"   Paramètres gelés: {stats['frozen']:,}")
    return model
