import torch
import torch.nn as nn
from torchvision import models
from typing import List


class FlowerCNN(nn.Module):
    """
    CNN pour la classification de fleurs basé sur ResNet50 avec transfer learning.
    
    Architecture :
    - Backbone : ResNet50 pré-entraîné (couches gelées par défaut)
    - Classifier : Couche fully-connected adaptée au nombre de classes
    """
    
    def __init__(self, num_classes: int = 5, freeze_backbone: bool = True):
        """
        Initialise le modèle.
        
        Args:
            num_classes: Nombre de classes de fleurs (5 par défaut)
            freeze_backbone: Si True, gèle les poids du backbone ResNet50
        """
        super(FlowerCNN, self).__init__()
        
        # Charger ResNet50 pré-entraîné sur ImageNet
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Geler le backbone si demandé
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Remplacer le classifier final
        # ResNet50 a 2048 features en sortie du backbone
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        self.num_classes = num_classes
    
    def unfreeze_backbone(self, layers: int = 0):
        """
        Dégele les dernières couches du backbone pour le fine-tuning.
        
        Args:
            layers: Nombre de couches à dégeler (0 = tout dégeler, -1 = tout geler)
        """
        if layers == -1:
            for param in self.backbone.parameters():
                param.requires_grad = False
            return
        
        if layers == 0:
            # Tout dégeler
            for param in self.backbone.parameters():
                param.requires_grad = True
            return
        
        # Dégele les N dernières couches (approche simplifiée)
        # Pour ResNet50, on dégele les couches du layer4 et fc
        for name, param in self.backbone.named_parameters():
            if 'layer4' in name or 'fc' in name:
                param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du modèle.
        
        Args:
            x: Tensor de forme (B, 3, H, W)
            
        Returns:
            Logits de forme (B, num_classes)
        """
        return self.backbone(x)
    
    def get_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retourne les prédictions (classes) pour un batch d'images.
        
        Args:
            x: Tensor de forme (B, 3, H, W)
            
        Returns:
            Tensor de classes prédites (B,)
        """
        with torch.no_grad():
            logits = self.forward(x)
            _, predicted = torch.max(logits, 1)
        return predicted
    
    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retourne les probabilités softmax pour un batch d'images.
        
        Args:
            x: Tensor de forme (B, 3, H, W)
            
        Returns:
            Tensor de probabilités (B, num_classes)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs
    
    def count_parameters(self) -> dict:
        """
        Compte les paramètres entraînables et totaux.
        
        Returns:
            Dictionnaire avec les statistiques
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
                f"backbone=ResNet50, "
                f"trainable_params={stats['trainable']:,}/{stats['total']:,})")


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def create_model(num_classes: int = 5, 
                 freeze_backbone: bool = True,
                 device: str = 'cuda') -> FlowerCNN:
    """
    Crée et initialise le modèle sur le device spécifié.
    
    Args:
        num_classes: Nombre de classes
        freeze_backbone: Geler le backbone
        device: 'cuda' ou 'cpu'
        
    Returns:
        Modèle initialisé
    """
    model = FlowerCNN(num_classes=num_classes, freeze_backbone=freeze_backbone)
    model = model.to(device)
    
    # Afficher les statistiques
    stats = model.count_parameters()
    print(f"✅ Modèle créé sur {device}")
    print(f"   Paramètres totaux: {stats['total']:,}")
    print(f"   Paramètres entraînables: {stats['trainable']:,} ({stats['trainable_percent']:.1f}%)")
    
    return model


def get_model_summary(model: nn.Module, input_size: tuple = (1, 3, 224, 224)):
    """
    Affiche un résumé du modèle (similaire à model.summary() de Keras).
    
    Args:
        model: Le modèle PyTorch
        input_size: Taille d'entrée pour le test
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
    stats = model.count_parameters() if hasattr(model, 'count_parameters') else {
        'total': sum(p.numel() for p in model.parameters()),
        'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    print(f"Paramètres totaux: {stats['total']:,}")
    print(f"Paramètres entraînables: {stats['trainable']:,}")
    
    # Test output
    with torch.no_grad():
        output = model(x)
    print(f"\nOutput shape: {output.shape}")
    print(f"Classes: {output.shape[1]}")
    
    print("=" * 60)