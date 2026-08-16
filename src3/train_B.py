"""
Module d'entraînement pour la classification de fleurs.
Gère la boucle d'entraînement, la validation, la sauvegarde et les métriques.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm


class FlowerDataset(Dataset):
    """
    Dataset PyTorch adapté pour les NumPy arrays chargés par LoadData.
    """
    
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        """
        Args:
            images: NumPy array de forme (N, H, W, 3)
            labels: NumPy array de forme (N,)
            transform: Transformations torchvision à appliquer
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Convertir en PIL Image pour les transformations torchvision
        image = self.images[idx]  # (H, W, 3), uint8
        label = self.labels[idx]
        
        # Convertir en PIL Image
        from PIL import Image
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class Trainer:
    """
    Classe gérant l'entraînement complet du modèle.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = 'cuda',
                 learning_rate: float = 0.001,
                 epochs: int = 20):
        """
        Initialise le trainer.
        
        Args:
            model: Instance de resNet
            train_loader: DataLoader d'entraînement
            val_loader: DataLoader de validation
            device: 'cuda' ou 'cpu'
            learning_rate: Taux d'apprentissage
            epochs: Nombre d'epochs
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        
        # Loss et optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        # Scheduler : réduit le LR si la validation stagne
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5, verbose=True
        )
        
        # Historique
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        self.best_val_acc = 0.0
        self.best_model_path = None
        
    def train_epoch(self) -> Tuple[float, float]:
        """
        Entraîne le modèle pour une epoch.
        
        Returns:
            (train_loss, train_accuracy)
        """
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc="Entraînement")
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Stats
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Mise à jour de la barre de progression
            acc = accuracy_score(all_labels, all_preds) * 100
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.2f}%'})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds) * 100
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> Tuple[float, float]:
        """
        Évalue le modèle sur le set de validation.
        
        Returns:
            (val_loss, val_accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = accuracy_score(all_labels, all_preds) * 100
        
        return epoch_loss, epoch_acc
    
    def fit(self, save_dir: str = "checkpoints") -> Dict:
        """
        Lance l'entraînement complet.
        
        Args:
            save_dir: Dossier pour sauvegarder les modèles
            
        Returns:
            Historique d'entraînement
        """
        total_start_time = time.time() 
        
        os.makedirs(save_dir, exist_ok=True)
        self.best_model_path = os.path.join(save_dir, "best_model_B.pth")
                
        print(f"\n{'='*60}")
        print(f"DÉBUT DE L'ENTRAÎNEMENT")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.epochs}")
        print(f"LR initial: {self.optimizer.param_groups[0]['lr']}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.epochs):
            start_time = time.time()
            
            # Entraînement
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc = self.validate()
            
            # Scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Historique
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Temps
            epoch_time = time.time() - start_time
            
            # Affichage
            print(f"\nEpoch [{epoch+1}/{self.epochs}] - {epoch_time:.1f}s")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
            # Sauvegarde du meilleur modèle
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, self.best_model_path)
                print(f"  💾 Meilleur modèle sauvegardé (val_acc: {val_acc:.2f}%)")
            
            print("-" * 60)
        
        print(f"\n{'='*60}")
        print(f"ENTRAÎNEMENT TERMINÉ")
        print(f"Meilleure validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Modèle sauvegardé: {self.best_model_path}")
        print(f"{'='*60}")

        total_time = time.time() - total_start_time   # ← AJOUTER à la fin
        print(f"\n⏱️  Temps d'entraînement total: {total_time/60:.1f} min")
        
        return self.history
    
    def plot_history(self, save_path: Optional[str] = None):
        """
        Affiche les courbes d'entraînement.
        
        Args:
            save_path: Chemin pour sauvegarder la figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Loss
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Validation')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy
        axes[1].plot(self.history['train_acc'], label='Train')
        axes[1].plot(self.history['val_acc'], label='Validation')
        axes[1].set_title('Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True)
        
        # Learning Rate
        axes[2].plot(self.history['lr'])
        axes[2].set_title('Learning Rate')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('LR')
        axes[2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"📊 Graphique sauvegardé: {save_path}")
        
        plt.close()
    
    def evaluate(self, test_loader: DataLoader, class_names: List[str], save_path: Optional[str] = None):
        """
        Évaluation finale avec matrice de confusion et rapport de classification.
        
        Args:
            test_loader: DataLoader de test
            class_names: Noms des classes
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Évaluation"):
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Métriques
        acc = accuracy_score(all_labels, all_preds) * 100
        f1_macro = f1_score(all_labels, all_preds, average='macro') * 100
        print(f"\n{'='*60}")
        print(f"ÉVALUATION FINALE")
        print(f"{'='*60}")
        print(f"Accuracy: {acc:.2f}%")
        print(f"  F1-score (macro): {f1_macro:.2f}%")   
        print(f"\nRapport de classification:")
        print(classification_report(all_labels, all_preds, target_names=class_names))
        
        # Matrice de confusion
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, cmap='Blues')
        plt.title('Matrice de Confusion')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        plt.xlabel('Prédit')
        plt.ylabel('Réel')
        
        # Annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha='center', va='center')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Matrice de confusion sauvegardée: {save_path}")
        plt.close()
        
        return acc


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def get_transforms(train: bool = True, image_size: int = 224):
    """
    Transformations pour la baseline (train=False) ou le Groupe A (train=True).
    """
    if train:
        # ═══════════════════════════════════════════════════════════════
        # GROUPE A — Data Augmentation complète
        # ═══════════════════════════════════════════════════════════════
        return transforms.Compose([
            # 1. CROPPING — Extrait une zone aléatoire puis resize
            transforms.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.7, 1.0),      # garde entre 70% et 100% de l'image
                ratio=(0.75, 1.33)     # ratio d'aspect toléré
            ),
            
            # 2. FLIP — Horizontal + Vertical (les fleurs peuvent être dans tous les sens)
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            
            # 3. ROTATION — Jusqu'à ±30° (les fleurs ne sont pas toujours droites)
            transforms.RandomRotation(degrees=30),
            
            # 4. AFFINE — Légère translation + shear (déformation douce)
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),   # translation jusqu'à 10%
                shear=(-10, 10)         # cisaillement ±10°
            ),
            
            # 5. PERSPECTIVE — Changement de point de vue
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            
            # 6. COLOR JITTER — Variations de couleur (essentiel pour les fleurs)
            transforms.ColorJitter(
                brightness=0.3,      # ±30% luminosité
                contrast=0.3,        # ±30% contraste
                saturation=0.3,      # ±30% saturation
                hue=0.1              # légère variation de teinte
            ),
            
            # 7. GAUSSIAN BLUR — Flou léger (simule mise au point imparfaite)
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
            ], p=0.2),
            
            # 8. NORMALISATION — Toujours en dernier
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    else:
        # ═══════════════════════════════════════════════════════════════
        # VAL/TEST — Pas d'augmentation, juste resize + normalisation
        # ═══════════════════════════════════════════════════════════════
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def prepare_dataloaders(images: np.ndarray, 
                      labels: np.ndarray,
                      batch_size: int = 32,
                      val_split: float = 0.2,
                      test_split: float = 0.1,
                      num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prépare les DataLoaders à partir des NumPy arrays.
    
    Args:
        images: NumPy array (N, H, W, 3)
        labels: NumPy array (N,)
        batch_size: Taille des batchs
        val_split: Fraction pour la validation
        test_split: Fraction pour le test
        num_workers: Nombre de workers pour le chargement
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    
    # ═══════════════════════════════════════════════════════════════
    # ÉTAPE 1 : Calculer les tailles de chaque split
    # ═══════════════════════════════════════════════════════════════
    # On part du total et on calcule combien d'images vont dans chaque
    # ensemble. L'ordre est important : test d'abord, val ensuite, 
    # le reste = train.
    
    total = len(images)
    test_size = int(total * test_split)   # ex: 4999 * 0.10 = 499 images
    val_size = int(total * val_split)     # ex: 4999 * 0.20 = 999 images
    train_size = total - val_size - test_size  # ex: 4999 - 999 - 499 = 3501 images
    
    # ═══════════════════════════════════════════════════════════════
    # ÉTAPE 2 : Tirer au sort les INDICES (pas les données !)
    # ═══════════════════════════════════════════════════════════════
    # Ici, on ne touche PAS aux images. On travaille uniquement avec
    # des NUMÉROS : [0, 1, 2, 3, ..., 4998].
    # 
    # random_split va diviser cette liste de numéros en 3 paquets :
    #   train_indices  → numéros aléatoires pour l'entraînement
    #   val_indices    → numéros aléatoires pour la validation
    #   test_indices   → numéros aléatoires pour le test
    #
    # C'est comme si on tirait au sort des numéros de casiers.
    # On ne regarde pas encore ce qu'il y a DEDANS.
    
    train_indices, val_indices, test_indices = random_split(
        range(total),                     # ← la "liste" de numéros 0 à N-1
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # seed pour reproductibilité
    )
    
    # ═══════════════════════════════════════════════════════════════
    # ÉTAPE 3 : Extraire les VRAIES DONNÉES avec les indices tirés
    # ═══════════════════════════════════════════════════════════════
    # Maintenant qu'on sait QUELS numéros on veut, on va chercher
    # les images et labels correspondants dans les arrays NumPy.
    #
    # images[train_indices.indices]  →  fancy indexing NumPy
    # Ça prend les images aux positions [4, 12, 33, 87...] et crée
    # un NOUVEL array avec SEULEMENT ces images.
    #
    # CONTRAIREMENT à l'ancien code :
    #   - Avant : on créait un Subset qui pointait vers TOUT le dataset
    #             et qui utilisait des indices pour "simuler" le split.
    #   - Maintenant : on crée 3 datasets qui possèdent CHACUN leurs
    #                  propres données. Plus de dépendance cachée.
    
    train_images = images[train_indices.indices]   # ex: shape (3501, 224, 224, 3)
    train_labels = labels[train_indices.indices]   # ex: shape (3501,)
    
    val_images = images[val_indices.indices]       # ex: shape (999, 224, 224, 3)
    val_labels = labels[val_indices.indices]       # ex: shape (999,)
    
    test_images = images[test_indices.indices]     # ex: shape (499, 224, 224, 3)
    test_labels = labels[test_indices.indices]       # ex: shape (499,)
    
    # ═══════════════════════════════════════════════════════════════
    # ÉTAPE 4 : Créer 3 datasets VRAIMENT indépendants
    # ═══════════════════════════════════════════════════════════════
    # Chaque FlowerDataset reçoit SES propres images + SES propres labels.
    # Ils ne partagent plus rien. Si tu modifies l'un, les autres ne
    # bougent pas.
    #
    # - Val/Test : pas d'augmentation, juste la normalisation ImageNet.
    #              On veut évaluer sur des images "réelles".
    
    train_dataset = FlowerDataset(
        train_images, 
        train_labels, 
        transform=get_transforms(train=True)   # ← augmentation activée
    )
    
    val_dataset = FlowerDataset(
        val_images, 
        val_labels, 
        transform=get_transforms(train=False)  # ← pas d'augmentation
    )
    
    test_dataset = FlowerDataset(
        test_images, 
        test_labels, 
        transform=get_transforms(train=False)  # ← pas d'augmentation
    )
    
    # ═══════════════════════════════════════════════════════════════
    # ÉTAPE 5 : Créer les DataLoaders
    # ═══════════════════════════════════════════════════════════════
    # Le DataLoader s'occupe de :
    #   - Découper le dataset en batchs (batch_size)
    #   - Mélanger les données à chaque epoch (shuffle=True pour train)
    #   - Charger en parallèle (num_workers)
    #   - Transférer rapidement vers GPU (pin_memory=True)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,           # ← mélange à chaque epoch (obligatoire pour train)
        num_workers=num_workers, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,          # ← pas besoin de mélanger pour val/test
        num_workers=num_workers, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True
    )
    
    # ═══════════════════════════════════════════════════════════════
    # ÉTAPE 6 : Afficher un résumé
    # ═══════════════════════════════════════════════════════════════
    
    print(f"📊 Dataset split:")
    print(f"   Train: {len(train_dataset)} ({len(train_dataset)/total*100:.1f}%)")
    print(f"   Val:   {len(val_dataset)} ({len(val_dataset)/total*100:.1f}%)")
    print(f"   Test:  {len(test_dataset)} ({len(test_dataset)/total*100:.1f}%)")
    
    return train_loader, val_loader, test_loader