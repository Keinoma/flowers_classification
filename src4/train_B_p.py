"""
Module d'entraînement pour la classification de fleurs.
Gère la boucle d'entraînement, la validation, la sauvegarde et les métriques.
GROUPE B' — Transfer Learning SANS data augmentation.
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

from model_B_p import FlowerResNet


class FlowerDataset(Dataset):
    """
    Dataset PyTorch adapté pour les NumPy arrays chargés par LoadData.
    """
    
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
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
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5, verbose=True
        )
        
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
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc="Entraînement")
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            acc = accuracy_score(all_labels, all_preds) * 100
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.2f}%'})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds) * 100
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> Tuple[float, float]:
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
        total_start_time = time.time() 
        
        os.makedirs(save_dir, exist_ok=True)
        self.best_model_path = os.path.join(save_dir, "best_model_B_p.pth")
                
        print(f"\n{'='*60}")
        print(f"DÉBUT DE L'ENTRAÎNEMENT")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.epochs}")
        print(f"LR initial: {self.optimizer.param_groups[0]['lr']}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.epochs):
            start_time = time.time()
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            epoch_time = time.time() - start_time
            
            print(f"\nEpoch [{epoch+1}/{self.epochs}] - {epoch_time:.1f}s")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
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

        total_time = time.time() - total_start_time
        print(f"\n⏱️  Temps d'entraînement total: {total_time/60:.1f} min")
        
        return self.history
    
    def plot_history(self, save_path: Optional[str] = None):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Validation')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(self.history['train_acc'], label='Train')
        axes[1].plot(self.history['val_acc'], label='Validation')
        axes[1].set_title('Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True)
        
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
        
        acc = accuracy_score(all_labels, all_preds) * 100
        f1_macro = f1_score(all_labels, all_preds, average='macro') * 100
        print(f"\n{'='*60}")
        print(f"ÉVALUATION FINALE")
        print(f"{'='*60}")
        print(f"Accuracy: {acc:.2f}%")
        print(f"  F1-score (macro): {f1_macro:.2f}%")   
        print(f"\nRapport de classification:")
        print(classification_report(all_labels, all_preds, target_names=class_names))
        
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
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha='center', va='center')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Matrice de confusion sauvegardée: {save_path}")
        plt.close()
        
        return acc


def get_transforms(train: bool = True, image_size: int = 224):
    """
    Transformations pour le Groupe B' (train=True) ou val/test (train=False).
    Groupe B' = Transfer Learning SANS data augmentation.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    else:
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
    
    total = len(images)
    test_size = int(total * test_split)
    val_size = int(total * val_split)
    train_size = total - val_size - test_size
    
    train_indices, val_indices, test_indices = random_split(
        range(total),
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_images = images[train_indices.indices]
    train_labels = labels[train_indices.indices]
    
    val_images = images[val_indices.indices]
    val_labels = labels[val_indices.indices]
    
    test_images = images[test_indices.indices]
    test_labels = labels[test_indices.indices]
    
    train_dataset = FlowerDataset(
        train_images, 
        train_labels, 
        transform=get_transforms(train=True)
    )
    
    val_dataset = FlowerDataset(
        val_images, 
        val_labels, 
        transform=get_transforms(train=False)
    )
    
    test_dataset = FlowerDataset(
        test_images, 
        test_labels, 
        transform=get_transforms(train=False)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
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
    
    print(f"📊 Dataset split:")
    print(f"   Train: {len(train_dataset)} ({len(train_dataset)/total*100:.1f}%)")
    print(f"   Val:   {len(val_dataset)} ({len(val_dataset)/total*100:.1f}%)")
    print(f"   Test:  {len(test_dataset)} ({len(test_dataset)/total*100:.1f}%)")
    
    return train_loader, val_loader, test_loader