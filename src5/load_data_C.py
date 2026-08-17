"""
Module de chargement des données pour la classification de fleurs.
Charge les images depuis le système de fichiers et les convertit en NumPy arrays.
"""

import os
import glob
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
from collections import Counter
import matplotlib.pyplot as plt


class LoadData:
    """
    Classe pour charger des images de fleurs depuis le système de fichiers.
    Utilise glob pour trouver les images et les convertit en NumPy arrays.
    """
    
    def __init__(self, data_path: str, image_size: Tuple[int, int] = (224, 224)):
        self.data_path = data_path
        self.image_size = image_size
        self.classes: List[str] = []
        
    def discover_classes(self) -> List[str]:
        """Découvre les classes en listant les sous-dossiers (triés alphabétiquement)."""
        self.classes = sorted([
            d for d in os.listdir(self.data_path)
            if os.path.isdir(os.path.join(self.data_path, d))
        ])
        return self.classes
    
    def find_images(self, extensions: Tuple[str, ...] = ('*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp')) -> List[str]:
        """Trouve toutes les images avec glob dans les sous-dossiers de classes."""
        image_paths = []
        for ext in extensions:
            pattern = os.path.join(self.data_path, '*', ext)
            image_paths.extend(glob.glob(pattern, recursive=False))
        return sorted(image_paths)
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Charge une image, la redimensionne et la convertit en NumPy array (H, W, 3), uint8."""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.image_size, Image.LANCZOS)
            return np.array(img, dtype=np.uint8)
        except Exception as e:
            print(f"⚠️ Erreur chargement {image_path}: {e}")
            return None
    
    def load_all(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Charge toutes les images et leurs labels.
        
        Returns:
            (images_array, labels_array, classes_names)
            - images_array: shape (N, H, W, 3), dtype uint8
            - labels_array: shape (N,), dtype int64
        """
        self.discover_classes() ##
        print(f"📁 Classes trouvées: {self.classes}")
        
        class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)} # donc Lilly->0 , Lotus->1, Orchid->2 etc...
        image_paths = self.find_images() ##
        print(f"🖼️  Images trouvées: {len(image_paths)}")
        
        images_list = []
        labels_list = []
        
        for img_path in image_paths:
            class_name = os.path.basename(os.path.dirname(img_path))
            label = class_to_idx[class_name]
            
            img_array = self.load_image(img_path) ##
            if img_array is not None:
                images_list.append(img_array)
                labels_list.append(label)
        
        images_array = np.array(images_list, dtype=np.uint8)
        labels_array = np.array(labels_list, dtype=np.int64)
        
        print(f"✅ Downloading finished: {len(images_array)} images downloaded")
        print(f"   Shape images: {images_array.shape}")
        print(f"   Shape labels: {labels_array.shape}")
        

        print(f"\n Distribution des classes:")
        class_counts = Counter(labels_list)
        for i, cls in enumerate(self.classes):
            count = class_counts[i]
            pct = 100 * count / len(labels_list)
            print(f"   {cls:12s}: {count:4d} images ({pct:5.1f}%)")

        return images_array, labels_array, self.classes

    def plot_class_distribution(self, labels: np.ndarray, save_path: Optional[str] = None):
        """
        Affiche un histogramme du nombre d'images par classe.
        """
        class_counts = [np.sum(labels == i) for i in range(len(self.classes))]
        
        plt.figure(figsize=(8, 5))
        bars = plt.bar(self.classes, class_counts, color='steelblue', edgecolor='black')
        plt.title("Classes distribution", fontsize=14, fontweight='bold')
        plt.xlabel("Flower classes", fontsize=12)
        plt.ylabel("Number of images", fontsize=12)
        plt.xticks(rotation=30, ha='right')
        
        # Valeurs au-dessus des barres
        for bar, count in zip(bars, class_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     str(count), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"📊 Distribution sauvegardée: {save_path}")
        
        plt.close()


      
        
def load_flower_data(data_path: str, image_size: Tuple[int, int] = (224, 224)) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Fonction utilitaire pour charger les données de fleurs."""
    loader = LoadData(data_path, image_size)
    return loader.load_all()