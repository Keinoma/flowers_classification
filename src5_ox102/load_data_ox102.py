"""
Module de chargement des données — Oxford 102 Flowers (src5_oxford102).
Remplace load_data_ox102.py : au lieu de lire des dossiers locaux organisés
par classe, utilise torchvision.datasets.Flowers102 (téléchargement et
cache automatiques au premier appel avec download=True).

Points clés :
- Split OFFICIEL du dataset : train (1020), val (1020), test (6149).
  On ne refait PAS de random_split 70/20/10 comme sur le dataset 5 classes.
- Les arrays retournés sont concaténés dans l'ordre [train | val | test],
  ce qui permet à prepare_dataloaders de faire de simples slices
  déterministes — le split est identique pour tous les groupes et tous
  les runs, par construction.
- Même format de sortie que l'ancien loader : images uint8 (N, 224, 224, 3),
  labels int64 (N,), noms de classes — tout l'aval est donc inchangé.
"""

import numpy as np
from PIL import Image
from typing import List, Tuple
from collections import Counter
import matplotlib.pyplot as plt
from torchvision import datasets

# ═══════════════════════════════════════════════════════════════
# Noms officiels des 102 catégories (Oxford 102 Category Flower Dataset)
# torchvision ne fournit que des indices 0-101 ; cette liste suit
# l'ordre officiel des labels (indice i ↔ OXFORD_CLASS_NAMES[i]).
# ═══════════════════════════════════════════════════════════════
OXFORD_CLASS_NAMES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
    "sweet pea", "english marigold", "tiger lily", "moon orchid",
    "bird of paradise", "monkshood", "globe thistle", "snapdragon",
    "colt's foot", "king protea", "spear thistle", "yellow iris",
    "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary",
    "red ginger", "grape hyacinth", "corn poppy",
    "prince of wales feathers", "stemless gentian", "artichoke",
    "sweet william", "carnation", "garden phlox", "love in the mist",
    "mexican aster", "alpine sea holly", "ruby-lipped cattleya",
    "cape flower", "great masterwort", "siam tulip", "lenten rose",
    "barbeton daisy", "daffodil", "sword lily", "poinsettia",
    "bolero deep blue", "wallflower", "marigold", "buttercup",
    "oxeye daisy", "common dandelion", "petunia", "wild pansy", "primula",
    "sunflower", "pelargonium", "bishop of llandaff", "gaura", "geranium",
    "orange dahlia", "pink-yellow dahlia?", "cautleya spicata",
    "japanese anemone", "black-eyed susan", "silverbush",
    "californian poppy", "osteospermum", "spring crocus", "bearded iris",
    "windflower", "tree poppy", "gazania", "azalea", "water lily", "rose",
    "thorn apple", "morning glory", "passion flower", "lotus", "toad lily",
    "anthurium", "frangipani", "clematis", "hibiscus", "columbine",
    "desert-rose", "tree mallow", "magnolia", "cyclamen", "watercress",
    "canna lily", "hippeastrum", "bee balm", "ball moss", "foxglove",
    "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia",
    "blanket flower", "trumpet creeper", "blackberry lily",
]

NUM_CLASSES = len(OXFORD_CLASS_NAMES)  # 102


class LoadDataOxford:
    """
    Charge le dataset Oxford 102 Flowers via torchvision.

    Le dataset est téléchargé automatiquement dans data_path au premier
    appel (download=True), puis réutilisé depuis le cache local.
    """

    def __init__(self, data_path, image_size: Tuple[int, int] = (224, 224)):
        self.data_path = str(data_path)
        self.image_size = image_size
        self.classes: List[str] = OXFORD_CLASS_NAMES

    def load_split(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Charge un split officiel ('train', 'val' ou 'test').

        Returns:
            (images, labels) — images uint8 (n, H, W, 3), labels int64 (n,)
        """
        dataset = datasets.Flowers102(root=self.data_path, split=split, download=True)

        images_list = []
        labels_list = []
        for img, label in dataset:
            img = img.convert('RGB').resize(self.image_size, Image.LANCZOS)
            images_list.append(np.array(img, dtype=np.uint8))
            labels_list.append(label)

        images_array = np.array(images_list, dtype=np.uint8)
        labels_array = np.array(labels_list, dtype=np.int64)
        print(f"   Split '{split}': {len(images_array)} images chargées")
        return images_array, labels_array

    def load_all(self) -> Tuple[np.ndarray, np.ndarray, List[str], Tuple[int, int, int]]:
        """
        Charge les 3 splits officiels et les concatène dans l'ordre
        [train | val | test].

        Returns:
            (images_array, labels_array, classes_names, split_sizes)
            - images_array: shape (N, H, W, 3), dtype uint8
            - labels_array: shape (N,), dtype int64
            - split_sizes: (n_train, n_val, n_test) — à passer tel quel
              à prepare_dataloaders pour retrouver le split officiel.
        """
        print(f"📁 Dataset: Oxford 102 Flowers ({NUM_CLASSES} classes)")
        print(f"   Racine du cache torchvision: {self.data_path}")

        train_images, train_labels = self.load_split('train')
        val_images, val_labels = self.load_split('val')
        test_images, test_labels = self.load_split('test')

        # Concaténation ordonnée : [train | val | test]
        images_array = np.concatenate([train_images, val_images, test_images], axis=0)
        labels_array = np.concatenate([train_labels, val_labels, test_labels], axis=0)
        split_sizes = (len(train_labels), len(val_labels), len(test_labels))

        print(f"✅ Chargement terminé: {len(images_array)} images")
        print(f"   Shape images: {images_array.shape}")
        print(f"   Shape labels: {labels_array.shape}")
        print(f"   Split officiel: train={split_sizes[0]}, val={split_sizes[1]}, test={split_sizes[2]}")

        # Distribution des classes (résumé — 102 lignes seraient illisibles)
        class_counts = np.array([np.sum(labels_array == i) for i in range(NUM_CLASSES)])
        print(f"\n Distribution des classes (102 classes):")
        print(f"   Min: {class_counts.min()} images | Max: {class_counts.max()} images")
        print(f"   Moyenne: {class_counts.mean():.1f} ± {class_counts.std():.1f} images/classe")
        print(f"   ⚠️  Dataset volontairement déséquilibré (le F1 macro est la métrique de référence)")

        return images_array, labels_array, self.classes, split_sizes

    def plot_class_distribution(self, labels: np.ndarray, save_path=None):
        """
        Histogramme du nombre d'images par classe.
        Adapté aux 102 classes : figure très large, pas d'annotations
        au-dessus des barres, noms en vertical en petite police.
        """
        class_counts = [int(np.sum(labels == i)) for i in range(NUM_CLASSES)]

        plt.figure(figsize=(24, 6))
        plt.bar(range(NUM_CLASSES), class_counts, color='steelblue', edgecolor='none')
        plt.title("Classes distribution — Oxford 102", fontsize=14, fontweight='bold')
        plt.xlabel("Flower classes", fontsize=12)
        plt.ylabel("Number of images", fontsize=12)
        plt.xticks(range(NUM_CLASSES), self.classes, rotation=90, fontsize=6)
        plt.xlim(-1, NUM_CLASSES)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"📊 Distribution sauvegardée: {save_path}")

        plt.close()


def load_flower_data(data_path, image_size: Tuple[int, int] = (224, 224)):
    """
    Fonction utilitaire — même nom que dans load_data_ox102.py pour limiter
    les changements dans les scripts appelants.

    Returns:
        (images, labels, class_names, split_sizes)
    """
    loader = LoadDataOxford(data_path, image_size)
    return loader.load_all()
