# %% [markdown]
# # Comparaison Original vs Augmenté
#
# Comment l'utiliser dans VSCode :
# 1. Installe l'extension "Python" de Microsoft si ce n'est pas déjà fait
#    (elle inclut le support Jupyter / Interactive Window — pas besoin
#    d'une extension spéciale pour les images).
# 2. Ouvre ce fichier .py normalement. VSCode reconnaît les séparateurs
#    "# %%" comme des cellules Jupyter et affiche "Run Cell" au-dessus
#    de chacune.
# 3. Modifie IMAGE_DIR ci-dessous pour pointer vers un dossier contenant
#    quelques-unes de tes images de fleurs, puis exécute les cellules
#    une par une (ou "Run All").
# 4. La grille d'images s'affiche directement dans le panneau Interactive
#    de VSCode (comme un notebook), et est aussi sauvegardée en PNG.
 
# %%
import random
from pathlib import Path
 
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
 
# %%
# --- 1. Configure ici ---
print("cul")
IMAGE_DIR = Path("C:/Flowers Classification/code/data/5_classes/Orchid/")  # dossier contenant quelques images
N_IMAGES = 4          # nombre d'images originales à afficher (une par ligne)
N_AUGMENTATIONS = 5   # nombre de versions augmentées par image (colonnes)
IMAGE_SIZE = 224
SEED = 0               # change la seed pour voir d'autres images/augmentations
 
random.seed(SEED)
 
# %%
# --- 2. Les deux pipelines à comparer (repris de ton code) ---
# IMPORTANT : ToTensor() + Normalize() sont retirés ici. Ce sont des étapes
# pour l'entraînement, pas pour l'affichage — Normalize() décale les valeurs
# de pixels avec les stats ImageNet, ce qui donnerait des couleurs fausses
# (et des warnings de clipping) si on essayait de les afficher telles quelles.
 
light_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
])  # Style Groupe B' : pas d'augmentation, juste le resize pour comparer à taille égale
 
heavy_transform = transforms.Compose([
    # 1. CROPPING
    transforms.RandomResizedCrop(
        size=(IMAGE_SIZE, IMAGE_SIZE),
        scale=(0.7, 1.0),
        ratio=(0.75, 1.33),
    ),
    # 2. FLIP
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    # 3. ROTATION
    transforms.RandomRotation(degrees=30),
    # 4. AFFINE
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=(-10, 10)),
    # 5. PERSPECTIVE
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    # 6. COLOR JITTER
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    # 7. GAUSSIAN BLUR
    transforms.RandomApply(
        [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2
    ),
])  # Style Groupe B : pipeline "lourd"
 
# %%
# --- 3. Sélection d'images au hasard dans le dossier ---
image_paths = (
    list(IMAGE_DIR.glob("**/*.jpg"))
    + list(IMAGE_DIR.glob("**/*.jpeg"))
    + list(IMAGE_DIR.glob("**/*.png"))
)
assert image_paths, f"Aucune image trouvée dans {IMAGE_DIR}"
sample_paths = random.sample(image_paths, min(N_IMAGES, len(image_paths)))
 
# %%
# --- 4. Grille : chaque ligne = 1 image, colonnes = original + N versions augmentées ---
fig, axes = plt.subplots(
    len(sample_paths),
    N_AUGMENTATIONS + 1,
    figsize=(3 * (N_AUGMENTATIONS + 1), 3 * len(sample_paths)),
)
if len(sample_paths) == 1:
    axes = axes[None, :]
 
for row, path in enumerate(sample_paths):
    img = Image.open(path).convert("RGB")
 
    axes[row, 0].imshow(light_transform(img))
    axes[row, 0].set_title("Original" if row == 0 else "")
    axes[row, 0].set_ylabel(path.name, fontsize=8)
    axes[row, 0].set_xticks([])
    axes[row, 0].set_yticks([])
 
    for col in range(1, N_AUGMENTATIONS + 1):
        axes[row, col].imshow(heavy_transform(img))
        axes[row, col].set_title(f"Aug {col}" if row == 0 else "")
        axes[row, col].axis("off")
 
plt.tight_layout()
plt.savefig("comparaison_augmentation.png", dpi=150)
plt.show()
 
# %% [markdown]
# ## Variante rapide : une seule image, plein d'augmentations
# Utile pour juger d'un coup d'œil si le pipeline est trop agressif
# (ex: pétales qui sortent du cadre, flou trop fort, couleurs qui dérivent).
 
# %%
img = Image.open(sample_paths[0]).convert("RGB")
n = 12
fig, axes = plt.subplots(3, 4, figsize=(12, 9))
for ax in axes.flat:
    ax.imshow(heavy_transform(img))
    ax.axis("off")
plt.suptitle(f"12 augmentations de : {sample_paths[0].name}")
plt.tight_layout()
plt.savefig("augmentations_une_image.png", dpi=150)
plt.show()