""" architecture 

C:\Flowers Classification\code\
├── data\
│    ├── 5_classes\
│    │    ├── Lilly/        # Images renammed : Li1.jpg, Li2.jpg...
│    │    ├── Lotus/        # Lo1.jpg, Lo2.jpg...
│    │    ├── Orchid/       # Or1.jpg, Or2.jpg...
│    │    ├── Sunflower/    # Su1.jpg, Su2.jpg...
│    │    └── Tulip/        # Tu1.jpg, Tu2.jpg...
│    │
│    └── 17_classes\
│         ├── files.txt
│         ├── image_001.jpg
│         ├── image_002.jpg
│         └── image_003.jpg etc...
├── src0\
│   ├── __init__.py
│   ├── load_data.py      # Classe LoadData : chargement images → NumPy arrays avec glob
│   ├── main.py           # Orchestration complète (entraînement complet)
│   ├── model.py          # Classe FlowerCNN : ResNet50 + classifier personnalisé
│   ├── predict.py        # Prédiction sur image unique via argument terminal
│   └── train.py          # Classe Trainer : boucle entraînement/validation complète
│── src0\
│   │── __init__.py
│   ├── load_data.py      # Classe LoadData : chargement images → NumPy arrays avec glob
│   ├── main.py           # Orchestration complète (entraînement complet)
│   ├── model.py          # Classe FlowerCNN : ResNet50 + classifier personnalisé
│   ├── predict.py        # Prédiction sur image unique via argument terminal
│   └── train.py          # Classe Trainer : boucle entraînement/validation complète
│
│── exerciceCNN.py\




"""