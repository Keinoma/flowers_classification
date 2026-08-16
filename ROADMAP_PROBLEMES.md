# 🧪 Protocole d'Ablation Study — Classification de Fleurs

## 1. Vue d'ensemble des groupes

| Groupe | Nom | Objectif principal | Architecture | Data Aug. | Transfer Learning | Fine-tuning |
|--------|-----|-------------------|--------------|-----------|-------------------|-------------|
| **Baseline** | CNN From Scratch | Établir la référence minimale | CNN simple (3-4 conv blocks) | ❌ Non | ❌ Non | ❌ Non |
| **Groupe A** | + Data Augmentation | Mesurer l'impact de l'augmentation de données | Même CNN que baseline | ✅ Oui | ❌ Non | ❌ Non |
| **Groupe B** | + Transfer Learning (frozen) | Mesurer l'impact des features pré-entraînées | ResNet50 (backbone gelée) | ✅ Oui | ✅ Oui | ❌ Non |
| **Groupe B'** | Transfer Learning seul *(optionnel)* | Isoler l'effet du transfer learning sans augmentation | ResNet50 (backbone gelée) | ❌ Non | ✅ Oui | ❌ Non |
| **Groupe C** | + Fine-tuning | Mesurer le gain de l'adaptation des features | ResNet50 (backbone dégelée) | ✅ Oui | ✅ Oui | ✅ Oui |

---

## 2. Détail de chaque groupe

### 🔷 Baseline — CNN From Scratch

| | |
|---|---|
| **But** | Établir la performance minimale atteignable avec un modèle naïf et des données brutes. |
| **Architecture** | CNN simple : 3-4 blocs Conv2D → BatchNorm → ReLU → MaxPool, suivi d'un classifieur fully-connected. **Pas de residual connections.** |
| **Entraînement** | Entraînement end-to-end from scratch. |
| **Hypothèse testée** | *Quelle performance obtient-on sans aucune aide (ni augmentation, ni connaissance préalable) ?* |
| **Ce qu'on apprend** | Le plafond de performance du modèle seul. Point de comparaison obligatoire pour tous les autres groupes. |

---

### 🔷 Groupe A — Baseline + Data Augmentation

| | |
|---|---|
| **But** | Quantifier le gain apporté uniquement par l'augmentation de données. |
| **Architecture** | **Identique à la baseline** (même CNN simple). |
| **Data Augmentation** | Rotation, flip horizontal, zoom léger, éventuellement color jitter. **Vérifier le balancement des classes post-augmentation.** |
| **Hypothèse testée** | *L'augmentation de données suffit-elle à améliorer significativement la généralisation ?* |
| **Ce qu'on apprend** | Si le gain est important → le modèle souffrait d'un manque de données. Si le gain est faible → le modèle est le bottleneck (capacity issue). |

---

### 🔷 Groupe B — Transfer Learning (Feature Extraction)

| | |
|---|---|
| **But** | Mesurer l'apport des features génériques d'ImageNet sans les modifier. |
| **Architecture** | ResNet50 (ou ResNet18) pré-entraîné sur ImageNet. **Toutes les couches de la backbone sont gelées.** Seule la tête de classification (fully-connected) est entraînée. |
| **Data Augmentation** | ✅ Oui (même pipeline que Groupe A). |
| **Hypothèse testée** | *Les features visuelles génériques (bords, textures, formes) suffisent-elles pour distinguer les fleurs ?* |
| **Ce qu'on apprend** | Si forte amélioration → la tâche bénéficie énormément des connaissances pré-entraînées. Si faible → les fleurs sont trop spécifiques pour les features génériques. |

---

### 🔷 Groupe B' — Transfer Learning sans Data Augmentation *(optionnel)*

| | |
|---|---|
| **But** | Isoler l'interaction entre transfer learning et augmentation. |
| **Architecture** | Identique au Groupe B (ResNet50 gelé). |
| **Data Augmentation** | ❌ Non (données brutes uniquement). |
| **Hypothèse testée** | *Le transfer learning compense-t-il le manque d'augmentation ? Ou l'augmentation reste-t-elle indispensable même avec un modèle pré-entraîné ?* |
| **Ce qu'on apprend** | Comparer B vs B' montre si l'augmentation est encore utile quand on a déjà des features riches. Comparer B' vs Baseline isole le pur effet du transfer learning. |

---

### 🔷 Groupe C — Fine-tuning

| | |
|---|---|
| **But** | Mesurer le gain maximal en adaptant les features au domaine spécifique (fleurs). |
| **Architecture** | ResNet50 pré-entraîné. **Dégel progressif** : d'abord les dernières couches (couches hautes), puis éventuellement tout le réseau avec un learning rate très faible. |
| **Data Augmentation** | ✅ Oui (même pipeline). |
| **Hypothèse testée** | *Adapter les features internes au domaine des fleurs apporte-t-il un gain significatif par rapport à les garder figées ?* |
| **Ce qu'on apprend** | Si C &gt; B de manière significative → les fleurs nécessitent des features spécialisées. Si C ≈ B → les features génériques suffisent et le fine-tuning n'est pas nécessaire (ou overfitte). |

---

## 3. Protocole expérimental strict

| Règle | Justification |
|-------|---------------|
| **Même split train/val/test** pour tous les groupes | Évite que les différences de performance viennent d'une répartition différente des données. |
| **Mêmes seeds** (PyTorch/TensorFlow + NumPy) | Garantit la reproductibilité. |
| **Même nombre d'époques** ou **même politique d'early stopping** | Évite qu'un groupe soit avantagé par un entraînement plus long. |
| **Même optimiseur et scheduler** (quand applicable) | Le learning rate du fine-tuning (C) sera plus faible, mais l'optimiseur reste le même. |
| **3 runs par groupe** (seeds différentes) | Permet de calculer moyenne ± écart-type et de parler de significativité statistique. |

---

## 4. Métriques à rapporter

| Métrique | Pourquoi |
|----------|----------|
| **Accuracy (Top-1)** | Métrique standard, facile à interpréter. |
| **F1-score macro** | Essentiel car tu surveilles le balancement. Plus honnête que l'accuracy sur dataset déséquilibré. |
| **Matrice de confusion** | Identifie les classes systématiquement confondues. |
| **Loss train / val** | Détecte underfitting (courbes élevées) ou overfitting (divergence train/val). |
| **Temps d'entraînement** | Le fine-tuning est plus lent que la feature extraction — à documenter. |

---

## 5. Interprétation attendue des résultats

| Scénario | Interprétation |
|----------|----------------|
| **Baseline &lt;&lt; A** | L'augmentation est cruciale (manque de données). |
| **A &lt;&lt; B** | Le modèle from scratch est sous-capacitaire. Le transfer learning est indispensable. |
| **B ≈ B'** | L'augmentation n'apporte plus rien quand on a des features pré-entraînées. |
| **B &lt;&lt; C** | Les features doivent être adaptées au domaine. Le fine-tuning est justifié. |
| **B ≈ C** | Les features génériques suffisent. Le fine-tuning est inutile ou overfitte. |

---

## 6. Checklist avant de lancer les expériences

- [ ] Dataset chargé et split fixé (mêmes indices pour tous les groupes)
- [ ] Vérification du balancement des classes (avant et après augmentation)
- [ ] Seeds fixés pour la reproductibilité
- [ ] Architecture de la baseline définie et testée (forward pass OK)
- [ ] Pipeline de data augmentation défini et documenté
- [ ] Modèle pré-entraîné choisi (ResNet50 recommandé) et chargé
- [ ] Stratégie de gel/dégel des couches définie pour B et C
- [ ] Hyperparamètres fixés (batch size, optimizer, scheduler, epochs)
- [ ] Script de logging des métriques prêt
- [ ] 3 seeds différentes choisies pour les runs multiples