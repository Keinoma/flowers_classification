# Ablation Study Protocol — Flower Classification

## 1. Overview of groups

| Group | Name | Main objective | Architecture | Data Aug. | Transfer Learning | Fine-tuning |
|--------|-----|-------------------|--------------|-----------|-------------------|-------------|
| Baseline | CNN From Scratch | Establish the minimal reference | Simple CNN (3-4 conv blocks) | No | No | No |
| Group A | + Data Augmentation | Measure the impact of data augmentation | Same CNN as baseline | Yes | No | No |
| Group B | + Transfer Learning (frozen) | Measure the impact of pretrained features | ResNet50 (frozen backbone) | Yes | Yes | No |
| Group B' | Transfer Learning alone (optional) | Isolate the effect of transfer learning without augmentation | ResNet50 (frozen backbone) | No | Yes | No |
| Group C | + Fine-tuning | Measure the gain from adapting the features | ResNet50 (unfrozen backbone) | Yes | Yes | Yes |

## 2. Detail of each group

### Baseline — CNN From Scratch - src1

| | |
|---|---|
| Purpose | Establish the minimum achievable performance with a naive model and raw data. |
| Architecture | Simple CNN: 3-4 Conv2D blocks -> BatchNorm -> ReLU -> MaxPool, followed by a fully-connected classifier. No residual connections. |
| Training | End-to-end training from scratch. |
| Hypothesis tested | What performance is achievable without any assistance (no augmentation, no prior knowledge)? |
| Learning outcome | The performance ceiling of the model alone. Mandatory comparison point for all other groups. |

### Group A — Baseline + Data Augmentation - src2

| | |
|---|---|
| Purpose | Quantify the gain from data augmentation alone. |
| Architecture | Identical to baseline (same simple CNN). |
| Data augmentation | Rotation, horizontal flip, light zoom, optionally color jitter. Check class balance after augmentation. |
| Hypothesis tested | Is data augmentation sufficient to significantly improve generalization? |
| Learning outcome | If the gain is large, the model was suffering from a lack of data. If the gain is small, the model is the bottleneck (capacity issue). |

### Group B — Transfer Learning (Feature Extraction) - src3

| | |
|---|---|
| Purpose | Measure the contribution of generic ImageNet features without modifying them. |
| Architecture | ResNet50 (or ResNet18) pretrained on ImageNet. All backbone layers are frozen. Only the classification head (fully-connected) is trained. |
| Data augmentation | Yes (same pipeline as Group A). |
| Hypothesis tested | Are generic visual features (edges, textures, shapes) sufficient to distinguish flowers? |
| Learning outcome | If improvement is strong, the task benefits greatly from pretrained knowledge. If weak, flowers are too specific for generic features. |

### Group B' — Transfer Learning without Data Augmentation - src4

| | |
|---|---|
| Purpose | Isolate the interaction between transfer learning and augmentation. |
| Architecture | Identical to Group B (frozen ResNet50). |
| Data augmentation | No (raw data only). |
| Hypothesis tested | Does transfer learning compensate for the lack of augmentation? Or is augmentation still necessary even with a pretrained model? |
| Learning outcome | Comparing B vs B' shows whether augmentation is still useful when rich features are already available. Comparing B' vs Baseline isolates the pure effect of transfer learning. |

### Group C — Fine-tuning - src5

| | |
|---|---|
| Purpose | Measure the maximum gain from adapting features to the specific domain (flowers). |
| Architecture | Pretrained ResNet50. Progressive unfreezing: first the last layers (high-level layers), then possibly the entire network with a very low learning rate. |
| Data augmentation | Yes (same pipeline). |
| Hypothesis tested | Does adapting internal features to the flower domain provide a significant gain compared to keeping them frozen? |
| Learning outcome | If C > B significantly, flowers require specialized features. If C ≈ B, generic features are sufficient and fine-tuning is not necessary (or overfits). |

## 3. Strict experimental protocol

| Rule | Justification |
|-------|---------------|
| Same train/val/test split for all groups | Prevents performance differences from coming from a different data split. |
| Same seeds (PyTorch/TensorFlow + NumPy) | Guarantees reproducibility. |
| Same number of epochs or same early stopping policy | Prevents any group from being advantaged by longer training. |
| Same optimizer and scheduler (where applicable) | The fine-tuning (C) learning rate will be lower, but the optimizer stays the same. |
| 3 runs per group (different seeds) | Allows computing mean ± standard deviation and discussing statistical significance. |

## 4. Metrics to report

| Metric | Reason |
|----------|----------|
| Accuracy (Top-1) | Standard metric, easy to interpret. |
| F1-score macro | Essential given the class balance concern. More honest than accuracy on an imbalanced dataset. |
| Confusion matrix | Identifies systematically confused classes. |
| Train / val loss | Detects underfitting (high curves) or overfitting (train/val divergence). |
| Training time | Fine-tuning is slower than feature extraction — should be documented. |

## 5. Expected interpretation of results

| Scenario | Interpretation |
|----------|----------------|
| Baseline << A | Augmentation is critical (data shortage). |
| A << B | The from-scratch model is under-capacity. Transfer learning is indispensable. |
| B ≈ B' | Augmentation adds nothing once pretrained features are available. |
| B << C | Features must be adapted to the domain. Fine-tuning is justified. |
| B ≈ C | Generic features are sufficient. Fine-tuning is unnecessary or overfits. |


## 6. If I have enough time :
Trying to make compete the two best models on the oxford 102 classes dataset
Using k-fold cross-validation to measure their performances. 

