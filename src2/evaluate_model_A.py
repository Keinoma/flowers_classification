#!/usr/bin/env python3
"""
evaluate_model_A.py
Comprehensive model evaluation script for the flower classification project.
Generates classification metrics, confusion matrix, K-Fold cross-validation,
temporal/spatial performance analysis, ROC-AUC, calibration, and top-k accuracy.
All outputs are saved to the results/ directory.

Run from src2/ with: python evaluate_model_A.py
"""

import os
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# sklearn metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, top_k_accuracy_score
)
from sklearn.model_selection import StratifiedKFold

# Plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Project imports
from load_data_A import load_flower_data
from model_A import create_model, FlowerCNN
from train_A import prepare_dataloaders, get_transforms, FlowerDataset

import argparse
warnings.filterwarnings('ignore')

eval_parser = argparse.ArgumentParser(description='Évaluation Groupe A')
eval_parser.add_argument('--seed', type=int, default=42, help='Seed du run à évaluer')
eval_args = eval_parser.parse_args()

# ============================================================
# CONFIGURATION
# ============================================================

SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
DATA_PATH = BASE_DIR / 'data' / '5_classes'
CHECKPOINT_PATH = SRC_DIR / 'checkpoints' / f'run_seed{eval_args.seed}' / 'best_model_A.pth'
RESULTS_DIR = SRC_DIR / 'results' / f'run_seed{eval_args.seed}'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
NUM_WORKERS = 4
NUM_CLASSES = 5
CLASS_NAMES = ['Lilly', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']

# K-Fold settings
K_FOLD_SPLITS = 5
K_FOLD_EPOCHS = 5  # Reduced epochs for cross-validation speed
K_FOLD_LR = 0.001

# Inference benchmark settings
BENCHMARK_BATCH_SIZES = [1, 8, 16, 32, 64]
BENCHMARK_WARMUP_RUNS = 100
BENCHMARK_MEASURE_RUNS = 1000

# Image size
IMAGE_SIZE = 224


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def ensure_dir(path: Path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def get_model_size_mb(model_path: Path) -> float:
    """Get model file size in megabytes."""
    return model_path.stat().st_size / (1024 * 1024)


def estimate_flops(model: nn.Module, input_size: tuple = (1, 3, IMAGE_SIZE, IMAGE_SIZE)) -> int:
    """
    Estimate FLOPs for the model by hooking into each layer.
    This is a manual estimation based on layer types.
    """
    total_flops = 0
    
    def conv_hook(module, input, output):
        nonlocal total_flops
        batch_size = output.shape[0]
        out_h, out_w = output.shape[2], output.shape[3]
        in_channels = module.in_channels
        out_channels = module.out_channels
        kernel_ops = module.kernel_size[0] * module.kernel_size[1]
        if module.groups > 1:
            in_channels = in_channels // module.groups
        flops = batch_size * out_channels * out_h * out_w * in_channels * kernel_ops
        total_flops += flops
    
    def linear_hook(module, input, output):
        nonlocal total_flops
        batch_size = input[0].shape[0]
        flops = batch_size * module.in_features * module.out_features
        total_flops += flops
    
    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
    
    dummy_input = torch.randn(input_size).to(next(model.parameters()).device)
    with torch.no_grad():
        model(dummy_input)
    
    for h in hooks:
        h.remove()
    
    return total_flops


def get_peak_memory_mb() -> Optional[float]:
    """Get peak GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return None


def convert_to_native(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ============================================================
# MODEL EVALUATOR CLASS
# ============================================================

class ModelEvaluator:
    """
    Comprehensive evaluator for the flower classification model.
    Handles all metrics, visualizations, and report generation.
    """
    
    def __init__(self):
        self.model: Optional[FlowerCNN] = None
        self.checkpoint: Optional[Dict] = None
        self.test_loader: Optional[DataLoader] = None
        self.test_images: Optional[np.ndarray] = None
        self.test_labels: Optional[np.ndarray] = None
        self.train_val_images: Optional[np.ndarray] = None
        self.train_val_labels: Optional[np.ndarray] = None
        self.all_preds: Optional[np.ndarray] = None
        self.all_labels: Optional[np.ndarray] = None
        self.all_probs: Optional[np.ndarray] = None
        
        # Results container
        self.results: Dict = {}
        
        # Ensure results directory exists
        ensure_dir(RESULTS_DIR)
        
    def load_data_and_model(self):
        """Load dataset, recreate test split, and load trained model."""
        print("=" * 70)
        print("  COMPREHENSIVE MODEL EVALUATION")
        print("=" * 70)
        
        # --- Load raw data ---
        print("\n[1/7] Loading dataset...")
        images, labels, class_names = load_flower_data(DATA_PATH, image_size=(224, 224))
                
        # --- Recreate the same split as training (seed=42) ---
        total = len(images)
        test_size = int(total * 0.1)
        val_size = int(total * 0.2)
        train_size = total - val_size - test_size
        
        from torch.utils.data import random_split
        train_indices, val_indices, test_indices = random_split(
            range(total),
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(eval_args.seed)
        )
        
        self.test_images = images[test_indices.indices]
        self.test_labels = labels[test_indices.indices]
        
        # Also keep train+val for K-Fold
        self.train_val_images = np.concatenate([
            images[train_indices.indices],
            images[val_indices.indices]
        ])
        self.train_val_labels = np.concatenate([
            labels[train_indices.indices],
            labels[val_indices.indices]
        ])
        
        # Create test DataLoader
        test_dataset = FlowerDataset(
            self.test_images, 
            self.test_labels, 
            transform=get_transforms(train=False)
        )
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False,
            num_workers=NUM_WORKERS, 
            pin_memory=True
        )
        
        print(f"   Test set: {len(test_dataset)} images")
        print(f"   Train+Val set (for K-Fold): {len(self.train_val_labels)} images")
        
        # --- Load model ---
        print("\n[2/7] Loading model...")
        self.model = create_model(num_classes=NUM_CLASSES, device=DEVICE)
        
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
        
        self.checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        
        epoch = self.checkpoint.get('epoch', 'N/A')
        val_acc = self.checkpoint.get('val_acc', 'N/A')
        print(f"   Model loaded (epoch {epoch}, val_acc: {val_acc:.2f}%)")
        
        # --- Run inference on test set ---
        print("\n[3/7] Running inference on test set...")
        self._run_inference()
        
    def _run_inference(self):
        """Run model inference on the test set and store predictions."""
        all_preds = []
        all_labels_list = []
        all_probs = []
        
        self.model.eval()
        with torch.no_grad():
            for imgs_batch, lbls_batch in tqdm(self.test_loader, desc="Inference"):
                imgs_batch = imgs_batch.to(DEVICE)
                outputs = self.model(imgs_batch)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels_list.extend(lbls_batch.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        self.all_preds = np.array(all_preds)
        self.all_labels = np.array(all_labels_list)
        self.all_probs = np.array(all_probs)
        
    # ============================================================
    # 1. CLASSIFICATION METRICS
    # ============================================================
    
    def evaluate_classification_metrics(self) -> Dict:
        """Compute standard classification metrics."""
        print("\n[4/7] Computing classification metrics...")
        
        acc = accuracy_score(self.all_labels, self.all_preds)
        
        # Per-class and averaged metrics
        precision_macro = precision_score(self.all_labels, self.all_preds, average='macro', zero_division=0)
        precision_micro = precision_score(self.all_labels, self.all_preds, average='micro', zero_division=0)
        precision_weighted = precision_score(self.all_labels, self.all_preds, average='weighted', zero_division=0)
        
        recall_macro = recall_score(self.all_labels, self.all_preds, average='macro', zero_division=0)
        recall_micro = recall_score(self.all_labels, self.all_preds, average='micro', zero_division=0)
        recall_weighted = recall_score(self.all_labels, self.all_preds, average='weighted', zero_division=0)
        
        f1_macro = f1_score(self.all_labels, self.all_preds, average='macro', zero_division=0)
        f1_micro = f1_score(self.all_labels, self.all_preds, average='micro', zero_division=0)
        f1_weighted = f1_score(self.all_labels, self.all_preds, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(self.all_labels, self.all_preds, average=None, zero_division=0)
        recall_per_class = recall_score(self.all_labels, self.all_preds, average=None, zero_division=0)
        f1_per_class = f1_score(self.all_labels, self.all_preds, average=None, zero_division=0)
        
        # Classification report as dict
        report = classification_report(
            self.all_labels, self.all_preds, 
            target_names=CLASS_NAMES, 
            output_dict=True,
            zero_division=0
        )
        
        metrics = {
            'accuracy': float(acc),
            'accuracy_percent': float(acc * 100),
            'precision': {
                'macro': float(precision_macro),
                'micro': float(precision_micro),
                'weighted': float(precision_weighted),
                'per_class': {name: float(v) for name, v in zip(CLASS_NAMES, precision_per_class)}
            },
            'recall': {
                'macro': float(recall_macro),
                'micro': float(recall_micro),
                'weighted': float(recall_weighted),
                'per_class': {name: float(v) for name, v in zip(CLASS_NAMES, recall_per_class)}
            },
            'f1_score': {
                'macro': float(f1_macro),
                'micro': float(f1_micro),
                'weighted': float(f1_weighted),
                'per_class': {name: float(v) for name, v in zip(CLASS_NAMES, f1_per_class)}
            },
            'classification_report': convert_to_native(report),
            'validation_loss': None,
            'validation_accuracy': float(self.checkpoint.get('val_acc', 0)) if self.checkpoint else None
        }
        
        # Try to get val loss from checkpoint if stored
        if self.checkpoint and 'val_loss' in self.checkpoint:
            metrics['validation_loss'] = float(self.checkpoint['val_loss'])
        
        self.results['classification_metrics'] = metrics
        
        # Console output
        print(f"\n   Accuracy: {acc*100:.2f}%")
        print(f"   Precision (macro): {precision_macro:.4f}")
        print(f"   Recall (macro):    {recall_macro:.4f}")
        print(f"   F1-Score (macro):  {f1_macro:.4f}")
        print(f"   F1-Score (weighted): {f1_weighted:.4f}")
        
        return metrics
    
    # ============================================================
    # 2. CONFUSION MATRIX
    # ============================================================
    
    def evaluate_confusion_matrix(self):
        """Generate and save confusion matrix (raw + normalized)."""
        print("\n[*] Generating confusion matrix...")
        
        cm = confusion_matrix(self.all_labels, self.all_preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)
        
        # Save as CSV
        csv_path = RESULTS_DIR / 'confusion_matrix_A.csv'
        np.savetxt(csv_path, cm, delimiter=',', fmt='%d', header=','.join(CLASS_NAMES), comments='')
        print(f"   CSV saved: {csv_path}")
        
        # --- Raw confusion matrix plot ---
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_title('Confusion Matrix_A', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax)
        
        tick_marks = np.arange(len(CLASS_NAMES))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color=color, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        raw_path = RESULTS_DIR / 'confusion_matrix_A.png'
        plt.savefig(raw_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Raw plot saved: {raw_path}")
        
        # --- Normalized confusion matrix plot ---
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm_normalized, cmap='Blues', vmin=0, vmax=1)
        ax.set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax)
        
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        
        for i in range(cm_normalized.shape[0]):
            for j in range(cm_normalized.shape[1]):
                color = 'white' if cm_normalized[i, j] > 0.5 else 'black'
                ax.text(j, i, f"{cm_normalized[i, j]:.2f}", ha='center', va='center',
                        color=color, fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        norm_path = RESULTS_DIR / 'confusion_matrix_normalized_A.png'
        plt.savefig(norm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Normalized plot saved: {norm_path}")
        
        self.results['confusion_matrix'] = {
            'raw': cm.tolist(),
            'normalized': cm_normalized.tolist()
        }
    
    # ============================================================
    # 3. ROC-AUC CURVES
    # ============================================================
    
    def evaluate_roc_auc(self):
        """Compute ROC curves and AUC for each class (One-vs-Rest)."""
        print("\n[*] Computing ROC-AUC curves...")
        
        # One-hot encode labels
        labels_onehot = np.zeros((len(self.all_labels), NUM_CLASSES))
        labels_onehot[np.arange(len(self.all_labels)), self.all_labels] = 1
        
        # Compute ROC curve and AUC for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i, class_name in enumerate(CLASS_NAMES):
            fpr[i], tpr[i], _ = roc_curve(labels_onehot[:, i], self.all_probs[:, i])
            roc_auc[i] = roc_auc_score(labels_onehot[:, i], self.all_probs[:, i])
        
        # Macro-average ROC-AUC
        macro_roc_auc = roc_auc_score(labels_onehot, self.all_probs, average='macro', multi_class='ovr')
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
        
        for i, class_name in enumerate(CLASS_NAMES):
            ax.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                    label=f"{class_name} (AUC = {roc_auc[i]:.3f})")
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        roc_path = RESULTS_DIR / 'roc_curves_A.png'
        plt.savefig(roc_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ROC curves saved: {roc_path}")
        
        self.results['roc_auc'] = {
            'per_class': {name: float(roc_auc[i]) for i, name in enumerate(CLASS_NAMES)},
            'macro_average': float(macro_roc_auc)
        }
        
        print(f"   Macro-average ROC-AUC: {macro_roc_auc:.4f}")
    
    # ============================================================
    # 4. CALIBRATION (Expected Calibration Error)
    # ============================================================
    
    def evaluate_calibration(self, n_bins: int = 10):
        """Compute Expected Calibration Error (ECE) and plot reliability diagram."""
        print("\n[*] Computing calibration metrics...")
        
        # Get confidence (max probability) and correctness
        confidences = np.max(self.all_probs, axis=1)
        predictions = np.argmax(self.all_probs, axis=1)
        correct = (predictions == self.all_labels).astype(float)
        
        # ECE calculation
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        bin_accs = []
        bin_confs = []
        bin_counts = []
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                bin_accs.append(accuracy_in_bin)
                bin_confs.append(avg_confidence_in_bin)
                bin_counts.append(in_bin.sum())
            else:
                bin_accs.append(0)
                bin_confs.append(0)
                bin_counts.append(0)
        
        # Reliability diagram
        fig, ax = plt.subplots(figsize=(8, 6))
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        ax.bar(bin_centers, bin_accs, width=0.08, alpha=0.7, color='steelblue', edgecolor='black', label='Accuracy')
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'Reliability Diagram (ECE = {ece:.4f})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        calib_path = RESULTS_DIR / 'reliability_diagram_A.png'
        plt.savefig(calib_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Reliability diagram saved: {calib_path}")
        
        self.results['calibration'] = {
            'expected_calibration_error': float(ece),
            'n_bins': n_bins,
            'bin_accuracies': [float(x) for x in bin_accs],
            'bin_confidences': [float(x) for x in bin_confs],
            'bin_counts': [int(x) for x in bin_counts]
        }
        
        print(f"   Expected Calibration Error (ECE): {ece:.4f}")
    
    # ============================================================
    # 5. TOP-K ACCURACY
    # ============================================================
    
    def evaluate_top_k_accuracy(self):
        """Compute Top-2 and Top-3 accuracy."""
        print("\n[*] Computing Top-K accuracy...")
        
        top2_acc = top_k_accuracy_score(self.all_labels, self.all_probs, k=2, labels=range(NUM_CLASSES))
        top3_acc = top_k_accuracy_score(self.all_labels, self.all_probs, k=3, labels=range(NUM_CLASSES))
        
        self.results['top_k_accuracy'] = {
            'top1': float(accuracy_score(self.all_labels, self.all_preds)),
            'top2': float(top2_acc),
            'top3': float(top3_acc)
        }
        
        print(f"   Top-1 Accuracy: {self.results['top_k_accuracy']['top1']*100:.2f}%")
        print(f"   Top-2 Accuracy: {top2_acc*100:.2f}%")
        print(f"   Top-3 Accuracy: {top3_acc*100:.2f}%")
    
    # ============================================================
    # 6. K-FOLD CROSS-VALIDATION
    # ============================================================
    
    """def evaluate_kfold_cross_validation(self):

        #Perform Stratified K-Fold cross-validation on the train+val set.
        #This estimates model robustness and generalization variance.
        #Each fold trains a fresh model from scratch with reduced epochs.

        print(f"\n[*] Running {K_FOLD_SPLITS}-Fold Stratified Cross-Validation...")
        print("   (Training fresh models from scratch on each fold)")
        
        skf = StratifiedKFold(n_splits=K_FOLD_SPLITS, shuffle=True, random_state=42)
        fold_results = []
        fold_accuracies = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(self.train_val_images, self.train_val_labels)):
            print(f"\n   Fold {fold_idx + 1}/{K_FOLD_SPLITS}")
            
            # Create datasets for this fold
            fold_train_images = self.train_val_images[train_idx]
            fold_train_labels = self.train_val_labels[train_idx]
            fold_val_images = self.train_val_images[val_idx]
            fold_val_labels = self.train_val_labels[val_idx]
            
            train_dataset = FlowerDataset(
                fold_train_images, fold_train_labels,
                transform=get_transforms(train=True)
            )
            val_dataset = FlowerDataset(
                fold_val_images, fold_val_labels,
                transform=get_transforms(train=False)
            )
            
            train_loader = DataLoader(
                train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                num_workers=NUM_WORKERS, pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=True
            )
            
            # Fresh model for this fold
            fold_model = create_model(num_classes=NUM_CLASSES, device=DEVICE)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(fold_model.parameters(), lr=K_FOLD_LR, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=2, factor=0.5, verbose=False
            )
            
            # Training loop (reduced epochs)
            best_fold_acc = 0.0
            for epoch in range(K_FOLD_EPOCHS):
                # Train
                fold_model.train()
                train_loss = 0.0
                for images, labels in train_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = fold_model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validate
                fold_model.eval()
                val_loss = 0.0
                all_fold_preds = []
                all_fold_labels = []
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(DEVICE), labels.to(DEVICE)
                        outputs = fold_model(images)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        all_fold_preds.extend(predicted.cpu().numpy())
                        all_fold_labels.extend(labels.cpu().numpy())
                
                val_loss /= len(val_loader)
                val_acc = accuracy_score(all_fold_labels, all_fold_preds) * 100
                scheduler.step(val_loss)
                
                if val_acc > best_fold_acc:
                    best_fold_acc = val_acc
            
            fold_results.append({
                'fold': fold_idx + 1,
                'best_val_accuracy': float(best_fold_acc),
                'train_size': len(train_idx),
                'val_size': len(val_idx)
            })
            fold_accuracies.append(best_fold_acc)
            print(f"      Best Val Accuracy: {best_fold_acc:.2f}%")
        
        # Aggregate statistics
        mean_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)
        ci_95 = 1.96 * std_acc / np.sqrt(K_FOLD_SPLITS)
        
        self.results['kfold_cross_validation'] = {
            'n_splits': K_FOLD_SPLITS,
            'epochs_per_fold': K_FOLD_EPOCHS,
            'fold_results': fold_results,
            'mean_accuracy': float(mean_acc),
            'std_accuracy': float(std_acc),
            'ci_95_lower': float(mean_acc - ci_95),
            'ci_95_upper': float(mean_acc + ci_95)
        }
        
        # Boxplot
        fig, ax = plt.subplots(figsize=(8, 6))
        bp = ax.boxplot(fold_accuracies, patch_artist=True,
                        boxprops=dict(facecolor='steelblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))
        ax.scatter([1] * len(fold_accuracies), fold_accuracies, color='black', alpha=0.6, zorder=3)
        ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
        ax.set_title(f'{K_FOLD_SPLITS}-Fold Cross-Validation Results', fontsize=14, fontweight='bold')
        ax.set_xticklabels(['All Folds'])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add mean and std text
        ax.text(1.15, mean_acc, f'Mean: {mean_acc:.2f}%\nStd: {std_acc:.2f}%',
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        kfold_path = RESULTS_DIR / 'kfold_boxplot_A.png'
        plt.savefig(kfold_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   K-Fold plot saved: {kfold_path}")
        print(f"\n   K-Fold Summary:")
        print(f"      Mean Accuracy: {mean_acc:.2f}% (+/- {std_acc:.2f}%)")
        print(f"      95% CI: [{mean_acc - ci_95:.2f}%, {mean_acc + ci_95:.2f}%]")
        
        """
    
    # ============================================================
    # 7. TEMPORAL PERFORMANCE (Inference Speed)
    # ============================================================
    
    def evaluate_temporal_performance(self):
        """
        Measure inference speed: latency, throughput, and percentiles.
        """
        print("\n[*] Measuring temporal performance (inference speed)...")
        
        self.model.eval()
        temporal_results = {}
        
        # Single-image latency
        print("   Benchmarking single-image inference...")
        dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
        
        # Warm-up
        with torch.no_grad():
            for _ in range(BENCHMARK_WARMUP_RUNS):
                _ = self.model(dummy_input)
        
        # Synchronize GPU if available
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Measure single-image latency
        latencies = []
        with torch.no_grad():
            for _ in range(BENCHMARK_MEASURE_RUNS):
                start = time.perf_counter()
                _ = self.model(dummy_input)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # ms
        
        latencies = np.array(latencies)
        temporal_results['single_image'] = {
            'mean_latency_ms': float(np.mean(latencies)),
            'std_latency_ms': float(np.std(latencies)),
            'p50_latency_ms': float(np.percentile(latencies, 50)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
            'min_latency_ms': float(np.min(latencies)),
            'max_latency_ms': float(np.max(latencies)),
            'throughput_images_per_sec': float(1000.0 / np.mean(latencies))
        }
        
        print(f"   Single-image mean latency: {temporal_results['single_image']['mean_latency_ms']:.3f} ms")
        print(f"   P95 latency: {temporal_results['single_image']['p95_latency_ms']:.3f} ms")
        print(f"   Throughput: {temporal_results['single_image']['throughput_images_per_sec']:.1f} img/s")
        
        # Batch throughput benchmark
        print("   Benchmarking batch throughput...")
        batch_results = []
        for bs in BENCHMARK_BATCH_SIZES:
            dummy_batch = torch.randn(bs, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
            
            # Warm-up
            with torch.no_grad():
                for _ in range(10):
                    _ = self.model(dummy_batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Measure
            n_runs = max(1, 500 // bs)
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(n_runs):
                    _ = self.model(dummy_batch)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
            end = time.perf_counter()
            
            total_time = end - start
            throughput = (bs * n_runs) / total_time
            latency_per_image = (total_time * 1000) / (bs * n_runs)
            
            batch_results.append({
                'batch_size': bs,
                'throughput_images_per_sec': float(throughput),
                'latency_per_image_ms': float(latency_per_image)
            })
            print(f"      Batch {bs:2d}: {throughput:.1f} img/s ({latency_per_image:.3f} ms/img)")
        
        temporal_results['batch_benchmark'] = batch_results
        
        # Plot throughput vs batch size
        fig, ax = plt.subplots(figsize=(8, 6))
        batch_sizes = [r['batch_size'] for r in batch_results]
        throughputs = [r['throughput_images_per_sec'] for r in batch_results]
        ax.plot(batch_sizes, throughputs, 'o-', color='steelblue', linewidth=2, markersize=8)
        ax.set_xlabel('Batch Size', fontsize=12)
        ax.set_ylabel('Throughput (images/sec)', fontsize=12)
        ax.set_title('Inference Throughput vs Batch Size', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        for bs, tp in zip(batch_sizes, throughputs):
            ax.text(bs, tp + max(throughputs)*0.02, f'{tp:.1f}', ha='center', fontsize=9)
        plt.tight_layout()
        throughput_path = RESULTS_DIR / 'inference_throughput_A.png'
        plt.savefig(throughput_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Throughput plot saved: {throughput_path}")
        
        self.results['temporal_performance'] = temporal_results
    
    # ============================================================
    # 8. SPATIAL PERFORMANCE (Model Size, Memory, FLOPs)
    # ============================================================
    
    def evaluate_spatial_performance(self):
        """
        Measure spatial performance: parameter count, model size, memory, FLOPs.
        """
        print("\n[*] Measuring spatial performance (model size, memory, FLOPs)...")
        
        # Parameter counts
        param_stats = self.model.count_parameters()
        
        # Model file size
        model_size_mb = get_model_size_mb(CHECKPOINT_PATH)
        
        # FLOPs estimation
        flops = estimate_flops(self.model)
        gflops = flops / 1e9
        
        # Memory usage during inference
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
            with torch.no_grad():
                _ = self.model(dummy_input)
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            torch.cuda.reset_peak_memory_stats()
        else:
            peak_memory_mb = None
        
        # Model complexity score (accuracy / log(params))
        test_acc = self.results.get('classification_metrics', {}).get('accuracy', 0)
        complexity_score = test_acc / np.log1p(param_stats['total']) if param_stats['total'] > 0 else 0
        
        spatial_results = {
            'parameters': {
                'total': int(param_stats['total']),
                'trainable': int(param_stats['trainable']),
                'frozen': int(param_stats['frozen']),
                'trainable_percent': float(param_stats['trainable_percent'])
            },
            'model_size_mb': float(model_size_mb),
            'flops': int(flops),
            'gflops': float(gflops),
            'peak_inference_memory_mb': float(peak_memory_mb) if peak_memory_mb else None,
            'complexity_score': float(complexity_score),
            'device': DEVICE
        }
        
        self.results['spatial_performance'] = spatial_results
        
        print(f"   Total parameters: {param_stats['total']:,}")
        print(f"   Trainable parameters: {param_stats['trainable']:,} ({param_stats['trainable_percent']:.1f}%)")
        print(f"   Model file size: {model_size_mb:.2f} MB")
        print(f"   Estimated FLOPs: {gflops:.3f} GFLOPs")
        if peak_memory_mb:
            print(f"   Peak GPU memory (inference): {peak_memory_mb:.2f} MB")
        print(f"   Complexity score (acc/log(params)): {complexity_score:.6f}")
    
    # ============================================================
    # 9. CONFIDENCE DISTRIBUTION
    # ============================================================
    
    def evaluate_confidence_distribution(self):
        """Plot distribution of confidence scores for correct vs incorrect predictions."""
        print("\n[*] Analyzing confidence distribution...")
        
        confidences = np.max(self.all_probs, axis=1)
        correct_mask = (self.all_preds == self.all_labels)
        
        correct_conf = confidences[correct_mask]
        incorrect_conf = confidences[~correct_mask]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bins = np.linspace(0, 1, 21)
        ax.hist(correct_conf, bins=bins, alpha=0.7, color='green', label=f'Correct ({len(correct_conf)})', edgecolor='black')
        if len(incorrect_conf) > 0:
            ax.hist(incorrect_conf, bins=bins, alpha=0.7, color='red', label=f'Incorrect ({len(incorrect_conf)})', edgecolor='black')
        
        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Confidence Distribution: Correct vs Incorrect Predictions', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add mean lines
        if len(correct_conf) > 0:
            ax.axvline(np.mean(correct_conf), color='darkgreen', linestyle='--', linewidth=2, label=f'Mean correct: {np.mean(correct_conf):.3f}')
        if len(incorrect_conf) > 0:
            ax.axvline(np.mean(incorrect_conf), color='darkred', linestyle='--', linewidth=2, label=f'Mean incorrect: {np.mean(incorrect_conf):.3f}')
        
        plt.tight_layout()
        conf_path = RESULTS_DIR / 'confidence_distribution_A.png'
        plt.savefig(conf_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Confidence distribution saved: {conf_path}")
        
        self.results['confidence_distribution'] = {
            'mean_confidence_correct': float(np.mean(correct_conf)) if len(correct_conf) > 0 else None,
            'mean_confidence_incorrect': float(np.mean(incorrect_conf)) if len(incorrect_conf) > 0 else None,
            'num_correct': int(len(correct_conf)),
            'num_incorrect': int(len(incorrect_conf))
        }
    
    # ============================================================
    # 10. PER-CLASS ERROR ANALYSIS
    # ============================================================
    
    def evaluate_per_class_errors(self):
        """Analyze which classes are most confused."""
        print("\n[*] Performing per-class error analysis...")
        
        cm = confusion_matrix(self.all_labels, self.all_preds)
        per_class_error = {}
        most_confused_pairs = []
        
        for i, class_name in enumerate(CLASS_NAMES):
            total = cm[i].sum()
            correct = cm[i, i]
            error_rate = (total - correct) / total if total > 0 else 0
            per_class_error[class_name] = {
                'total_samples': int(total),
                'correct': int(correct),
                'error_rate': float(error_rate)
            }
            
            # Find most confused class for this class
            off_diagonal = cm[i].copy()
            off_diagonal[i] = 0
            if off_diagonal.sum() > 0:
                most_confused_idx = np.argmax(off_diagonal)
                most_confused_pairs.append({
                    'true_class': class_name,
                    'predicted_class': CLASS_NAMES[most_confused_idx],
                    'count': int(off_diagonal[most_confused_idx]),
                    'error_rate': float(off_diagonal[most_confused_idx] / total) if total > 0 else 0
                })
        
        # Sort by error rate
        most_confused_pairs.sort(key=lambda x: x['error_rate'], reverse=True)
        
        self.results['per_class_error_analysis'] = {
            'per_class': per_class_error,
            'most_confused_pairs': most_confused_pairs
        }
        
        print("   Per-class error rates:")
        for name, stats in per_class_error.items():
            print(f"      {name:12s}: {stats['error_rate']*100:.1f}% error ({stats['correct']}/{stats['total_samples']})")
        
        if most_confused_pairs:
            print(f"   Most confused pair: {most_confused_pairs[0]['true_class']} -> {most_confused_pairs[0]['predicted_class']} "
                  f"({most_confused_pairs[0]['count']} cases)")
    
    # ============================================================
    # 11. GENERATE REPORTS
    # ============================================================
    
    def generate_reports(self):
        """Save JSON report and formatted text summary."""
        print("\n[*] Generating evaluation reports...")
        
        # --- JSON Report ---
        json_path = RESULTS_DIR / 'evaluation_report_A.json'
        with open(json_path, 'w') as f:
            json.dump(convert_to_native(self.results), f, indent=2)
        print(f"   JSON report saved: {json_path}")
        
        # --- Text Summary ---
        summary_lines = []
        summary_lines.append("=" * 70)
        summary_lines.append("  COMPREHENSIVE MODEL EVALUATION REPORT")
        summary_lines.append("=" * 70)
        summary_lines.append("")
        summary_lines.append(f"Model: FlowerCNN (Groupe A — Data Augmentation)")
        summary_lines.append(f"Device: {DEVICE}")
        summary_lines.append(f"Checkpoint: {CHECKPOINT_PATH}")
        summary_lines.append(f"Test Set Size: {len(self.test_labels)} images")
        summary_lines.append("")
        
        # Classification Metrics
        cm = self.results.get('classification_metrics', {})
        summary_lines.append("-" * 70)
        summary_lines.append("CLASSIFICATION METRICS")
        summary_lines.append("-" * 70)
        summary_lines.append(f"Accuracy:           {cm.get('accuracy_percent', 0):.2f}%")
        summary_lines.append(f"Precision (macro):  {cm.get('precision', {}).get('macro', 0):.4f}")
        summary_lines.append(f"Recall (macro):     {cm.get('recall', {}).get('macro', 0):.4f}")
        summary_lines.append(f"F1-Score (macro):   {cm.get('f1_score', {}).get('macro', 0):.4f}")
        summary_lines.append(f"F1-Score (weighted): {cm.get('f1_score', {}).get('weighted', 0):.4f}")
        summary_lines.append(f"Validation Acc (checkpoint): {cm.get('validation_accuracy', 0):.2f}%")
        summary_lines.append("")
        
        # Top-K
        tk = self.results.get('top_k_accuracy', {})
        summary_lines.append("Top-K Accuracy:")
        summary_lines.append(f"   Top-1: {tk.get('top1', 0)*100:.2f}%")
        summary_lines.append(f"   Top-2: {tk.get('top2', 0)*100:.2f}%")
        summary_lines.append(f"   Top-3: {tk.get('top3', 0)*100:.2f}%")
        summary_lines.append("")
        
        # ROC-AUC
        roc = self.results.get('roc_auc', {})
        summary_lines.append(f"ROC-AUC (macro): {roc.get('macro_average', 0):.4f}")
        summary_lines.append("")
        
        # Calibration
        cal = self.results.get('calibration', {})
        summary_lines.append(f"Expected Calibration Error (ECE): {cal.get('expected_calibration_error', 0):.4f}")
        summary_lines.append("")
        
        # K-Fold
        kf = self.results.get('kfold_cross_validation', {})
        summary_lines.append("-" * 70)
        summary_lines.append("K-FOLD CROSS-VALIDATION")
        summary_lines.append("-" * 70)
        summary_lines.append(f"Folds: {kf.get('n_splits', 0)} | Epochs per fold: {kf.get('epochs_per_fold', 0)}")
        summary_lines.append(f"Mean Accuracy: {kf.get('mean_accuracy', 0):.2f}%")
        summary_lines.append(f"Std Accuracy:  {kf.get('std_accuracy', 0):.2f}%")
        summary_lines.append(f"95% CI: [{kf.get('ci_95_lower', 0):.2f}%, {kf.get('ci_95_upper', 0):.2f}%]")
        summary_lines.append("")
        
        # Temporal
        tp = self.results.get('temporal_performance', {})
        si = tp.get('single_image', {})
        summary_lines.append("-" * 70)
        summary_lines.append("TEMPORAL PERFORMANCE")
        summary_lines.append("-" * 70)
        summary_lines.append(f"Single-image latency: {si.get('mean_latency_ms', 0):.3f} ms (+/- {si.get('std_latency_ms', 0):.3f})")
        summary_lines.append(f"P95 latency:          {si.get('p95_latency_ms', 0):.3f} ms")
        summary_lines.append(f"P99 latency:          {si.get('p99_latency_ms', 0):.3f} ms")
        summary_lines.append(f"Throughput:           {si.get('throughput_images_per_sec', 0):.1f} images/sec")
        summary_lines.append("")
        
        # Spatial
        sp = self.results.get('spatial_performance', {})
        summary_lines.append("-" * 70)
        summary_lines.append("SPATIAL PERFORMANCE")
        summary_lines.append("-" * 70)
        params = sp.get('parameters', {})
        summary_lines.append(f"Total parameters:     {params.get('total', 0):,}")
        summary_lines.append(f"Trainable parameters: {params.get('trainable', 0):,} ({params.get('trainable_percent', 0):.1f}%)")
        summary_lines.append(f"Model file size:      {sp.get('model_size_mb', 0):.2f} MB")
        summary_lines.append(f"Estimated FLOPs:      {sp.get('gflops', 0):.3f} GFLOPs")
        if sp.get('peak_inference_memory_mb'):
            summary_lines.append(f"Peak GPU memory:      {sp['peak_inference_memory_mb']:.2f} MB")
        summary_lines.append(f"Complexity score:     {sp.get('complexity_score', 0):.6f}")
        summary_lines.append("")
        
        # Per-class errors
        pce = self.results.get('per_class_error_analysis', {})
        summary_lines.append("-" * 70)
        summary_lines.append("PER-CLASS ERROR ANALYSIS")
        summary_lines.append("-" * 70)
        for name, stats in pce.get('per_class', {}).items():
            summary_lines.append(f"{name:12s}: {stats['error_rate']*100:5.1f}% error")
        summary_lines.append("")
        
        summary_lines.append("-" * 70)
        summary_lines.append("Generated Files:")
        summary_lines.append("-" * 70)
        for f in sorted(RESULTS_DIR.iterdir()):
            if f.is_file():
                summary_lines.append(f"   {f.name}")
        summary_lines.append("")
        summary_lines.append("=" * 70)
        summary_lines.append("  EVALUATION COMPLETE")
        summary_lines.append("=" * 70)
        
        summary_text = "\n".join(summary_lines)
        txt_path = RESULTS_DIR / 'evaluation_summary_A.txt'
        with open(txt_path, 'w') as f:
            f.write(summary_text)
        print(f"   Text summary saved: {txt_path}")
        
        # Print to console
        print("\n" + summary_text)
    
    # ============================================================
    # MAIN EVALUATION PIPELINE
    # ============================================================
    
    def run_full_evaluation(self):
        """Run the complete evaluation pipeline."""
        self.load_data_and_model()
        self.evaluate_classification_metrics()
        self.evaluate_confusion_matrix()
        self.evaluate_roc_auc()
        self.evaluate_calibration()
        self.evaluate_top_k_accuracy()
        #self.evaluate_kfold_cross_validation()
        self.evaluate_temporal_performance()
        self.evaluate_spatial_performance()
        self.evaluate_confidence_distribution()
        self.evaluate_per_class_errors()
        self.generate_reports()


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    evaluator = ModelEvaluator()
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()