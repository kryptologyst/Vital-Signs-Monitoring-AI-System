"""Loss functions and metrics for vital signs monitoring."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)
import logging

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class.
            gamma: Focusing parameter.
            reduction: Reduction method ('mean', 'sum', 'none').
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            inputs: Predicted logits.
            targets: Ground truth labels.
            
        Returns:
            Focal loss value.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    """Tversky Loss for segmentation tasks."""
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        """Initialize Tversky Loss.
        
        Args:
            alpha: Weight for false positives.
            beta: Weight for false negatives.
            smooth: Smoothing factor.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            inputs: Predicted probabilities.
            targets: Ground truth labels.
            
        Returns:
            Tversky loss value.
        """
        # Convert to probabilities if logits
        if inputs.dim() > 1 and inputs.size(1) > 1:
            inputs = F.softmax(inputs, dim=1)
            inputs = inputs[:, 1]  # Take positive class probability
        
        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()
        
        # Calculate Tversky index
        true_positives = (inputs * targets).sum()
        false_negatives = (targets * (1 - inputs)).sum()
        false_positives = ((1 - targets) * inputs).sum()
        
        tversky_index = (true_positives + self.smooth) / (
            true_positives + self.alpha * false_positives + self.beta * false_negatives + self.smooth
        )
        
        return 1 - tversky_index


class ReconstructionLoss(nn.Module):
    """Reconstruction loss for autoencoder models."""
    
    def __init__(self, loss_type: str = 'mse', reduction: str = 'mean'):
        """Initialize reconstruction loss.
        
        Args:
            loss_type: Type of loss ('mse', 'mae', 'huber').
            reduction: Reduction method.
        """
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            inputs: Reconstructed inputs.
            targets: Original inputs.
            
        Returns:
            Reconstruction loss value.
        """
        if self.loss_type == 'mse':
            loss = F.mse_loss(inputs, targets, reduction=self.reduction)
        elif self.loss_type == 'mae':
            loss = F.l1_loss(inputs, targets, reduction=self.reduction)
        elif self.loss_type == 'huber':
            loss = F.huber_loss(inputs, targets, reduction=self.reduction)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss


class VitalSignsMetrics:
    """Metrics calculator for vital signs monitoring."""
    
    def __init__(self, num_classes: int = 2):
        """Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes.
        """
        self.num_classes = num_classes
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def update(self, predictions: np.ndarray, targets: np.ndarray, 
               probabilities: Optional[np.ndarray] = None) -> None:
        """Update metrics with new predictions.
        
        Args:
            predictions: Predicted labels.
            targets: Ground truth labels.
            probabilities: Predicted probabilities.
        """
        self.predictions.extend(predictions.tolist())
        self.targets.extend(targets.tolist())
        
        if probabilities is not None:
            self.probabilities.extend(probabilities.tolist())
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics.
        
        Returns:
            Dictionary of metric values.
        """
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        if len(self.probabilities) > 0:
            probabilities = np.array(self.probabilities)
        else:
            probabilities = None
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = self._compute_accuracy(predictions, targets)
        metrics['precision'] = precision_score(targets, predictions, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(targets, predictions, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(targets, predictions, average='weighted', zero_division=0)
        
        # ROC AUC
        if probabilities is not None and self.num_classes == 2:
            try:
                metrics['roc_auc'] = roc_auc_score(targets, probabilities)
            except ValueError:
                metrics['roc_auc'] = 0.0
        
        # Precision-Recall AUC
        if probabilities is not None and self.num_classes == 2:
            try:
                precision, recall, _ = precision_recall_curve(targets, probabilities)
                metrics['pr_auc'] = auc(recall, precision)
            except ValueError:
                metrics['pr_auc'] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        return metrics
    
    def _compute_accuracy(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute accuracy."""
        return np.mean(predictions == targets)
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        return confusion_matrix(self.targets, self.predictions)
    
    def get_classification_report(self) -> str:
        """Get detailed classification report."""
        return classification_report(
            self.targets, 
            self.predictions,
            target_names=['Normal', 'Anomalous'] if self.num_classes == 2 else None
        )


class AnomalyDetectionMetrics:
    """Specialized metrics for anomaly detection."""
    
    def __init__(self):
        """Initialize anomaly detection metrics."""
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.anomaly_scores = []
        self.labels = []
        self.thresholds = []
    
    def update(self, anomaly_scores: np.ndarray, labels: np.ndarray) -> None:
        """Update metrics with new anomaly scores and labels.
        
        Args:
            anomaly_scores: Anomaly scores.
            labels: Ground truth labels (0=normal, 1=anomaly).
        """
        self.anomaly_scores.extend(anomaly_scores.tolist())
        self.labels.extend(labels.tolist())
    
    def compute_at_threshold(self, threshold: float) -> Dict[str, float]:
        """Compute metrics at a specific threshold.
        
        Args:
            threshold: Anomaly threshold.
            
        Returns:
            Dictionary of metric values.
        """
        scores = np.array(self.anomaly_scores)
        labels = np.array(self.labels)
        
        predictions = (scores >= threshold).astype(int)
        
        # Confusion matrix
        tn, fp, fn, tp = self._get_confusion_matrix(predictions, labels)
        
        metrics = {}
        metrics['threshold'] = threshold
        
        # Basic metrics
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # F1 score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1'] = 0.0
        
        # Youden's J statistic
        metrics['youden_j'] = metrics['sensitivity'] + metrics['specificity'] - 1
        
        return metrics
    
    def compute_optimal_threshold(self, metric: str = 'youden_j') -> Dict[str, float]:
        """Find optimal threshold based on a metric.
        
        Args:
            metric: Metric to optimize ('youden_j', 'f1', 'precision', 'recall').
            
        Returns:
            Dictionary with optimal threshold and metrics.
        """
        scores = np.array(self.anomaly_scores)
        labels = np.array(self.labels)
        
        # Get unique thresholds
        thresholds = np.unique(scores)
        
        best_metric = -1
        best_threshold = 0
        best_metrics = {}
        
        for threshold in thresholds:
            metrics = self.compute_at_threshold(threshold)
            
            if metrics[metric] > best_metric:
                best_metric = metrics[metric]
                best_threshold = threshold
                best_metrics = metrics
        
        return best_metrics
    
    def _get_confusion_matrix(self, predictions: np.ndarray, labels: np.ndarray) -> Tuple[int, int, int, int]:
        """Get confusion matrix components."""
        cm = confusion_matrix(labels, predictions)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle edge cases
            tn = fp = fn = tp = 0
            if len(np.unique(labels)) == 1:
                if labels[0] == 0:
                    tn = len(labels)
                else:
                    tp = len(labels)
            elif len(np.unique(predictions)) == 1:
                if predictions[0] == 0:
                    tn = np.sum(labels == 0)
                    fn = np.sum(labels == 1)
                else:
                    fp = np.sum(labels == 0)
                    tp = np.sum(labels == 1)
        
        return tn, fp, fn, tp


class CalibrationMetrics:
    """Metrics for model calibration."""
    
    def __init__(self, n_bins: int = 10):
        """Initialize calibration metrics.
        
        Args:
            n_bins: Number of bins for calibration.
        """
        self.n_bins = n_bins
    
    def compute_ece(self, probabilities: np.ndarray, labels: np.ndarray) -> float:
        """Compute Expected Calibration Error (ECE).
        
        Args:
            probabilities: Predicted probabilities.
            labels: Ground truth labels.
            
        Returns:
            ECE value.
        """
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def compute_brier_score(self, probabilities: np.ndarray, labels: np.ndarray) -> float:
        """Compute Brier score.
        
        Args:
            probabilities: Predicted probabilities.
            labels: Ground truth labels.
            
        Returns:
            Brier score.
        """
        return np.mean((probabilities - labels) ** 2)
    
    def compute_reliability_diagram(self, probabilities: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute reliability diagram data.
        
        Args:
            probabilities: Predicted probabilities.
            labels: Ground truth labels.
            
        Returns:
            Dictionary with bin centers, accuracies, and counts.
        """
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = (bin_lowers + bin_uppers) / 2
        accuracies = []
        counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.sum()
            counts.append(prop_in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                accuracies.append(accuracy_in_bin)
            else:
                accuracies.append(0)
        
        return {
            'bin_centers': bin_centers,
            'accuracies': np.array(accuracies),
            'counts': np.array(counts)
        }
