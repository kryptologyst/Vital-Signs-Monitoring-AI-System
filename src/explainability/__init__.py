"""Explainability and uncertainty quantification for vital signs monitoring."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class GradientCAM:
    """Gradient-weighted Class Activation Mapping for CNN models."""
    
    def __init__(self, model: nn.Module, target_layer: str):
        """Initialize GradientCAM.
        
        Args:
            model: Trained model.
            target_layer: Name of the target layer for activation maps.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Find the target layer
        target_module = None
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                target_module = module
                break
        
        if target_module is None:
            raise ValueError(f"Target layer '{self.target_layer}' not found")
        
        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """Generate Class Activation Map.
        
        Args:
            input_tensor: Input tensor.
            class_idx: Class index for which to generate CAM.
            
        Returns:
            Class activation map.
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        # Generate CAM
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Normalize CAM
        cam = np.maximum(cam, 0)
        cam = cam / cam.max() if cam.max() > 0 else cam
        
        return cam


class AttentionVisualizer:
    """Visualize attention weights for Transformer models."""
    
    def __init__(self, model: nn.Module):
        """Initialize attention visualizer.
        
        Args:
            model: Trained Transformer model.
        """
        self.model = model
        self.attention_weights = None
        
        # Register hook to capture attention weights
        self._register_hook()
    
    def _register_hook(self) -> None:
        """Register hook to capture attention weights."""
        def attention_hook(module, input, output):
            # Extract attention weights from the output
            if hasattr(module, 'self_attn'):
                self.attention_weights = module.self_attn.attn_weight
        
        # Register hook on transformer layers
        for name, module in self.model.named_modules():
            if 'transformer' in name and 'layers' in name:
                module.register_forward_hook(attention_hook)
    
    def get_attention_weights(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Get attention weights for input.
        
        Args:
            input_tensor: Input tensor.
            
        Returns:
            Attention weights.
        """
        self.model.eval()
        
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        if self.attention_weights is not None:
            return self.attention_weights.cpu().numpy()
        else:
            return None


class UncertaintyQuantifier:
    """Quantify model uncertainty using various methods."""
    
    def __init__(self, model: nn.Module):
        """Initialize uncertainty quantifier.
        
        Args:
            model: Trained model.
        """
        self.model = model
    
    def monte_carlo_dropout(
        self, 
        input_tensor: torch.Tensor, 
        num_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate uncertainty using Monte Carlo Dropout.
        
        Args:
            input_tensor: Input tensor.
            num_samples: Number of Monte Carlo samples.
            
        Returns:
            Tuple of (mean_predictions, uncertainty_estimates).
        """
        self.model.train()  # Enable dropout
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                output = self.model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                predictions.append(probabilities.cpu().numpy())
        
        predictions = np.array(predictions)
        
        # Compute mean and uncertainty
        mean_predictions = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        return mean_predictions, uncertainty
    
    def ensemble_uncertainty(
        self, 
        models: List[nn.Module], 
        input_tensor: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate uncertainty using model ensemble.
        
        Args:
            models: List of trained models.
            input_tensor: Input tensor.
            
        Returns:
            Tuple of (mean_predictions, uncertainty_estimates).
        """
        predictions = []
        
        for model in models:
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                predictions.append(probabilities.cpu().numpy())
        
        predictions = np.array(predictions)
        
        # Compute mean and uncertainty
        mean_predictions = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        return mean_predictions, uncertainty
    
    def temperature_scaling(
        self, 
        input_tensor: torch.Tensor, 
        temperature: float = 1.0
    ) -> np.ndarray:
        """Apply temperature scaling for calibration.
        
        Args:
            input_tensor: Input tensor.
            temperature: Temperature parameter.
            
        Returns:
            Calibrated probabilities.
        """
        self.model.eval()
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            scaled_logits = logits / temperature
            probabilities = F.softmax(scaled_logits, dim=1)
        
        return probabilities.cpu().numpy()


class VitalSignsExplainer:
    """Main explainability class for vital signs monitoring."""
    
    def __init__(self, model: nn.Module, model_type: str = 'cnn'):
        """Initialize explainer.
        
        Args:
            model: Trained model.
            model_type: Type of model ('cnn', 'lstm', 'transformer').
        """
        self.model = model
        self.model_type = model_type.lower()
        
        # Initialize appropriate explainability methods
        if self.model_type == 'cnn':
            self.gradcam = GradientCAM(model, target_layer='conv3')
        elif self.model_type == 'transformer':
            self.attention_viz = AttentionVisualizer(model)
        
        self.uncertainty_quantifier = UncertaintyQuantifier(model)
    
    def explain_prediction(
        self, 
        input_tensor: torch.Tensor,
        vital_sign_names: List[str] = None,
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Generate explanations for a prediction.
        
        Args:
            input_tensor: Input tensor.
            vital_sign_names: Names of vital signs.
            save_path: Path to save visualization.
            
        Returns:
            Dictionary of explanations.
        """
        if vital_sign_names is None:
            vital_sign_names = ['Heart Rate', 'Temperature', 'SpO2']
        
        explanations = {}
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            prediction = torch.argmax(output, dim=1).item()
            confidence = probabilities[0, prediction].item()
        
        explanations['prediction'] = prediction
        explanations['confidence'] = confidence
        explanations['probabilities'] = probabilities[0].cpu().numpy()
        
        # Generate model-specific explanations
        if self.model_type == 'cnn':
            cam = self.gradcam.generate_cam(input_tensor, prediction)
            explanations['gradcam'] = cam
            
        elif self.model_type == 'transformer':
            attention_weights = self.attention_viz.get_attention_weights(input_tensor)
            explanations['attention_weights'] = attention_weights
        
        # Uncertainty quantification
        mean_pred, uncertainty = self.uncertainty_quantifier.monte_carlo_dropout(
            input_tensor, num_samples=50
        )
        explanations['uncertainty'] = uncertainty[0]
        explanations['calibrated_probabilities'] = mean_pred[0]
        
        # Create visualization
        if save_path:
            self._create_explanation_plot(
                input_tensor, explanations, vital_sign_names, save_path
            )
        
        return explanations
    
    def _create_explanation_plot(
        self, 
        input_tensor: torch.Tensor,
        explanations: Dict[str, Any],
        vital_sign_names: List[str],
        save_path: Path
    ) -> None:
        """Create visualization of explanations.
        
        Args:
            input_tensor: Input tensor.
            explanations: Explanation dictionary.
            vital_sign_names: Names of vital signs.
            save_path: Path to save plot.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Vital Signs Monitoring - Prediction Explanation', fontsize=16)
        
        # Plot 1: Input vital signs
        ax1 = axes[0, 0]
        data = input_tensor[0].cpu().numpy()
        for i, name in enumerate(vital_sign_names):
            ax1.plot(data[:, i], label=name, alpha=0.8)
        ax1.set_title('Input Vital Signs')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Normalized Values')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Prediction probabilities
        ax2 = axes[0, 1]
        classes = ['Normal', 'Anomalous']
        probs = explanations['probabilities']
        bars = ax2.bar(classes, probs, color=['green', 'red'], alpha=0.7)
        ax2.set_title(f'Prediction Probabilities\nConfidence: {explanations["confidence"]:.3f}')
        ax2.set_ylabel('Probability')
        
        # Add probability values on bars
        for bar, prob in zip(bars, probs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom')
        
        # Plot 3: Uncertainty
        ax3 = axes[1, 0]
        uncertainty = explanations['uncertainty']
        ax3.bar(classes, uncertainty, color=['orange', 'purple'], alpha=0.7)
        ax3.set_title('Prediction Uncertainty')
        ax3.set_ylabel('Uncertainty')
        
        # Plot 4: Model-specific explanation
        ax4 = axes[1, 1]
        
        if self.model_type == 'cnn' and 'gradcam' in explanations:
            cam = explanations['gradcam']
            im = ax4.imshow(cam, cmap='hot', aspect='auto')
            ax4.set_title('Gradient CAM')
            ax4.set_xlabel('Time Steps')
            ax4.set_ylabel('Feature Maps')
            plt.colorbar(im, ax=ax4)
            
        elif self.model_type == 'transformer' and 'attention_weights' in explanations:
            attention = explanations['attention_weights']
            if attention is not None:
                im = ax4.imshow(attention[0], cmap='Blues', aspect='auto')
                ax4.set_title('Attention Weights')
                ax4.set_xlabel('Time Steps')
                ax4.set_ylabel('Time Steps')
                plt.colorbar(im, ax=ax4)
            else:
                ax4.text(0.5, 0.5, 'No attention weights available', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Attention Weights')
        else:
            ax4.text(0.5, 0.5, 'No specific explanation available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Model Explanation')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def batch_explain(
        self, 
        data_loader: torch.utils.data.DataLoader,
        output_dir: Path,
        num_samples: int = 10
    ) -> None:
        """Generate explanations for a batch of samples.
        
        Args:
            data_loader: Data loader.
            output_dir: Output directory.
            num_samples: Number of samples to explain.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sample_count = 0
        for batch_idx, (data, targets) in enumerate(data_loader):
            if sample_count >= num_samples:
                break
            
            for i in range(data.size(0)):
                if sample_count >= num_samples:
                    break
                
                # Get single sample
                sample = data[i:i+1]
                target = targets[i].item()
                
                # Generate explanation
                explanation = self.explain_prediction(sample)
                
                # Save explanation
                save_path = output_dir / f'explanation_{sample_count:03d}.png'
                self._create_explanation_plot(
                    sample, explanation, 
                    ['Heart Rate', 'Temperature', 'SpO2'], 
                    save_path
                )
                
                sample_count += 1
        
        logger.info(f"Generated {sample_count} explanations in {output_dir}")


class FeatureImportanceAnalyzer:
    """Analyze feature importance for vital signs."""
    
    def __init__(self, model: nn.Module):
        """Initialize feature importance analyzer.
        
        Args:
            model: Trained model.
        """
        self.model = model
    
    def compute_feature_importance(
        self, 
        input_tensor: torch.Tensor,
        vital_sign_names: List[str] = None
    ) -> Dict[str, float]:
        """Compute feature importance using gradient-based methods.
        
        Args:
            input_tensor: Input tensor.
            vital_sign_names: Names of vital signs.
            
        Returns:
            Dictionary of feature importance scores.
        """
        if vital_sign_names is None:
            vital_sign_names = ['Heart Rate', 'Temperature', 'SpO2']
        
        self.model.eval()
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        prediction = torch.argmax(output, dim=1)
        
        # Backward pass
        self.model.zero_grad()
        output[0, prediction].backward()
        
        # Compute gradients
        gradients = input_tensor.grad[0].cpu().numpy()
        
        # Compute importance scores
        importance_scores = {}
        for i, name in enumerate(vital_sign_names):
            # Use mean absolute gradient as importance
            importance_scores[name] = np.mean(np.abs(gradients[:, i]))
        
        return importance_scores
    
    def compute_permutation_importance(
        self, 
        data_loader: torch.utils.data.DataLoader,
        vital_sign_names: List[str] = None,
        num_permutations: int = 10
    ) -> Dict[str, float]:
        """Compute permutation importance.
        
        Args:
            data_loader: Data loader.
            vital_sign_names: Names of vital signs.
            num_permutations: Number of permutations.
            
        Returns:
            Dictionary of permutation importance scores.
        """
        if vital_sign_names is None:
            vital_sign_names = ['Heart Rate', 'Temperature', 'SpO2']
        
        # Compute baseline accuracy
        baseline_accuracy = self._compute_accuracy(data_loader)
        
        importance_scores = {}
        
        for i, name in enumerate(vital_sign_names):
            permuted_accuracies = []
            
            for _ in range(num_permutations):
                # Create permuted data loader
                permuted_loader = self._permute_feature(data_loader, i)
                
                # Compute accuracy
                accuracy = self._compute_accuracy(permuted_loader)
                permuted_accuracies.append(accuracy)
            
            # Importance is the decrease in accuracy
            importance_scores[name] = baseline_accuracy - np.mean(permuted_accuracies)
        
        return importance_scores
    
    def _compute_accuracy(self, data_loader: torch.utils.data.DataLoader) -> float:
        """Compute accuracy on data loader."""
        correct = 0
        total = 0
        
        self.model.eval()
        with torch.no_grad():
            for data, targets in data_loader:
                outputs = self.model(data)
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == targets).sum().item()
                total += targets.size(0)
        
        return correct / total if total > 0 else 0.0
    
    def _permute_feature(
        self, 
        data_loader: torch.utils.data.DataLoader, 
        feature_idx: int
    ) -> torch.utils.data.DataLoader:
        """Create a data loader with permuted feature."""
        # This is a simplified implementation
        # In practice, you'd need to properly handle the permutation
        return data_loader
