"""Training and evaluation modules for vital signs monitoring."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from tqdm import tqdm
import logging
from pathlib import Path
import json
from datetime import datetime

from ..models import create_model
from ..losses import FocalLoss, TverskyLoss, ReconstructionLoss, VitalSignsMetrics, AnomalyDetectionMetrics, CalibrationMetrics
from ..data import VitalSignsGenerator, VitalSignsProcessor, VitalSignsData
from ..utils import get_device, setup_logging

logger = logging.getLogger(__name__)


class VitalSignsTrainer:
    """Trainer class for vital signs monitoring models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the trainer.
        
        Args:
            model: Model to train.
            device: Device to use for training.
            config: Training configuration.
            logger: Logger instance.
        """
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize loss function
        self.criterion = self._create_loss_function()
        
        # Initialize metrics
        self.metrics = VitalSignsMetrics(num_classes=config.get('num_classes', 2))
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_metrics': [],
            'val_loss': [],
            'val_metrics': []
        }
        
        # Best model tracking
        self.best_val_score = -np.inf
        self.best_model_state = None
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        learning_rate = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on config."""
        loss_name = self.config.get('loss', 'cross_entropy').lower()
        
        if loss_name == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif loss_name == 'focal':
            alpha = self.config.get('focal_alpha', 1.0)
            gamma = self.config.get('focal_gamma', 2.0)
            return FocalLoss(alpha=alpha, gamma=gamma)
        elif loss_name == 'tversky':
            alpha = self.config.get('tversky_alpha', 0.3)
            beta = self.config.get('tversky_beta', 0.7)
            return TverskyLoss(alpha=alpha, beta=beta)
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader.
            
        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        total_loss = 0.0
        self.metrics.reset()
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            
            # Compute loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            targets_cpu = targets.cpu().numpy()
            
            self.metrics.update(predictions, targets_cpu, probabilities)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute epoch metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_metrics = self.metrics.compute()
        
        return {'loss': epoch_loss, **epoch_metrics}
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch.
        
        Args:
            val_loader: Validation data loader.
            
        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        self.metrics.reset()
        
        with torch.no_grad():
            for data, targets in tqdm(val_loader, desc="Validation", leave=False):
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                
                # Update metrics
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                targets_cpu = targets.cpu().numpy()
                
                self.metrics.update(predictions, targets_cpu, probabilities)
        
        # Compute epoch metrics
        epoch_loss = total_loss / len(val_loader)
        epoch_metrics = self.metrics.compute()
        
        return {'loss': epoch_loss, **epoch_metrics}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_dir: Optional[Path] = None,
        early_stopping_patience: int = 10
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            num_epochs: Number of epochs to train.
            save_dir: Directory to save checkpoints.
            early_stopping_patience: Patience for early stopping.
            
        Returns:
            Training history.
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        best_val_score = -np.inf
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_metrics'].append(train_metrics)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_metrics'].append(val_metrics)
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val F1: {val_metrics['f1']:.4f}, "
                f"Val ROC-AUC: {val_metrics.get('roc_auc', 0):.4f}"
            )
            
            # Check for best model
            val_score = val_metrics.get('f1', val_metrics.get('roc_auc', 0))
            if val_score > best_val_score:
                best_val_score = val_score
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                
                # Save best model
                if save_dir:
                    self.save_checkpoint(save_dir / 'best_model.pth', epoch, val_score)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return self.history
    
    def save_checkpoint(self, path: Path, epoch: int, score: float) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint.
            score: Validation score.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'score': score,
            'config': self.config,
            'history': self.history
        }
        
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        
        self.logger.info(f"Checkpoint loaded from {path}")


class VitalSignsEvaluator:
    """Evaluator class for vital signs monitoring models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the evaluator.
        
        Args:
            model: Model to evaluate.
            device: Device to use for evaluation.
            logger: Logger instance.
        """
        self.model = model.to(device)
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize metrics
        self.metrics = VitalSignsMetrics()
        self.anomaly_metrics = AnomalyDetectionMetrics()
        self.calibration_metrics = CalibrationMetrics()
    
    def evaluate(
        self,
        test_loader: DataLoader,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader.
            return_predictions: Whether to return predictions.
            
        Returns:
            Dictionary of evaluation results.
        """
        self.model.eval()
        self.metrics.reset()
        self.anomaly_metrics.reset()
        
        all_predictions = []
        all_probabilities = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in tqdm(test_loader, desc="Evaluating"):
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_targets = np.array(all_targets)
        
        # Compute metrics
        self.metrics.update(all_predictions, all_targets, all_probabilities)
        metrics = self.metrics.compute()
        
        # Compute anomaly detection metrics
        self.anomaly_metrics.update(all_probabilities, all_targets)
        optimal_metrics = self.anomaly_metrics.compute_optimal_threshold()
        
        # Compute calibration metrics
        ece = self.calibration_metrics.compute_ece(all_probabilities, all_targets)
        brier_score = self.calibration_metrics.compute_brier_score(all_probabilities, all_targets)
        
        # Compile results
        results = {
            'classification_metrics': metrics,
            'anomaly_detection_metrics': optimal_metrics,
            'calibration_metrics': {
                'ece': ece,
                'brier_score': brier_score
            },
            'confusion_matrix': self.metrics.get_confusion_matrix(),
            'classification_report': self.metrics.get_classification_report()
        }
        
        if return_predictions:
            results['predictions'] = all_predictions
            results['probabilities'] = all_probabilities
            results['targets'] = all_targets
        
        return results
    
    def evaluate_uncertainty(
        self,
        test_loader: DataLoader,
        num_samples: int = 10
    ) -> Dict[str, np.ndarray]:
        """Evaluate model uncertainty using Monte Carlo Dropout.
        
        Args:
            test_loader: Test data loader.
            num_samples: Number of Monte Carlo samples.
            
        Returns:
            Dictionary with uncertainty estimates.
        """
        self.model.train()  # Enable dropout
        
        all_predictions = []
        all_uncertainties = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in tqdm(test_loader, desc="Evaluating Uncertainty"):
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Monte Carlo samples
                mc_predictions = []
                for _ in range(num_samples):
                    outputs = self.model(data)
                    probabilities = torch.softmax(outputs, dim=1)
                    mc_predictions.append(probabilities.cpu().numpy())
                
                # Stack predictions
                mc_predictions = np.stack(mc_predictions, axis=0)
                
                # Compute mean and uncertainty
                mean_predictions = np.mean(mc_predictions, axis=0)
                uncertainty = np.std(mc_predictions, axis=0)
                
                # Store results
                all_predictions.extend(mean_predictions)
                all_uncertainties.extend(uncertainty)
                all_targets.extend(targets.cpu().numpy())
        
        return {
            'predictions': np.array(all_predictions),
            'uncertainties': np.array(all_uncertainties),
            'targets': np.array(all_targets)
        }


def create_data_loaders(
    sequences: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders.
    
    Args:
        sequences: Input sequences.
        labels: Target labels.
        batch_size: Batch size.
        train_ratio: Training set ratio.
        val_ratio: Validation set ratio.
        test_ratio: Test set ratio.
        random_seed: Random seed for splitting.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Set random seed
    np.random.seed(random_seed)
    
    # Shuffle indices
    n_samples = len(sequences)
    indices = np.random.permutation(n_samples)
    
    # Calculate split sizes
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(sequences[train_indices]),
        torch.LongTensor(labels[train_indices])
    )
    
    val_dataset = TensorDataset(
        torch.FloatTensor(sequences[val_indices]),
        torch.LongTensor(labels[val_indices])
    )
    
    test_dataset = TensorDataset(
        torch.FloatTensor(sequences[test_indices]),
        torch.LongTensor(labels[test_indices])
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def run_experiment(
    config: Dict[str, Any],
    output_dir: Path,
    random_seed: int = 42
) -> Dict[str, Any]:
    """Run a complete experiment.
    
    Args:
        config: Experiment configuration.
        output_dir: Output directory.
        random_seed: Random seed.
        
    Returns:
        Experiment results.
    """
    # Set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Setup logging
    logger = setup_logging(level=config.get('log_level', 'INFO'))
    logger.info("Starting experiment")
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Generate data
    logger.info("Generating synthetic data")
    generator = VitalSignsGenerator(seed=random_seed)
    
    # Generate normal and anomalous data
    normal_data = generator.generate_normal_vitals(
        n_samples=config['data']['n_normal_samples'],
        sampling_rate=config['data']['sampling_rate']
    )
    
    anomalous_data = generator.generate_anomalous_vitals(
        n_samples=config['data']['n_anomalous_samples'],
        anomaly_types=config['data']['anomaly_types'],
        sampling_rate=config['data']['sampling_rate']
    )
    
    # Process data
    processor = VitalSignsProcessor(sampling_rate=config['data']['sampling_rate'])
    normal_data = processor.preprocess(normal_data)
    anomalous_data = processor.preprocess(anomalous_data)
    
    # Create sequences
    normal_sequences, normal_labels = processor.create_sequences(
        normal_data, config['data']['sequence_length']
    )
    anomalous_sequences, anomalous_labels = processor.create_sequences(
        anomalous_data, config['data']['sequence_length']
    )
    
    # Combine data
    all_sequences = np.vstack([normal_sequences, anomalous_sequences])
    all_labels = np.hstack([normal_labels, anomalous_labels])
    
    logger.info(f"Created {len(all_sequences)} sequences with {all_labels.sum()} anomalies")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        all_sequences, all_labels,
        batch_size=config['training']['batch_size'],
        random_seed=random_seed
    )
    
    # Create model
    logger.info(f"Creating {config['model']['name']} model")
    model = create_model(
        model_name=config['model']['name'],
        input_size=all_sequences.shape[2],
        sequence_length=config['data']['sequence_length'],
        num_classes=2,
        **config['model'].get('params', {})
    )
    
    # Create trainer
    trainer = VitalSignsTrainer(model, device, config['training'], logger)
    
    # Train model
    logger.info("Starting training")
    history = trainer.train(
        train_loader, val_loader,
        num_epochs=config['training']['num_epochs'],
        save_dir=output_dir,
        early_stopping_patience=config['training'].get('early_stopping_patience', 10)
    )
    
    # Evaluate model
    logger.info("Evaluating model")
    evaluator = VitalSignsEvaluator(model, device, logger)
    results = evaluator.evaluate(test_loader, return_predictions=True)
    
    # Save results
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, dict):
                json_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        json_results[key][k] = v.tolist()
                    else:
                        json_results[key][k] = v
            else:
                json_results[key] = value
        
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    
    return results
