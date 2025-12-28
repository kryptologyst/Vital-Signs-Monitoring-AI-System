"""Tests for vital signs monitoring system."""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data import VitalSignsGenerator, VitalSignsProcessor, VitalSignsData
from src.models import create_model, VitalSignsCNN, VitalSignsLSTM, VitalSignsTransformer
from src.losses import FocalLoss, VitalSignsMetrics, AnomalyDetectionMetrics
from src.utils import VitalSignsThresholds, get_device, set_seed


class TestVitalSignsGenerator:
    """Test vital signs data generation."""
    
    def test_generate_normal_vitals(self):
        """Test normal vital signs generation."""
        generator = VitalSignsGenerator(seed=42)
        data = generator.generate_normal_vitals(n_samples=100)
        
        assert isinstance(data, VitalSignsData)
        assert len(data.heart_rate) == 100
        assert len(data.temperature) == 100
        assert len(data.spo2) == 100
        
        # Check that values are within reasonable ranges
        assert np.all(data.heart_rate >= 50)
        assert np.all(data.heart_rate <= 120)
        assert np.all(data.temperature >= 35.0)
        assert np.all(data.temperature <= 38.0)
        assert np.all(data.spo2 >= 90)
        assert np.all(data.spo2 <= 100)
    
    def test_generate_anomalous_vitals(self):
        """Test anomalous vital signs generation."""
        generator = VitalSignsGenerator(seed=42)
        data = generator.generate_anomalous_vitals(
            n_samples=100,
            anomaly_types=['tachycardia', 'hypothermia']
        )
        
        assert isinstance(data, VitalSignsData)
        assert len(data.heart_rate) == 100
        assert len(data.temperature) == 100
        assert len(data.spo2) == 100
    
    def test_reproducibility(self):
        """Test that generation is reproducible with same seed."""
        generator1 = VitalSignsGenerator(seed=42)
        generator2 = VitalSignsGenerator(seed=42)
        
        data1 = generator1.generate_normal_vitals(n_samples=50)
        data2 = generator2.generate_normal_vitals(n_samples=50)
        
        np.testing.assert_array_equal(data1.heart_rate, data2.heart_rate)
        np.testing.assert_array_equal(data1.temperature, data2.temperature)
        np.testing.assert_array_equal(data1.spo2, data2.spo2)


class TestVitalSignsProcessor:
    """Test vital signs data processing."""
    
    def test_preprocess(self):
        """Test data preprocessing."""
        generator = VitalSignsGenerator(seed=42)
        data = generator.generate_normal_vitals(n_samples=100)
        
        processor = VitalSignsProcessor()
        processed_data = processor.preprocess(data)
        
        assert isinstance(processed_data, VitalSignsData)
        assert len(processed_data.heart_rate) == 100
        assert len(processed_data.temperature) == 100
        assert len(processed_data.spo2) == 100
    
    def test_create_sequences(self):
        """Test sequence creation."""
        generator = VitalSignsGenerator(seed=42)
        data = generator.generate_normal_vitals(n_samples=200)
        
        processor = VitalSignsProcessor()
        sequences, labels = processor.create_sequences(data, sequence_length=50)
        
        assert sequences.shape[1] == 50  # sequence length
        assert sequences.shape[2] == 3   # number of vital signs
        assert len(labels) == len(sequences)
        assert all(label in [0, 1] for label in labels)


class TestModels:
    """Test neural network models."""
    
    def test_cnn_model(self):
        """Test CNN model creation and forward pass."""
        model = VitalSignsCNN(
            input_channels=3,
            sequence_length=100,
            num_classes=2
        )
        
        # Test forward pass
        x = torch.randn(2, 100, 3)
        output = model(x)
        
        assert output.shape == (2, 2)
        assert not torch.isnan(output).any()
    
    def test_lstm_model(self):
        """Test LSTM model creation and forward pass."""
        model = VitalSignsLSTM(
            input_size=3,
            hidden_size=64,
            num_classes=2
        )
        
        # Test forward pass
        x = torch.randn(2, 100, 3)
        output = model(x)
        
        assert output.shape == (2, 2)
        assert not torch.isnan(output).any()
    
    def test_transformer_model(self):
        """Test Transformer model creation and forward pass."""
        model = VitalSignsTransformer(
            input_size=3,
            d_model=64,
            num_classes=2
        )
        
        # Test forward pass
        x = torch.randn(2, 100, 3)
        output = model(x)
        
        assert output.shape == (2, 2)
        assert not torch.isnan(output).any()
    
    def test_create_model_function(self):
        """Test model creation function."""
        models = ['cnn', 'lstm', 'transformer']
        
        for model_name in models:
            model = create_model(
                model_name=model_name,
                input_size=3,
                sequence_length=100,
                num_classes=2
            )
            
            assert isinstance(model, torch.nn.Module)
            
            # Test forward pass
            x = torch.randn(2, 100, 3)
            output = model(x)
            assert output.shape == (2, 2)


class TestLosses:
    """Test loss functions."""
    
    def test_focal_loss(self):
        """Test Focal Loss."""
        focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
        
        inputs = torch.randn(4, 2)
        targets = torch.randint(0, 2, (4,))
        
        loss = focal_loss(inputs, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_vital_signs_metrics(self):
        """Test vital signs metrics."""
        metrics = VitalSignsMetrics(num_classes=2)
        
        predictions = np.array([0, 1, 0, 1])
        targets = np.array([0, 1, 1, 1])
        probabilities = np.array([0.1, 0.9, 0.3, 0.8])
        
        metrics.update(predictions, targets, probabilities)
        results = metrics.compute()
        
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1' in results
        assert 0 <= results['accuracy'] <= 1
    
    def test_anomaly_detection_metrics(self):
        """Test anomaly detection metrics."""
        metrics = AnomalyDetectionMetrics()
        
        anomaly_scores = np.array([0.1, 0.3, 0.7, 0.9])
        labels = np.array([0, 0, 1, 1])
        
        metrics.update(anomaly_scores, labels)
        optimal_metrics = metrics.compute_optimal_threshold()
        
        assert 'threshold' in optimal_metrics
        assert 'accuracy' in optimal_metrics
        assert 'precision' in optimal_metrics
        assert 'recall' in optimal_metrics


class TestUtils:
    """Test utility functions."""
    
    def test_vital_signs_thresholds(self):
        """Test vital signs thresholds."""
        thresholds = VitalSignsThresholds()
        
        # Test normal ranges
        assert thresholds.is_normal("heart_rate", 70)
        assert thresholds.is_normal("temperature", 37.0)
        assert thresholds.is_normal("spo2", 98)
        
        # Test abnormal ranges
        assert not thresholds.is_normal("heart_rate", 150)
        assert not thresholds.is_normal("temperature", 40.0)
        assert not thresholds.is_normal("spo2", 85)
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test that random numbers are reproducible
        np.random.seed(42)
        val1 = np.random.random()
        
        set_seed(42)
        np.random.seed(42)
        val2 = np.random.random()
        
        assert val1 == val2


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from data generation to prediction."""
        # Generate data
        generator = VitalSignsGenerator(seed=42)
        normal_data = generator.generate_normal_vitals(n_samples=100)
        anomalous_data = generator.generate_anomalous_vitals(
            n_samples=50,
            anomaly_types=['tachycardia']
        )
        
        # Process data
        processor = VitalSignsProcessor()
        normal_data = processor.preprocess(normal_data)
        anomalous_data = processor.preprocess(anomalous_data)
        
        # Create sequences
        normal_sequences, normal_labels = processor.create_sequences(
            normal_data, sequence_length=50
        )
        anomalous_sequences, anomalous_labels = processor.create_sequences(
            anomalous_data, sequence_length=50
        )
        
        # Combine data
        all_sequences = np.vstack([normal_sequences, anomalous_sequences])
        all_labels = np.hstack([normal_labels, anomalous_labels])
        
        # Create model
        model = create_model(
            model_name='cnn',
            input_size=3,
            sequence_length=50,
            num_classes=2
        )
        
        # Test prediction
        sample = torch.FloatTensor(all_sequences[:1])
        with torch.no_grad():
            output = model(sample)
            prediction = torch.argmax(output, dim=1).item()
        
        assert prediction in [0, 1]
    
    def test_metrics_integration(self):
        """Test metrics integration with model predictions."""
        # Generate test data
        generator = VitalSignsGenerator(seed=42)
        data = generator.generate_normal_vitals(n_samples=100)
        
        processor = VitalSignsProcessor()
        data = processor.preprocess(data)
        sequences, labels = processor.create_sequences(data, sequence_length=50)
        
        # Create model and get predictions
        model = create_model(
            model_name='cnn',
            input_size=3,
            sequence_length=50,
            num_classes=2
        )
        
        predictions = []
        probabilities = []
        
        for i in range(0, len(sequences), 10):
            batch = sequences[i:i+10]
            batch_tensor = torch.FloatTensor(batch)
            
            with torch.no_grad():
                output = model(batch_tensor)
                probs = torch.softmax(output, dim=1)
                preds = torch.argmax(output, dim=1)
                
                predictions.extend(preds.numpy())
                probabilities.extend(probs[:, 1].numpy())
        
        # Test metrics
        metrics = VitalSignsMetrics()
        metrics.update(np.array(predictions), labels[:len(predictions)], np.array(probabilities))
        results = metrics.compute()
        
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1' in results


if __name__ == "__main__":
    pytest.main([__file__])
