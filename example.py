#!/usr/bin/env python3
"""Simple example script demonstrating vital signs monitoring."""

import sys
from pathlib import Path
import numpy as np
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data import VitalSignsGenerator, VitalSignsProcessor
from src.models import create_model
from src.utils import VitalSignsThresholds, get_device


def main():
    """Main example function."""
    print("🏥 Vital Signs Monitoring AI - Example")
    print("=" * 50)
    
    # Disclaimer
    print("⚠️  DISCLAIMER: This is for research/education only, NOT for clinical use!")
    print()
    
    # Initialize components
    print("Initializing components...")
    generator = VitalSignsGenerator(seed=42)
    processor = VitalSignsProcessor(sampling_rate=1.0)
    thresholds = VitalSignsThresholds()
    device = get_device()
    
    print(f"Using device: {device}")
    print()
    
    # Generate synthetic data
    print("Generating synthetic vital signs data...")
    normal_data = generator.generate_normal_vitals(n_samples=200)
    anomalous_data = generator.generate_anomalous_vitals(
        n_samples=100,
        anomaly_types=['tachycardia', 'hypothermia']
    )
    
    print(f"Normal data: {normal_data.shape}")
    print(f"Anomalous data: {anomalous_data.shape}")
    print()
    
    # Process data
    print("Processing data...")
    normal_data = processor.preprocess(normal_data)
    anomalous_data = processor.preprocess(anomalous_data)
    
    # Create sequences
    sequence_length = 50
    normal_sequences, normal_labels = processor.create_sequences(
        normal_data, sequence_length
    )
    anomalous_sequences, anomalous_labels = processor.create_sequences(
        anomalous_data, sequence_length
    )
    
    print(f"Normal sequences: {normal_sequences.shape}")
    print(f"Anomalous sequences: {anomalous_sequences.shape}")
    print()
    
    # Combine data
    all_sequences = np.vstack([normal_sequences, anomalous_sequences])
    all_labels = np.hstack([normal_labels, anomalous_labels])
    
    print(f"Total sequences: {len(all_sequences)}")
    print(f"Anomaly rate: {np.sum(all_labels) / len(all_labels) * 100:.1f}%")
    print()
    
    # Create and test model
    print("Creating CNN model...")
    model = create_model(
        model_name='cnn',
        input_size=3,
        sequence_length=sequence_length,
        num_classes=2
    )
    
    model = model.to(device)
    model.eval()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Test predictions
    print("Making predictions on sample data...")
    sample_indices = [0, 50, 100, 150]  # Test different samples
    
    for idx in sample_indices:
        sample = all_sequences[idx:idx+1]
        true_label = all_labels[idx]
        
        # Convert to tensor
        sample_tensor = torch.FloatTensor(sample).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(sample_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(output, dim=1).item()
            confidence = probabilities[0, prediction].item()
        
        # Check vital signs against thresholds
        hr_mean = np.mean(sample[0, :, 0])
        temp_mean = np.mean(sample[0, :, 1])
        spo2_mean = np.mean(sample[0, :, 2])
        
        hr_normal = thresholds.is_normal("heart_rate", hr_mean)
        temp_normal = thresholds.is_normal("temperature", temp_mean)
        spo2_normal = thresholds.is_normal("spo2", spo2_mean)
        
        print(f"Sample {idx}:")
        print(f"  True label: {'Anomalous' if true_label == 1 else 'Normal'}")
        print(f"  Prediction: {'Anomalous' if prediction == 1 else 'Normal'}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Correct: {'✅' if prediction == true_label else '❌'}")
        print(f"  Vital signs: HR={hr_mean:.1f}, Temp={temp_mean:.1f}, SpO2={spo2_mean:.1f}")
        print(f"  Threshold check: HR={'✅' if hr_normal else '❌'}, "
              f"Temp={'✅' if temp_normal else '❌'}, "
              f"SpO2={'✅' if spo2_normal else '❌'}")
        print()
    
    # Overall performance
    print("Computing overall performance...")
    predictions = []
    probabilities = []
    
    batch_size = 32
    for i in range(0, len(all_sequences), batch_size):
        batch = all_sequences[i:i+batch_size]
        batch_tensor = torch.FloatTensor(batch).to(device)
        
        with torch.no_grad():
            output = model(batch_tensor)
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(output, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs[:, 1].cpu().numpy())
    
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    
    # Compute metrics
    accuracy = np.mean(predictions == all_labels)
    precision = np.sum((predictions == 1) & (all_labels == 1)) / np.sum(predictions == 1)
    recall = np.sum((predictions == 1) & (all_labels == 1)) / np.sum(all_labels == 1)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Overall Performance:")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1-Score: {f1:.3f}")
    print()
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, predictions)
    print(f"Confusion Matrix:")
    print(f"  Normal   Anomalous")
    print(f"Normal    {cm[0,0]:8d} {cm[0,1]:10d}")
    print(f"Anomalous {cm[1,0]:8d} {cm[1,1]:10d}")
    print()
    
    print("✅ Example completed successfully!")
    print()
    print("Next steps:")
    print("1. Run 'python scripts/train.py' to train a model")
    print("2. Run 'streamlit run demo/streamlit_demo.py' for interactive demo")
    print("3. Check the notebooks/ directory for more examples")


if __name__ == "__main__":
    main()
