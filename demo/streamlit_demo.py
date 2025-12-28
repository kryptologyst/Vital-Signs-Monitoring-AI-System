"""Streamlit demo for vital signs monitoring system."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data import VitalSignsGenerator, VitalSignsProcessor
from src.models import create_model
from src.explainability import VitalSignsExplainer
from src.utils import get_device, VitalSignsThresholds
from src.losses import VitalSignsMetrics


# Page configuration
st.set_page_config(
    page_title="Vital Signs Monitoring AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .anomaly-alert {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer banner
st.markdown("""
<div class="disclaimer">
    <h4>⚠️ IMPORTANT DISCLAIMER</h4>
    <p><strong>This is a research demonstration tool and is NOT intended for clinical use.</strong></p>
    <p>This system is for educational and research purposes only. It should not be used for:</p>
    <ul>
        <li>Medical diagnosis or treatment decisions</li>
        <li>Replacing professional medical advice</li>
        <li>Clinical decision support without physician supervision</li>
    </ul>
    <p>Always consult with qualified healthcare professionals for medical concerns.</p>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🏥 Vital Signs Monitoring AI</h1>', unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'explainer' not in st.session_state:
    st.session_state.explainer = None
if 'generated_data' not in st.session_state:
    st.session_state.generated_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None


@st.cache_resource
def load_model(model_name: str = "cnn"):
    """Load a pre-trained model."""
    try:
        device = get_device()
        model = create_model(
            model_name=model_name,
            input_size=3,
            sequence_length=100,
            num_classes=2
        )
        
        # Load pre-trained weights if available
        model_path = Path("checkpoints") / f"{model_name}_model.pth"
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            st.success(f"Loaded pre-trained {model_name.upper()} model")
        else:
            st.warning(f"No pre-trained model found. Using randomly initialized {model_name.upper()} model.")
        
        model.eval()
        return model.to(device)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def generate_synthetic_data(
    n_samples: int,
    anomaly_types: List[str],
    sampling_rate: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic vital signs data."""
    generator = VitalSignsGenerator(seed=42)
    
    # Generate normal data
    normal_data = generator.generate_normal_vitals(
        n_samples=n_samples // 2,
        sampling_rate=sampling_rate
    )
    
    # Generate anomalous data
    anomalous_data = generator.generate_anomalous_vitals(
        n_samples=n_samples // 2,
        anomaly_types=anomaly_types,
        sampling_rate=sampling_rate
    )
    
    # Process data
    processor = VitalSignsProcessor(sampling_rate=sampling_rate)
    normal_data = processor.preprocess(normal_data)
    anomalous_data = processor.preprocess(anomalous_data)
    
    # Create sequences
    normal_sequences, normal_labels = processor.create_sequences(
        normal_data, sequence_length=100
    )
    anomalous_sequences, anomalous_labels = processor.create_sequences(
        anomalous_data, sequence_length=100
    )
    
    # Combine data
    all_sequences = np.vstack([normal_sequences, anomalous_sequences])
    all_labels = np.hstack([normal_labels, anomalous_labels])
    
    return all_sequences, all_labels


def create_vital_signs_plot(data: np.ndarray, title: str = "Vital Signs") -> go.Figure:
    """Create a plotly figure for vital signs data."""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=['Heart Rate', 'Temperature', 'SpO2'],
        vertical_spacing=0.1
    )
    
    time_steps = np.arange(data.shape[0])
    vital_sign_names = ['Heart Rate', 'Temperature', 'SpO2']
    colors = ['red', 'blue', 'green']
    
    for i, (name, color) in enumerate(zip(vital_sign_names, colors)):
        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=data[:, i],
                mode='lines',
                name=name,
                line=dict(color=color, width=2)
            ),
            row=i+1, col=1
        )
    
    fig.update_layout(
        title=title,
        height=600,
        showlegend=False
    )
    
    return fig


def main():
    """Main demo function."""
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "Select Model",
        ["cnn", "lstm", "transformer"],
        index=0
    )
    
    # Data generation parameters
    st.sidebar.subheader("Data Generation")
    n_samples = st.sidebar.slider("Number of Samples", 100, 1000, 500)
    
    anomaly_types = st.sidebar.multiselect(
        "Anomaly Types",
        ["tachycardia", "bradycardia", "hyperthermia", "hypothermia", "hypoxemia"],
        default=["tachycardia", "hypothermia", "hypoxemia"]
    )
    
    sampling_rate = st.sidebar.slider("Sampling Rate (Hz)", 0.1, 5.0, 1.0)
    
    # Load model
    if st.sidebar.button("Load Model"):
        with st.spinner("Loading model..."):
            st.session_state.model = load_model(model_name)
            if st.session_state.model is not None:
                st.session_state.explainer = VitalSignsExplainer(
                    st.session_state.model, model_name
                )
    
    # Generate data
    if st.sidebar.button("Generate Data"):
        with st.spinner("Generating synthetic data..."):
            sequences, labels = generate_synthetic_data(
                n_samples, anomaly_types, sampling_rate
            )
            st.session_state.generated_data = (sequences, labels)
            st.success(f"Generated {len(sequences)} samples")
    
    # Main content
    if st.session_state.model is None:
        st.info("Please load a model from the sidebar to begin.")
        return
    
    if st.session_state.generated_data is None:
        st.info("Please generate data from the sidebar to begin.")
        return
    
    sequences, labels = st.session_state.generated_data
    
    # Data overview
    st.subheader("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(sequences))
    
    with col2:
        st.metric("Normal Samples", np.sum(labels == 0))
    
    with col3:
        st.metric("Anomalous Samples", np.sum(labels == 1))
    
    with col4:
        anomaly_rate = np.sum(labels == 1) / len(labels) * 100
        st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
    
    # Sample selection
    st.subheader("🔍 Sample Analysis")
    
    sample_idx = st.selectbox(
        "Select Sample to Analyze",
        range(len(sequences)),
        format_func=lambda x: f"Sample {x} ({'Anomalous' if labels[x] == 1 else 'Normal'})"
    )
    
    selected_sample = sequences[sample_idx:sample_idx+1]
    true_label = labels[sample_idx]
    
    # Display selected sample
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = create_vital_signs_plot(
            selected_sample[0],
            f"Sample {sample_idx} - {'Anomalous' if true_label == 1 else 'Normal'}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Vital Signs Summary")
        
        # Calculate statistics
        hr_mean = np.mean(selected_sample[0, :, 0])
        temp_mean = np.mean(selected_sample[0, :, 1])
        spo2_mean = np.mean(selected_sample[0, :, 2])
        
        st.metric("Heart Rate (avg)", f"{hr_mean:.1f}")
        st.metric("Temperature (avg)", f"{temp_mean:.1f}")
        st.metric("SpO2 (avg)", f"{spo2_mean:.1f}")
        
        # Check thresholds
        thresholds = VitalSignsThresholds()
        
        hr_normal = thresholds.is_normal("heart_rate", hr_mean)
        temp_normal = thresholds.is_normal("temperature", temp_mean)
        spo2_normal = thresholds.is_normal("spo2", spo2_mean)
        
        if not all([hr_normal, temp_normal, spo2_normal]):
            st.markdown('<div class="anomaly-alert">⚠️ Some vital signs are outside normal ranges</div>', 
                       unsafe_allow_html=True)
    
    # Model prediction
    st.subheader("🤖 AI Prediction")
    
    if st.button("Analyze Sample"):
        with st.spinner("Analyzing sample..."):
            device = get_device()
            sample_tensor = torch.FloatTensor(selected_sample).to(device)
            
            # Get prediction
            with torch.no_grad():
                output = st.session_state.model(sample_tensor)
                probabilities = torch.softmax(output, dim=1)
                prediction = torch.argmax(output, dim=1).item()
                confidence = probabilities[0, prediction].item()
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Prediction", "Anomalous" if prediction == 1 else "Normal")
            
            with col2:
                st.metric("Confidence", f"{confidence:.3f}")
            
            with col3:
                correct = "✅" if prediction == true_label else "❌"
                st.metric("Correct", correct)
            
            # Probability distribution
            fig = go.Figure(data=[
                go.Bar(
                    x=['Normal', 'Anomalous'],
                    y=probabilities[0].cpu().numpy(),
                    marker_color=['green', 'red']
                )
            ])
            fig.update_layout(
                title="Prediction Probabilities",
                yaxis_title="Probability",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Uncertainty quantification
            st.subheader("🎯 Uncertainty Analysis")
            
            mean_pred, uncertainty = st.session_state.explainer.uncertainty_quantifier.monte_carlo_dropout(
                sample_tensor, num_samples=50
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Prediction Uncertainty", f"{uncertainty[0, prediction]:.3f}")
            
            with col2:
                st.metric("Calibrated Confidence", f"{mean_pred[0, prediction]:.3f}")
            
            # Uncertainty visualization
            fig = go.Figure(data=[
                go.Bar(
                    x=['Normal', 'Anomalous'],
                    y=uncertainty[0],
                    marker_color=['orange', 'purple'],
                    name='Uncertainty'
                )
            ])
            fig.update_layout(
                title="Prediction Uncertainty",
                yaxis_title="Uncertainty",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Batch evaluation
    st.subheader("📈 Batch Evaluation")
    
    if st.button("Evaluate All Samples"):
        with st.spinner("Evaluating all samples..."):
            device = get_device()
            all_predictions = []
            all_probabilities = []
            
            batch_size = 32
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch).to(device)
                
                with torch.no_grad():
                    outputs = st.session_state.model(batch_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predictions = torch.argmax(outputs, dim=1)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_probabilities.extend(probabilities[:, 1].cpu().numpy())
            
            all_predictions = np.array(all_predictions)
            all_probabilities = np.array(all_probabilities)
            
            # Compute metrics
            metrics = VitalSignsMetrics()
            metrics.update(all_predictions, labels, all_probabilities)
            results = metrics.compute()
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{results['accuracy']:.3f}")
            
            with col2:
                st.metric("Precision", f"{results['precision']:.3f}")
            
            with col3:
                st.metric("Recall", f"{results['recall']:.3f}")
            
            with col4:
                st.metric("F1 Score", f"{results['f1']:.3f}")
            
            # ROC curve
            if 'roc_auc' in results:
                st.metric("ROC-AUC", f"{results['roc_auc']:.3f}")
            
            # Confusion matrix
            cm = metrics.get_confusion_matrix()
            
            fig = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title="Confusion Matrix",
                labels=dict(x="Predicted", y="Actual"),
                color_continuous_scale="Blues"
            )
            fig.update_xaxes(tickmode='array', tickvals=[0, 1], ticktext=['Normal', 'Anomalous'])
            fig.update_yaxes(tickmode='array', tickvals=[0, 1], ticktext=['Normal', 'Anomalous'])
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
