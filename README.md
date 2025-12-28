# Vital Signs Monitoring AI System

A comprehensive AI-powered system for monitoring and analyzing vital signs data, designed for research and educational purposes.

## ⚠️ IMPORTANT DISCLAIMER

**This system is for research and educational purposes only and is NOT intended for clinical use.**

- This tool should not be used for medical diagnosis or treatment decisions
- It does not replace professional medical advice or clinical judgment
- Always consult with qualified healthcare professionals for medical concerns
- The system uses synthetic data and should not be used with real patient data without proper safeguards

## Project Overview

This project implements a modern, AI-powered vital signs monitoring system that can:

- **Detect anomalies** in vital signs data (heart rate, temperature, SpO2)
- **Provide explanations** for AI predictions using various interpretability methods
- **Quantify uncertainty** in predictions using Monte Carlo methods
- **Support multiple models** including CNN, LSTM, and Transformer architectures
- **Generate synthetic data** for research and testing purposes

## Quick Start

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.0 or higher
- CUDA-capable GPU (optional, for faster training)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Vital-Signs-Monitoring-AI-System.git
cd Vital-Signs-Monitoring-AI-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

### Basic Usage

1. **Train a model:**
```bash
python scripts/train.py --config configs/cnn.yaml
```

2. **Run the interactive demo:**
```bash
streamlit run demo/streamlit_demo.py
```

3. **Generate synthetic data:**
```python
from src.data import VitalSignsGenerator

generator = VitalSignsGenerator(seed=42)
normal_data = generator.generate_normal_vitals(n_samples=1000)
anomalous_data = generator.generate_anomalous_vitals(
    n_samples=300, 
    anomaly_types=['tachycardia', 'hypothermia']
)
```

## 📁 Project Structure

```
health-monitoring-ai/
├── src/                          # Source code
│   ├── models/                   # Neural network models
│   ├── data/                     # Data generation and processing
│   ├── losses/                   # Loss functions and metrics
│   ├── utils/                    # Utility functions
│   ├── train/                    # Training and evaluation
│   └── explainability/           # Explainability methods
├── configs/                      # Configuration files
├── scripts/                      # Training and evaluation scripts
├── demo/                         # Interactive demos
├── tests/                        # Unit tests
├── assets/                       # Generated assets and visualizations
├── checkpoints/                  # Model checkpoints
└── docs/                         # Documentation
```

## Models

The system supports multiple neural network architectures:

### 1D CNN
- Convolutional layers for temporal pattern recognition
- Batch normalization and dropout for regularization
- Suitable for detecting local patterns in vital signs

### LSTM
- Bidirectional LSTM for sequence modeling
- Captures long-term dependencies in time series
- Good for detecting gradual changes in vital signs

### Transformer
- Self-attention mechanism for global context
- Positional encoding for temporal information
- Excellent for complex pattern recognition

### Ensemble
- Combines multiple models for improved performance
- Weighted voting based on validation performance
- Provides robust predictions

## Data

### Synthetic Data Generation

The system generates realistic synthetic vital signs data:

- **Normal patterns**: Based on clinical ranges and physiological variations
- **Anomaly types**: Tachycardia, bradycardia, hyperthermia, hypothermia, hypoxemia
- **Noise modeling**: Realistic sensor noise and physiological drift
- **Temporal correlations**: Realistic time-dependent patterns

### Data Processing

- **Outlier removal**: IQR and Z-score based methods
- **Signal smoothing**: Moving average and Gaussian filtering
- **Normalization**: Z-score and min-max scaling
- **Sequence creation**: Sliding window approach for time series modeling

## Evaluation Metrics

### Classification Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC and PR-AUC
- Sensitivity, Specificity, PPV, NPV

### Anomaly Detection Metrics
- Optimal threshold detection using Youden's J statistic
- Episode-level sensitivity analysis
- Detection latency evaluation

### Calibration Metrics
- Expected Calibration Error (ECE)
- Brier Score
- Reliability diagrams

## Explainability

The system provides multiple explainability methods:

### Gradient-based Methods
- **Gradient CAM**: Visualizes important regions in CNN feature maps
- **Feature importance**: Gradient-based feature attribution

### Attention Visualization
- **Attention weights**: Shows which time steps the model focuses on
- **Attention maps**: Visual representation of attention patterns

### Uncertainty Quantification
- **Monte Carlo Dropout**: Estimates prediction uncertainty
- **Ensemble methods**: Model disagreement as uncertainty measure
- **Temperature scaling**: Improves prediction calibration

## Configuration

The system uses YAML configuration files for easy experimentation:

```yaml
# Model configuration
model:
  name: "cnn"
  params:
    dropout_rate: 0.2

# Data configuration
data:
  n_normal_samples: 1000
  n_anomalous_samples: 300
  sequence_length: 100
  anomaly_types: ["tachycardia", "hypothermia"]

# Training configuration
training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.001
  optimizer: "adam"
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## Interactive Demo

The Streamlit demo provides an interactive interface for:

- **Model selection**: Choose between different architectures
- **Data generation**: Generate synthetic vital signs data
- **Sample analysis**: Analyze individual samples with explanations
- **Batch evaluation**: Evaluate model performance on entire dataset
- **Uncertainty visualization**: View prediction uncertainty estimates

## 🔧 Development

### Code Quality

The project uses modern Python development practices:

- **Type hints**: Full type annotation coverage
- **Documentation**: Google-style docstrings
- **Formatting**: Black code formatting
- **Linting**: Ruff for code quality checks
- **Testing**: Pytest for unit testing

### Pre-commit Hooks

Install pre-commit hooks:

```bash
pre-commit install
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Research Applications

This system is designed for:

- **Educational purposes**: Learning about AI in healthcare
- **Research**: Developing new anomaly detection methods
- **Benchmarking**: Comparing different model architectures
- **Prototyping**: Testing new explainability methods

## Privacy and Security

- **No PHI/PII**: System uses only synthetic data
- **De-identification**: Built-in utilities for data anonymization
- **Secure defaults**: Conservative privacy settings
- **Audit logging**: Comprehensive logging for compliance

## Citation

If you use this system in your research, please cite:

```bibtex
@software{vital_signs_monitoring_ai,
  title={Vital Signs Monitoring AI System},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Vital-Signs-Monitoring-AI-System}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions, issues, or contributions:

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for general questions
- **Email**: Contact the research team for collaboration inquiries

## Future Work

Planned improvements include:

- **Real-time monitoring**: Integration with wearable devices
- **Federated learning**: Privacy-preserving distributed training
- **Active learning**: Intelligent sample selection for annotation
- **Multi-modal data**: Integration of additional sensor data
- **Clinical validation**: Collaboration with medical professionals

---

**Remember: This system is for research and educational purposes only. Always consult with qualified healthcare professionals for medical concerns.**
# Vital-Signs-Monitoring-AI-System
