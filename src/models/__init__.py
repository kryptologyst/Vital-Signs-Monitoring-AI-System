"""Neural network models for vital signs monitoring and anomaly detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class VitalSignsCNN(nn.Module):
    """1D CNN for vital signs time series classification."""
    
    def __init__(
        self,
        input_channels: int = 3,
        sequence_length: int = 100,
        num_classes: int = 2,
        dropout_rate: float = 0.2
    ):
        """Initialize the CNN model.
        
        Args:
            input_channels: Number of input channels (vital signs).
            sequence_length: Length of input sequences.
            num_classes: Number of output classes.
            dropout_rate: Dropout rate.
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        
        # Calculate flattened size
        with torch.no_grad():
            x = torch.randn(1, input_channels, sequence_length)
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.pool3(self.conv3(x))
            self.flattened_size = x.numel()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_channels).
            
        Returns:
            Output tensor of shape (batch_size, num_classes).
        """
        # Reshape to (batch_size, input_channels, sequence_length)
        x = x.transpose(1, 2)
        
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


class VitalSignsLSTM(nn.Module):
    """LSTM model for vital signs time series classification."""
    
    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout_rate: float = 0.2,
        bidirectional: bool = True
    ):
        """Initialize the LSTM model.
        
        Args:
            input_size: Number of input features (vital signs).
            hidden_size: Hidden state size.
            num_layers: Number of LSTM layers.
            num_classes: Number of output classes.
            dropout_rate: Dropout rate.
            bidirectional: Whether to use bidirectional LSTM.
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_size, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).
            
        Returns:
            Output tensor of shape (batch_size, num_classes).
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = F.relu(self.fc1(last_output))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


class VitalSignsTransformer(nn.Module):
    """Transformer model for vital signs time series classification."""
    
    def __init__(
        self,
        input_size: int = 3,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        num_classes: int = 2,
        dropout_rate: float = 0.1,
        max_seq_length: int = 1000
    ):
        """Initialize the Transformer model.
        
        Args:
            input_size: Number of input features (vital signs).
            d_model: Model dimension.
            nhead: Number of attention heads.
            num_layers: Number of transformer layers.
            num_classes: Number of output classes.
            dropout_rate: Dropout rate.
            max_seq_length: Maximum sequence length.
        """
        super().__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(max_seq_length, d_model) * 0.1
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).
            
        Returns:
            Output tensor of shape (batch_size, num_classes).
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Transformer encoder
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return x


class VitalSignsAutoEncoder(nn.Module):
    """Autoencoder for unsupervised anomaly detection."""
    
    def __init__(
        self,
        input_size: int = 3,
        sequence_length: int = 100,
        encoding_dim: int = 32,
        dropout_rate: float = 0.2
    ):
        """Initialize the autoencoder.
        
        Args:
            input_size: Number of input features (vital signs).
            sequence_length: Length of input sequences.
            encoding_dim: Dimension of the encoded representation.
            dropout_rate: Dropout rate.
        """
        super().__init__()
        
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.encoding_dim = encoding_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size * sequence_length, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, encoding_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, input_size * sequence_length)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).
            
        Returns:
            Tuple of (encoded, decoded) tensors.
        """
        batch_size = x.size(0)
        
        # Flatten input
        x_flat = x.view(batch_size, -1)
        
        # Encode
        encoded = self.encoder(x_flat)
        
        # Decode
        decoded_flat = self.decoder(encoded)
        decoded = decoded_flat.view(batch_size, self.sequence_length, self.input_size)
        
        return encoded, decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        return self.encoder(x_flat)
    
    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output."""
        batch_size = encoded.size(0)
        decoded_flat = self.decoder(encoded)
        return decoded_flat.view(batch_size, self.sequence_length, self.input_size)


class VitalSignsEnsemble(nn.Module):
    """Ensemble model combining multiple architectures."""
    
    def __init__(
        self,
        input_size: int = 3,
        sequence_length: int = 100,
        num_classes: int = 2,
        dropout_rate: float = 0.2
    ):
        """Initialize the ensemble model.
        
        Args:
            input_size: Number of input features (vital signs).
            sequence_length: Length of input sequences.
            num_classes: Number of output classes.
            dropout_rate: Dropout rate.
        """
        super().__init__()
        
        # Individual models
        self.cnn = VitalSignsCNN(
            input_channels=input_size,
            sequence_length=sequence_length,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
        
        self.lstm = VitalSignsLSTM(
            input_size=input_size,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
        
        self.transformer = VitalSignsTransformer(
            input_size=input_size,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
        
        # Ensemble weights
        self.weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).
            
        Returns:
            Output tensor of shape (batch_size, num_classes).
        """
        # Get predictions from each model
        cnn_out = self.cnn(x)
        lstm_out = self.lstm(x)
        transformer_out = self.transformer(x)
        
        # Weighted ensemble
        weights = F.softmax(self.weights, dim=0)
        ensemble_out = (
            weights[0] * cnn_out +
            weights[1] * lstm_out +
            weights[2] * transformer_out
        )
        
        return ensemble_out


def create_model(
    model_name: str,
    input_size: int = 3,
    sequence_length: int = 100,
    num_classes: int = 2,
    **kwargs
) -> nn.Module:
    """Create a model by name.
    
    Args:
        model_name: Name of the model to create.
        input_size: Number of input features.
        sequence_length: Length of input sequences.
        num_classes: Number of output classes.
        **kwargs: Additional model parameters.
        
    Returns:
        Initialized model.
        
    Raises:
        ValueError: If model name is not recognized.
    """
    models = {
        'cnn': VitalSignsCNN,
        'lstm': VitalSignsLSTM,
        'transformer': VitalSignsTransformer,
        'autoencoder': VitalSignsAutoEncoder,
        'ensemble': VitalSignsEnsemble
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(models.keys())}")
    
    model_class = models[model_name]
    
    # Filter kwargs to only include valid parameters
    valid_params = {
        'input_size': input_size,
        'sequence_length': sequence_length,
        'num_classes': num_classes,
        **kwargs
    }
    
    # Remove invalid parameters for each model
    if model_name == 'cnn':
        valid_params = {k: v for k, v in valid_params.items() 
                       if k in ['input_channels', 'sequence_length', 'num_classes', 'dropout_rate']}
        valid_params['input_channels'] = valid_params.pop('input_size')
    elif model_name == 'lstm':
        valid_params = {k: v for k, v in valid_params.items() 
                       if k in ['input_size', 'hidden_size', 'num_layers', 'num_classes', 'dropout_rate', 'bidirectional']}
    elif model_name == 'transformer':
        valid_params = {k: v for k, v in valid_params.items() 
                       if k in ['input_size', 'd_model', 'nhead', 'num_layers', 'num_classes', 'dropout_rate', 'max_seq_length']}
    elif model_name == 'autoencoder':
        valid_params = {k: v for k, v in valid_params.items() 
                       if k in ['input_size', 'sequence_length', 'encoding_dim', 'dropout_rate']}
    elif model_name == 'ensemble':
        valid_params = {k: v for k, v in valid_params.items() 
                       if k in ['input_size', 'sequence_length', 'num_classes', 'dropout_rate']}
    
    return model_class(**valid_params)
