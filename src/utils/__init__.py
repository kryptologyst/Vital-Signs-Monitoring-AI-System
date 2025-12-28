"""Core utilities for health monitoring system."""

import random
import logging
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
from omegaconf import DictConfig


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        PyTorch device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        level: Logging level.
        log_file: Optional log file path.
        
    Returns:
        Configured logger.
    """
    logger = logging.getLogger("health_monitoring")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class VitalSignsThresholds:
    """Clinical thresholds for vital signs monitoring."""
    
    # Normal ranges for adults
    HEART_RATE = (60, 100)  # bpm
    TEMPERATURE = (36.1, 37.5)  # Celsius
    SPO2 = (95, 100)  # percentage
    RESPIRATORY_RATE = (12, 20)  # breaths per minute
    BLOOD_PRESSURE_SYSTOLIC = (90, 140)  # mmHg
    BLOOD_PRESSURE_DIASTOLIC = (60, 90)  # mmHg
    
    # Critical thresholds
    CRITICAL_HEART_RATE = (40, 150)
    CRITICAL_TEMPERATURE = (35.0, 40.0)
    CRITICAL_SPO2 = (85, 100)
    
    @classmethod
    def get_normal_range(cls, vital_sign: str) -> Tuple[float, float]:
        """Get normal range for a vital sign.
        
        Args:
            vital_sign: Name of the vital sign.
            
        Returns:
            Tuple of (lower_bound, upper_bound).
            
        Raises:
            ValueError: If vital sign is not recognized.
        """
        vital_sign = vital_sign.lower().replace(" ", "_")
        
        ranges = {
            "heart_rate": cls.HEART_RATE,
            "temperature": cls.TEMPERATURE,
            "spo2": cls.SPO2,
            "respiratory_rate": cls.RESPIRATORY_RATE,
            "blood_pressure_systolic": cls.BLOOD_PRESSURE_SYSTOLIC,
            "blood_pressure_diastolic": cls.BLOOD_PRESSURE_DIASTOLIC,
        }
        
        if vital_sign not in ranges:
            raise ValueError(f"Unknown vital sign: {vital_sign}")
        
        return ranges[vital_sign]
    
    @classmethod
    def get_critical_range(cls, vital_sign: str) -> Tuple[float, float]:
        """Get critical range for a vital sign.
        
        Args:
            vital_sign: Name of the vital sign.
            
        Returns:
            Tuple of (lower_bound, upper_bound).
            
        Raises:
            ValueError: If vital sign is not recognized.
        """
        vital_sign = vital_sign.lower().replace(" ", "_")
        
        ranges = {
            "heart_rate": cls.CRITICAL_HEART_RATE,
            "temperature": cls.CRITICAL_TEMPERATURE,
            "spo2": cls.CRITICAL_SPO2,
        }
        
        if vital_sign not in ranges:
            raise ValueError(f"No critical range defined for: {vital_sign}")
        
        return ranges[vital_sign]
    
    @classmethod
    def is_normal(cls, vital_sign: str, value: float) -> bool:
        """Check if a vital sign value is within normal range.
        
        Args:
            vital_sign: Name of the vital sign.
            value: Value to check.
            
        Returns:
            True if value is within normal range.
        """
        try:
            low, high = cls.get_normal_range(vital_sign)
            return low <= value <= high
        except ValueError:
            return False
    
    @classmethod
    def is_critical(cls, vital_sign: str, value: float) -> bool:
        """Check if a vital sign value is in critical range.
        
        Args:
            vital_sign: Name of the vital sign.
            value: Value to check.
            
        Returns:
            True if value is in critical range.
        """
        try:
            low, high = cls.get_critical_range(vital_sign)
            return low <= value <= high
        except ValueError:
            return False


def validate_config(config: DictConfig) -> DictConfig:
    """Validate configuration parameters.
    
    Args:
        config: Configuration object.
        
    Returns:
        Validated configuration.
        
    Raises:
        ValueError: If configuration is invalid.
    """
    required_keys = ["model", "data", "training"]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Validate model config
    if "name" not in config.model:
        raise ValueError("Model name is required")
    
    # Validate data config
    if "sequence_length" not in config.data:
        raise ValueError("Data sequence_length is required")
    
    if config.data.sequence_length <= 0:
        raise ValueError("Sequence length must be positive")
    
    # Validate training config
    if "batch_size" not in config.training:
        raise ValueError("Training batch_size is required")
    
    if config.training.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    
    return config


def format_vital_sign_name(name: str) -> str:
    """Format vital sign name for display.
    
    Args:
        name: Raw vital sign name.
        
    Returns:
        Formatted name.
    """
    return name.replace("_", " ").title()


def calculate_anomaly_score(
    values: np.ndarray, 
    baseline_mean: float, 
    baseline_std: float
) -> np.ndarray:
    """Calculate anomaly scores using z-score normalization.
    
    Args:
        values: Array of vital sign values.
        baseline_mean: Baseline mean value.
        baseline_std: Baseline standard deviation.
        
    Returns:
        Array of anomaly scores.
    """
    if baseline_std == 0:
        return np.zeros_like(values)
    
    return np.abs((values - baseline_mean) / baseline_std)


def smooth_signal(signal: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Apply moving average smoothing to a signal.
    
    Args:
        signal: Input signal.
        window_size: Size of the smoothing window.
        
    Returns:
        Smoothed signal.
    """
    if window_size <= 1:
        return signal
    
    # Pad the signal
    padded = np.pad(signal, (window_size // 2, window_size - window_size // 2), mode='edge')
    
    # Apply moving average
    smoothed = np.convolve(padded, np.ones(window_size) / window_size, mode='valid')
    
    return smoothed
