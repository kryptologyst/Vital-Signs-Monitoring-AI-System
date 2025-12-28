"""Data generation and processing for vital signs monitoring."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import signal
import logging

from .utils import VitalSignsThresholds, format_vital_sign_name


logger = logging.getLogger(__name__)


@dataclass
class VitalSignsData:
    """Container for vital signs data."""
    
    heart_rate: np.ndarray
    temperature: np.ndarray
    spo2: np.ndarray
    respiratory_rate: Optional[np.ndarray] = None
    blood_pressure_systolic: Optional[np.ndarray] = None
    blood_pressure_diastolic: Optional[np.ndarray] = None
    timestamps: Optional[np.ndarray] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        data = {
            'heart_rate': self.heart_rate,
            'temperature': self.temperature,
            'spo2': self.spo2,
        }
        
        if self.respiratory_rate is not None:
            data['respiratory_rate'] = self.respiratory_rate
        if self.blood_pressure_systolic is not None:
            data['blood_pressure_systolic'] = self.blood_pressure_systolic
        if self.blood_pressure_diastolic is not None:
            data['blood_pressure_diastolic'] = self.blood_pressure_diastolic
        if self.timestamps is not None:
            data['timestamp'] = self.timestamps
            
        return pd.DataFrame(data)
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get data shape (n_samples, n_features)."""
        n_samples = len(self.heart_rate)
        n_features = 3  # heart_rate, temperature, spo2
        
        if self.respiratory_rate is not None:
            n_features += 1
        if self.blood_pressure_systolic is not None:
            n_features += 1
        if self.blood_pressure_diastolic is not None:
            n_features += 1
            
        return (n_samples, n_features)


class VitalSignsGenerator:
    """Generate synthetic vital signs data for research and testing."""
    
    def __init__(self, seed: int = 42):
        """Initialize the generator.
        
        Args:
            seed: Random seed for reproducibility.
        """
        self.rng = np.random.RandomState(seed)
        self.thresholds = VitalSignsThresholds()
    
    def generate_normal_vitals(
        self, 
        n_samples: int, 
        sampling_rate: float = 1.0,
        add_noise: bool = True
    ) -> VitalSignsData:
        """Generate normal vital signs data.
        
        Args:
            n_samples: Number of samples to generate.
            sampling_rate: Sampling rate in Hz.
            add_noise: Whether to add realistic noise.
            
        Returns:
            VitalSignsData object with normal vital signs.
        """
        logger.info(f"Generating {n_samples} normal vital signs samples")
        
        # Generate timestamps
        timestamps = np.arange(n_samples) / sampling_rate
        
        # Generate normal heart rate (60-100 bpm with slight variation)
        hr_mean = self.rng.uniform(70, 85)
        hr_std = self.rng.uniform(5, 10)
        heart_rate = self.rng.normal(hr_mean, hr_std, n_samples)
        heart_rate = np.clip(heart_rate, 60, 100)
        
        # Generate normal temperature (36.1-37.5°C)
        temp_mean = self.rng.uniform(36.3, 37.0)
        temp_std = self.rng.uniform(0.1, 0.3)
        temperature = self.rng.normal(temp_mean, temp_std, n_samples)
        temperature = np.clip(temperature, 36.1, 37.5)
        
        # Generate normal SpO2 (95-100%)
        spo2_mean = self.rng.uniform(97, 99)
        spo2_std = self.rng.uniform(0.5, 1.5)
        spo2 = self.rng.normal(spo2_mean, spo2_std, n_samples)
        spo2 = np.clip(spo2, 95, 100)
        
        # Add realistic noise and correlations
        if add_noise:
            heart_rate = self._add_physiological_noise(heart_rate, timestamps)
            temperature = self._add_physiological_noise(temperature, timestamps)
            spo2 = self._add_physiological_noise(spo2, timestamps)
        
        return VitalSignsData(
            heart_rate=heart_rate,
            temperature=temperature,
            spo2=spo2,
            timestamps=timestamps
        )
    
    def generate_anomalous_vitals(
        self, 
        n_samples: int,
        anomaly_types: List[str] = None,
        sampling_rate: float = 1.0
    ) -> VitalSignsData:
        """Generate vital signs with anomalies.
        
        Args:
            n_samples: Number of samples to generate.
            anomaly_types: Types of anomalies to include.
            sampling_rate: Sampling rate in Hz.
            
        Returns:
            VitalSignsData object with anomalous vital signs.
        """
        if anomaly_types is None:
            anomaly_types = ['tachycardia', 'hypothermia', 'hypoxemia']
        
        logger.info(f"Generating {n_samples} samples with anomalies: {anomaly_types}")
        
        # Start with normal vitals
        normal_data = self.generate_normal_vitals(n_samples, sampling_rate)
        
        # Apply anomalies
        for anomaly_type in anomaly_types:
            if anomaly_type == 'tachycardia':
                self._add_tachycardia(normal_data, n_samples)
            elif anomaly_type == 'bradycardia':
                self._add_bradycardia(normal_data, n_samples)
            elif anomaly_type == 'hyperthermia':
                self._add_hyperthermia(normal_data, n_samples)
            elif anomaly_type == 'hypothermia':
                self._add_hypothermia(normal_data, n_samples)
            elif anomaly_type == 'hypoxemia':
                self._add_hypoxemia(normal_data, n_samples)
            elif anomaly_type == 'hypertension':
                self._add_hypertension(normal_data, n_samples)
            elif anomaly_type == 'hypotension':
                self._add_hypotension(normal_data, n_samples)
        
        return normal_data
    
    def _add_physiological_noise(self, signal: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """Add realistic physiological noise to a signal."""
        # Add high-frequency noise
        noise = self.rng.normal(0, 0.1, len(signal))
        
        # Add low-frequency drift
        drift = np.sin(2 * np.pi * timestamps / 3600) * 0.05  # 1-hour cycle
        
        return signal + noise + drift
    
    def _add_tachycardia(self, data: VitalSignsData, n_samples: int) -> None:
        """Add tachycardia episodes."""
        # Randomly select time windows for tachycardia
        n_episodes = self.rng.randint(1, 3)
        for _ in range(n_episodes):
            start = self.rng.randint(0, n_samples - 100)
            duration = self.rng.randint(50, 200)
            end = min(start + duration, n_samples)
            
            # Increase heart rate to 100-150 bpm
            data.heart_rate[start:end] = self.rng.uniform(100, 150, end - start)
    
    def _add_bradycardia(self, data: VitalSignsData, n_samples: int) -> None:
        """Add bradycardia episodes."""
        n_episodes = self.rng.randint(1, 2)
        for _ in range(n_episodes):
            start = self.rng.randint(0, n_samples - 100)
            duration = self.rng.randint(50, 150)
            end = min(start + duration, n_samples)
            
            # Decrease heart rate to 40-60 bpm
            data.heart_rate[start:end] = self.rng.uniform(40, 60, end - start)
    
    def _add_hyperthermia(self, data: VitalSignsData, n_samples: int) -> None:
        """Add hyperthermia episodes."""
        n_episodes = self.rng.randint(1, 2)
        for _ in range(n_episodes):
            start = self.rng.randint(0, n_samples - 100)
            duration = self.rng.randint(30, 120)
            end = min(start + duration, n_samples)
            
            # Increase temperature to 37.5-40.0°C
            data.temperature[start:end] = self.rng.uniform(37.5, 40.0, end - start)
    
    def _add_hypothermia(self, data: VitalSignsData, n_samples: int) -> None:
        """Add hypothermia episodes."""
        n_episodes = self.rng.randint(1, 2)
        for _ in range(n_episodes):
            start = self.rng.randint(0, n_samples - 100)
            duration = self.rng.randint(30, 120)
            end = min(start + duration, n_samples)
            
            # Decrease temperature to 35.0-36.1°C
            data.temperature[start:end] = self.rng.uniform(35.0, 36.1, end - start)
    
    def _add_hypoxemia(self, data: VitalSignsData, n_samples: int) -> None:
        """Add hypoxemia episodes."""
        n_episodes = self.rng.randint(1, 3)
        for _ in range(n_episodes):
            start = self.rng.randint(0, n_samples - 100)
            duration = self.rng.randint(20, 100)
            end = min(start + duration, n_samples)
            
            # Decrease SpO2 to 85-95%
            data.spo2[start:end] = self.rng.uniform(85, 95, end - start)
    
    def _add_hypertension(self, data: VitalSignsData, n_samples: int) -> None:
        """Add hypertension episodes."""
        if data.blood_pressure_systolic is None:
            data.blood_pressure_systolic = np.full(n_samples, 120)
            data.blood_pressure_diastolic = np.full(n_samples, 80)
        
        n_episodes = self.rng.randint(1, 2)
        for _ in range(n_episodes):
            start = self.rng.randint(0, n_samples - 100)
            duration = self.rng.randint(30, 120)
            end = min(start + duration, n_samples)
            
            # Increase blood pressure
            data.blood_pressure_systolic[start:end] = self.rng.uniform(140, 180, end - start)
            data.blood_pressure_diastolic[start:end] = self.rng.uniform(90, 110, end - start)
    
    def _add_hypotension(self, data: VitalSignsData, n_samples: int) -> None:
        """Add hypotension episodes."""
        if data.blood_pressure_systolic is None:
            data.blood_pressure_systolic = np.full(n_samples, 120)
            data.blood_pressure_diastolic = np.full(n_samples, 80)
        
        n_episodes = self.rng.randint(1, 2)
        for _ in range(n_episodes):
            start = self.rng.randint(0, n_samples - 100)
            duration = self.rng.randint(30, 120)
            end = min(start + duration, n_samples)
            
            # Decrease blood pressure
            data.blood_pressure_systolic[start:end] = self.rng.uniform(70, 90, end - start)
            data.blood_pressure_diastolic[start:end] = self.rng.uniform(40, 60, end - start)


class VitalSignsProcessor:
    """Process and preprocess vital signs data."""
    
    def __init__(self, sampling_rate: float = 1.0):
        """Initialize the processor.
        
        Args:
            sampling_rate: Sampling rate in Hz.
        """
        self.sampling_rate = sampling_rate
        self.thresholds = VitalSignsThresholds()
    
    def preprocess(self, data: VitalSignsData) -> VitalSignsData:
        """Apply preprocessing to vital signs data.
        
        Args:
            data: Input vital signs data.
            
        Returns:
            Preprocessed vital signs data.
        """
        logger.info("Preprocessing vital signs data")
        
        # Create a copy to avoid modifying original data
        processed_data = VitalSignsData(
            heart_rate=data.heart_rate.copy(),
            temperature=data.temperature.copy(),
            spo2=data.spo2.copy(),
            respiratory_rate=data.respiratory_rate.copy() if data.respiratory_rate is not None else None,
            blood_pressure_systolic=data.blood_pressure_systolic.copy() if data.blood_pressure_systolic is not None else None,
            blood_pressure_diastolic=data.blood_pressure_diastolic.copy() if data.blood_pressure_diastolic is not None else None,
            timestamps=data.timestamps.copy() if data.timestamps is not None else None
        )
        
        # Remove outliers
        processed_data.heart_rate = self._remove_outliers(processed_data.heart_rate)
        processed_data.temperature = self._remove_outliers(processed_data.temperature)
        processed_data.spo2 = self._remove_outliers(processed_data.spo2)
        
        # Apply smoothing
        processed_data.heart_rate = self._smooth_signal(processed_data.heart_rate)
        processed_data.temperature = self._smooth_signal(processed_data.temperature)
        processed_data.spo2 = self._smooth_signal(processed_data.spo2)
        
        # Normalize
        processed_data.heart_rate = self._normalize_signal(processed_data.heart_rate)
        processed_data.temperature = self._normalize_signal(processed_data.temperature)
        processed_data.spo2 = self._normalize_signal(processed_data.spo2)
        
        return processed_data
    
    def _remove_outliers(self, signal: np.ndarray, method: str = 'iqr') -> np.ndarray:
        """Remove outliers from a signal.
        
        Args:
            signal: Input signal.
            method: Outlier removal method ('iqr' or 'zscore').
            
        Returns:
            Signal with outliers removed.
        """
        if method == 'iqr':
            Q1 = np.percentile(signal, 25)
            Q3 = np.percentile(signal, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Clip outliers
            signal = np.clip(signal, lower_bound, upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((signal - np.mean(signal)) / np.std(signal))
            signal = np.where(z_scores > 3, np.median(signal), signal)
        
        return signal
    
    def _smooth_signal(self, signal: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Apply moving average smoothing to a signal.
        
        Args:
            signal: Input signal.
            window_size: Size of the smoothing window.
            
        Returns:
            Smoothed signal.
        """
        if window_size <= 1:
            return signal
        
        # Use scipy's uniform_filter for efficient smoothing
        from scipy.ndimage import uniform_filter1d
        return uniform_filter1d(signal, size=window_size, mode='nearest')
    
    def _normalize_signal(self, signal: np.ndarray, method: str = 'zscore') -> np.ndarray:
        """Normalize a signal.
        
        Args:
            signal: Input signal.
            method: Normalization method ('zscore' or 'minmax').
            
        Returns:
            Normalized signal.
        """
        if method == 'zscore':
            mean = np.mean(signal)
            std = np.std(signal)
            if std == 0:
                return signal - mean
            return (signal - mean) / std
        
        elif method == 'minmax':
            min_val = np.min(signal)
            max_val = np.max(signal)
            if max_val == min_val:
                return np.zeros_like(signal)
            return (signal - min_val) / (max_val - min_val)
        
        return signal
    
    def create_sequences(
        self, 
        data: VitalSignsData, 
        sequence_length: int,
        overlap: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series modeling.
        
        Args:
            data: Vital signs data.
            sequence_length: Length of each sequence.
            overlap: Overlap ratio between sequences.
            
        Returns:
            Tuple of (sequences, labels) where labels indicate anomaly presence.
        """
        logger.info(f"Creating sequences with length {sequence_length}")
        
        # Convert to array
        features = []
        features.append(data.heart_rate)
        features.append(data.temperature)
        features.append(data.spo2)
        
        if data.respiratory_rate is not None:
            features.append(data.respiratory_rate)
        if data.blood_pressure_systolic is not None:
            features.append(data.blood_pressure_systolic)
        if data.blood_pressure_diastolic is not None:
            features.append(data.blood_pressure_diastolic)
        
        data_array = np.column_stack(features)
        
        # Create sequences
        step_size = int(sequence_length * (1 - overlap))
        sequences = []
        labels = []
        
        for i in range(0, len(data_array) - sequence_length + 1, step_size):
            sequence = data_array[i:i + sequence_length]
            sequences.append(sequence)
            
            # Label as anomalous if any vital sign is outside normal range
            is_anomalous = False
            for j, vital_sign in enumerate(['heart_rate', 'temperature', 'spo2']):
                if j < len(features):
                    values = sequence[:, j]
                    if not all(self.thresholds.is_normal(vital_sign, val) for val in values):
                        is_anomalous = True
                        break
            
            labels.append(1 if is_anomalous else 0)
        
        return np.array(sequences), np.array(labels)
