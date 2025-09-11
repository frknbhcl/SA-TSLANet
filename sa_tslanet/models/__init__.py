"""SA-TSLANet: Spindle-Aware Time Series Lightweight Adaptive Network for Chatter Detection."""

from .models.sa_tslanet import TSLANet_Detect
from .models.components import (
    SpindleMaskGenerator,
    ICB,
    Modified_Adaptive_Spectral_Block,
    Modified_TSLANet_layer
)
from .data.preprocessing import TimeSeriesPreprocessor, ChatterDataset
from .data.loaders import prepare_train_val_loaders, create_test_loader
from .training.trainer import ChatterDetectionTrainer
from .utils.visualization import DetectionVisualizer, plot_training_history
from .utils.metrics import calculate_metrics, evaluate_detection
from .config.default_config import SAConfig

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    # Models
    "TSLANet_Detect",
    "SpindleMaskGenerator",
    "ICB",
    "Modified_Adaptive_Spectral_Block",
    "Modified_TSLANet_layer",
    
    # Data
    "TimeSeriesPreprocessor",
    "ChatterDataset",
    "prepare_train_val_loaders",
    "create_test_loader",
    
    # Training
    "ChatterDetectionTrainer",
    
    # Utils
    "DetectionVisualizer",
    "plot_training_history",
    "calculate_metrics",
    "evaluate_detection",
    
    # Config
    "SAConfig",
]

# Package metadata
__doc__ = """
SA-TSLANet: Advanced deep learning model for chatter detection in milling operations.

This package provides:
- Spindle-aware frequency domain processing
- Multi-objective loss functions for robust detection
- Real-time capable inference
- Comprehensive visualization tools
"""
