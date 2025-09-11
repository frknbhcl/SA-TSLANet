"""Default configuration for SA-TSLANet."""

from dataclasses import dataclass


@dataclass
class SAConfig:
    """SA-TSLANet configuration."""
    
    # Model architecture
    seq_len: int = 1000
    pred_len: int = 500
    window_len: int = 1500
    patch_size_detect: int = 4
    patch_size_forecast: int = 16
    emb_dim: int = 64
    depth: int = 2
    dropout: float = 0.1
    
    # Training
    batch_size: int = 6
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    train_epochs: int = 20
    
    # Data processing
    sampling_rate: int = 25600
    stride_train: int = 750
    stride_test: int = 750
    
    # Loss weights
    a_spindle: float = 1e8
    a_ratio: float = 1.0
    a_crest: float = 1.0
    
    # Features
    adaptive_filter: bool = True
    ICB: bool = True
    ASB: bool = True
