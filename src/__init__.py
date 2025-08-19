# AuraNet - Dual-Stream Forensic Network

## Package Initialization
"""
AuraNet: A Dual-Stream Forensic Network for Face Manipulation Detection

This package implements a state-of-the-art deep learning model for detecting face manipulation
in images. The model uses a dual-stream architecture that processes images through both
spatial and frequency domains for robust detection.

Key Components:
- MSAF: Multi-Scale Artifact Fusion for initial processing
- HAFT: Hierarchical Adaptive Frequency Transform for frequency stream
- Cross-Fusion: Bidirectional attention between spatial and frequency streams
- DSF: Dynamic Selection Fusion for final output generation

Example Usage:
    ```python
    from auranet import create_auranet, AuraNetTrainer
    
    # Create model
    model = create_auranet(config_path='config.yaml')
    
    # Initialize trainer
    trainer = AuraNetTrainer(config_path='config.yaml', save_dir='./checkpoints')
    
    # Train model
    trainer.finetune(train_loader, val_loader, num_epochs=50)
    ```
"""

from .auranet import create_auranet, AuraNet
from .train import AuraNetTrainer
from .data_loader import AuraNetDataset, create_data_loaders
from .training import (
    CombinedPretrainLoss, 
    CombinedFinetuneLoss,
    get_optimizer,
    get_scheduler
)

__version__ = "1.0.0"
__author__ = "AuraNet Team"
__email__ = "auranet@example.com"

__all__ = [
    'create_auranet',
    'AuraNet', 
    'AuraNetTrainer',
    'AuraNetDataset',
    'create_data_loaders',
    'CombinedPretrainLoss',
    'CombinedFinetuneLoss',
    'get_optimizer',
    'get_scheduler'
]
