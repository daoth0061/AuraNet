#!/usr/bin/env python3
"""
Quick demo and test script for AuraNet
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

def test_imports():
    """Test if all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src import create_auranet, AuraNetTrainer
        print("✓ Core modules imported successfully")
    except Exception as e:
        print(f"✗ Failed to import core modules: {e}")
        return False
    
    try:
        from src.utils import LayerNorm, Block, GRN, CBAMChannelAttention
        print("✓ Utility modules imported successfully") 
    except Exception as e:
        print(f"✗ Failed to import utility modules: {e}")
        return False
    
    try:
        from src.initial_processing import MSAF, MBConvDownsample
        print("✓ Initial processing modules imported successfully")
    except Exception as e:
        print(f"✗ Failed to import initial processing modules: {e}")
        return False
    
    try:
        from src.haft import HAFT
        print("✓ HAFT module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import HAFT module: {e}")
        return False
    
    try:
        from src.cross_fusion import CrossFusionBlock
        print("✓ Cross-fusion modules imported successfully")
    except Exception as e:
        print(f"✗ Failed to import cross-fusion modules: {e}")
        return False
    
    return True


def test_model_creation():
    """Test model creation and basic forward pass."""
    print("\nTesting model creation...")
    
    try:
        from src import create_auranet
        
        # Create model with default config
        model = create_auranet()
        print("✓ Model created successfully")
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return model
    
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(model):
    """Test forward pass with dummy input."""
    print("\nTesting forward pass...")
    
    try:
        # Create dummy input
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 256, 256)
        
        print(f"Input shape: {dummy_input.shape}")
        
        with torch.no_grad():
            # Test fine-tuning mode
            outputs = model(dummy_input, mode='finetune')
            
            print("✓ Forward pass successful")
            print("Outputs:")
            for key, value in outputs.items():
                if torch.is_tensor(value):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {type(value)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_individual_components():
    """Test individual model components."""
    print("\nTesting individual components...")
    
    success = True
    
    # Test MSAF
    try:
        from src.initial_processing import MSAF
        msaf = MSAF()
        test_input = torch.randn(1, 3, 256, 256)
        
        with torch.no_grad():
            output = msaf(test_input)
        
        print(f"✓ MSAF: {test_input.shape} -> {output.shape}")
    except Exception as e:
        print(f"✗ MSAF failed: {e}")
        success = False
    
    # Test HAFT (with smaller input for speed)
    try:
        from src.haft import HAFT
        haft = HAFT(in_channels=32, num_haft_levels=2, num_radial_bins=8)
        test_input = torch.randn(1, 32, 32, 32)  # Smaller for demo
        
        with torch.no_grad():
            output = haft(test_input)
        
        print(f"✓ HAFT: {test_input.shape} -> {output.shape}")
    except Exception as e:
        print(f"✗ HAFT failed: {e}")
        success = False
    
    # Test Cross-Fusion
    try:
        from src.cross_fusion import CrossFusionBlock
        cross_fusion = CrossFusionBlock(dim=64, use_deformable=False)  # Use standard attention for simplicity
        spatial_feat = torch.randn(1, 64, 16, 16)
        freq_feat = torch.randn(1, 64, 16, 16)
        
        with torch.no_grad():
            enhanced_spatial, enhanced_freq = cross_fusion(spatial_feat, freq_feat)
        
        print(f"✓ Cross-Fusion: inputs {spatial_feat.shape} -> outputs {enhanced_spatial.shape}")
    except Exception as e:
        print(f"✗ Cross-Fusion failed: {e}")
        success = False
    
    return success


def test_configuration():
    """Test configuration loading and customization."""
    print("\nTesting configuration...")
    
    try:
        from src import create_auranet
        
        # Test custom configuration
        custom_config = {
            'img_size': [224, 224],
            'depths': [1, 1, 1, 1],  # Minimal depth for quick test
            'dims': [32, 64, 128, 256]  # Smaller dimensions
        }
        
        model = create_auranet(**custom_config)
        config = model.get_config()
        
        print("✓ Configuration loaded successfully")
        print(f"  Image size: {config['img_size']}")
        print(f"  Depths: {config['depths']}")
        print(f"  Dimensions: {config['dims']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def run_quick_demo():
    """Run a comprehensive quick demo."""
    print("AuraNet Quick Demo")
    print("=" * 50)
    
    # Test 1: Imports
    if not test_imports():
        print("❌ Import test failed!")
        return False
    
    # Test 2: Model creation
    model = test_model_creation()
    if model is None:
        print("❌ Model creation failed!")
        return False
    
    # Test 3: Forward pass
    if not test_forward_pass(model):
        print("❌ Forward pass test failed!")
        return False
    
    # Test 4: Individual components
    if not test_individual_components():
        print("❌ Component test failed!")
        return False
    
    # Test 5: Configuration
    if not test_configuration():
        print("❌ Configuration test failed!")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed successfully!")
    print("✅ AuraNet is ready to use!")
    print("\nNext steps:")
    print("1. Prepare your dataset")
    print("2. Run training with: python train_example.py")
    print("3. Use the trained model for inference")
    
    return True


if __name__ == '__main__':
    success = run_quick_demo()
    exit(0 if success else 1)
