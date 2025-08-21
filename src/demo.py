"""
Quick demo script for AuraNet
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os

from auranet import create_auranet
from evaluate import AuraNetEvaluator


def create_demo_model():
    """Create a demo model with random weights for testing."""
    print("Creating demo AuraNet model...")
    
    # Use default configuration
    model = create_auranet()
    model.eval()
    
    print("Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model


def demo_forward_pass():
    """Demonstrate a forward pass through the model."""
    print("\n" + "="*50)
    print("DEMO: Forward Pass")
    print("="*50)
    
    model = create_demo_model()
    
    # Create dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 256, 256)
    
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        # Test pre-training mode
        print("\nTesting pre-training mode...")
        pretrain_outputs = model(dummy_input, mode='pretrain')
        
        print("Pre-training outputs:")
        for key, value in pretrain_outputs.items():
            if torch.is_tensor(value):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")
        
        # Test fine-tuning mode
        print("\nTesting fine-tuning mode...")
        finetune_outputs = model(dummy_input, mode='finetune')
        
        print("Fine-tuning outputs:")
        for key, value in finetune_outputs.items():
            if torch.is_tensor(value):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")
    
    print("\nForward pass completed successfully!")


def demo_individual_components():
    """Demonstrate individual model components."""
    print("\n" + "="*50)
    print("DEMO: Individual Components")
    print("="*50)
    
    # Test MSAF
    print("Testing MSAF component...")
    from initial_processing import MSAF
    
    msaf = MSAF()
    test_input = torch.randn(1, 3, 256, 256)
    
    try:
        with torch.no_grad():
            msaf_output = msaf(test_input)
        print(f"✓ MSAF: {test_input.shape} -> {msaf_output.shape}")
    except Exception as e:
        print(f"✗ MSAF failed: {e}")
    
    # Test HAFT
    print("Testing HAFT component...")
    from haft import HAFT
    
    haft = HAFT(in_channels=64, num_haft_levels=3)
    test_input = torch.randn(1, 64, 64, 64)
    
    try:
        with torch.no_grad():
            haft_output = haft(test_input)
        print(f"✓ HAFT: {test_input.shape} -> {haft_output.shape}")
    except Exception as e:
        print(f"✗ HAFT failed: {e}")
    
    # Test Cross-Fusion
    print("Testing Cross-Fusion component...")
    from cross_fusion import CrossFusionBlock
    
    cross_fusion = CrossFusionBlock(dim=64, use_deformable=True)
    spatial_feat = torch.randn(1, 64, 32, 32)
    freq_feat = torch.randn(1, 64, 32, 32)
    
    try:
        with torch.no_grad():
            enhanced_spatial, enhanced_freq = cross_fusion(spatial_feat, freq_feat)
        print(f"✓ Cross-Fusion: {spatial_feat.shape} -> {enhanced_spatial.shape}, {enhanced_freq.shape}")
    except Exception as e:
        print(f"✗ Cross-Fusion failed: {e}")
    
    # Test DSF
    print("Testing DSF component...")
    from output_heads import DSF
    
    dsf = DSF(dim=512)
    spatial_feat = torch.randn(1, 512, 8, 8)
    freq_feat = torch.randn(1, 512, 8, 8)
    
    try:
        with torch.no_grad():
            fused_output, pooled_output = dsf(spatial_feat, freq_feat)
        print(f"✓ DSF: ({spatial_feat.shape}, {freq_feat.shape}) -> {fused_output.shape}, {pooled_output.shape}")
    except Exception as e:
        print(f"✗ DSF failed: {e}")


def demo_configuration():
    """Demonstrate configuration loading."""
    print("\n" + "="*50)
    print("DEMO: Configuration")
    print("="*50)
    
    # Test with default config
    print("Creating model with default configuration...")
    model_default = create_auranet()
    config_default = model_default.get_config()
    
    print("Default configuration:")
    for key, value in config_default.items():
        print(f"  {key}: {value}")
    
    # Test with custom config
    print("\nCreating model with custom configuration...")
    custom_config = {
        'img_size': [224, 224],
        'depths': [1, 1, 2, 1],
        'dims': [32, 64, 128, 256]
    }
    
    model_custom = create_auranet(**custom_config)
    config_custom = model_custom.get_config()
    
    print("Custom configuration (modified values only):")
    for key in custom_config:
        print(f"  {key}: {config_custom[key]}")
    
    print(f"\nParameter count comparison:")
    print(f"  Default model: {sum(p.numel() for p in model_default.parameters()):,}")
    print(f"  Custom model: {sum(p.numel() for p in model_custom.parameters()):,}")


def create_sample_image(save_path="demo_image.jpg"):
    """Create a sample image for testing."""
    # Create a simple test image
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Add some patterns to make it more interesting
    # Horizontal stripes
    for i in range(0, 256, 20):
        image[i:i+10, :, :] = [255, 0, 0]  # Red stripes
    
    # Vertical stripes
    for j in range(0, 256, 30):
        image[:, j:j+15, :] = [0, 255, 0]  # Green stripes
    
    # Save image
    pil_image = Image.fromarray(image)
    pil_image.save(save_path)
    
    print(f"Sample image saved to {save_path}")
    return save_path


def demo_complete_pipeline():
    """Demonstrate the complete pipeline with a sample image."""
    print("\n" + "="*50)
    print("DEMO: Complete Pipeline")
    print("="*50)
    
    # Create sample image
    sample_image_path = create_sample_image()
    
    # Create and test model
    print("Creating model...")
    model = create_demo_model()
    
    # Test image loading and preprocessing
    print("Loading and preprocessing image...")
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(sample_image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    print(f"Preprocessed image shape: {input_tensor.shape}")
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model(input_tensor, mode='finetune')
    
    print("Inference results:")
    logits = outputs['classification_logits']
    probs = torch.softmax(logits, dim=1)
    pred_label = torch.argmax(logits, dim=1).item()
    
    print(f"  Predicted label: {pred_label} ({'Fake' if pred_label == 1 else 'Real'})")
    print(f"  Confidence: {probs.max().item():.3f}")
    print(f"  Probabilities: Real={probs[0, 0].item():.3f}, Fake={probs[0, 1].item():.3f}")
    
    seg_mask = outputs['segmentation_mask']
    print(f"  Segmentation mask shape: {seg_mask.shape}")
    print(f"  Mask value range: [{seg_mask.min().item():.3f}, {seg_mask.max().item():.3f}]")
    
    # Clean up
    os.remove(sample_image_path)
    print("\nDemo completed successfully!")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='AuraNet Demo')
    parser.add_argument('--demo', type=str, 
                       choices=['forward', 'components', 'config', 'pipeline', 'all'],
                       default='all', help='Which demo to run')
    
    args = parser.parse_args()
    
    print("AuraNet - Dual-Stream Forensic Network")
    print("Demo Script")
    print("="*50)
    
    try:
        if args.demo == 'forward' or args.demo == 'all':
            demo_forward_pass()
        
        if args.demo == 'components' or args.demo == 'all':
            demo_individual_components()
        
        if args.demo == 'config' or args.demo == 'all':
            demo_configuration()
        
        if args.demo == 'pipeline' or args.demo == 'all':
            demo_complete_pipeline()
        
        print("\n" + "="*50)
        print("All demos completed successfully!")
        print("The AuraNet implementation is ready to use.")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
