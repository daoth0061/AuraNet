"""
Script to count AuraNet model parameters
"""

import sys
from pathlib import Path
import torch
import yaml

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from auranet import create_auranet

def count_parameters(model):
    """Count model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params

def format_number(num):
    """Format number with appropriate units."""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)

def main():
    """Main function."""
    # Load configuration
    config_path = Path(__file__).parent / 'config_celeb_df.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Creating AuraNet model...")
    
    # Create model
    model = create_auranet(config=config)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    
    print("\n" + "="*60)
    print("AuraNet Model Parameter Count")
    print("="*60)
    print(f"Total parameters:     {format_number(total_params):>15} ({total_params:,})")
    print(f"Trainable parameters: {format_number(trainable_params):>15} ({trainable_params:,})")
    print(f"Non-trainable params: {format_number(total_params - trainable_params):>15} ({total_params - trainable_params:,})")
    print("="*60)
    
    # Print model summary by modules
    print("\nModel Structure Summary:")
    print("-"*60)
    
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        print(f"{name:<25}: {format_number(module_params):>10} params")
    
    # Memory estimation (assuming float32)
    model_size_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
    print(f"\nEstimated model size: {model_size_mb:.2f} MB")
    
    # Input tensor size estimation for batch_size=1, img_size=[256, 256]
    img_size = config.get('img_size', [256, 256])
    input_size_mb = (1 * 3 * img_size[0] * img_size[1] * 4) / (1024 * 1024)
    print(f"Input tensor size (1, 3, {img_size[0]}, {img_size[1]}): {input_size_mb:.2f} MB")

if __name__ == '__main__':
    main()
