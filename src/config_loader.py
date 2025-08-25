"""
Configuration utilities for AuraNet multi-resolution training.
Handles loading base config + resolution-specific overrides.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


def deep_merge(base_dict: Dict, override_dict: Dict) -> Dict:
    """
    Deep merge two dictionaries. Override dict values take precedence.
    
    Args:
        base_dict: Base dictionary
        override_dict: Dictionary with override values
        
    Returns:
        Merged dictionary
    """
    result = base_dict.copy()
    
    for key, value in override_dict.items():
        if (key in result and 
            isinstance(result[key], dict) and 
            isinstance(value, dict)):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def load_config(base_config_path: str, override_config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from base config file and optional override file.
    
    Args:
        base_config_path: Path to base config file
        override_config_path: Optional path to override config file
        
    Returns:
        Merged configuration dictionary
    """
    # Load base configuration
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # If override config is provided, load and merge it
    if override_config_path:
        with open(override_config_path, 'r') as f:
            override_config = yaml.safe_load(f)
        final_config = deep_merge(base_config, override_config)
    else:
        final_config = base_config
    
    return final_config


def load_resolution_config(img_size: int, config_dir: str = "configs") -> Dict[str, Any]:
    """
    Load configuration for specific image resolution.
    Combines base config with resolution-specific overrides.
    
    Args:
        img_size: Image size (64, 128, or 256)
        config_dir: Directory containing config files
        
    Returns:
        Complete configuration dictionary
    """
    config_dir = Path(config_dir)
    
    # Load base configuration
    base_config_path = config_dir / "base_config.yaml"
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")
    
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Load resolution-specific configuration
    resolution_config_path = config_dir / f"config_{img_size}.yaml"
    if not resolution_config_path.exists():
        raise FileNotFoundError(f"Resolution config not found: {resolution_config_path}")
    
    with open(resolution_config_path, 'r') as f:
        resolution_config = yaml.safe_load(f)
    
    # Merge configurations (resolution config overrides base)
    final_config = deep_merge(base_config, resolution_config)
    
    return final_config


def save_merged_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save merged configuration to file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save the config file
    """
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def update_config_from_args(config: Dict[str, Any], args) -> Dict[str, Any]:
    """
    Update configuration with command-line arguments.
    
    Args:
        config: Base configuration dictionary
        args: Command line arguments
        
    Returns:
        Updated configuration
    """
    config = config.copy()
    
    # Update data root if provided
    if hasattr(args, 'data_root') and args.data_root:
        config['dataset']['data_root'] = args.data_root
    
    # Update batch size if provided
    if hasattr(args, 'batch_size') and args.batch_size:
        if 'pretrain' in config.get('training', {}):
            config['training']['pretrain']['batch_size'] = args.batch_size
        if 'finetune' in config.get('training', {}):
            config['training']['finetune']['batch_size'] = args.batch_size
    
    # Update learning rate if provided
    if hasattr(args, 'learning_rate') and args.learning_rate:
        if 'pretrain' in config.get('training', {}):
            config['training']['pretrain']['learning_rate'] = args.learning_rate
        if 'finetune' in config.get('training', {}):
            config['training']['finetune']['encoder_lr'] = args.learning_rate * 0.1
            config['training']['finetune']['head_lr'] = args.learning_rate
    
    # Update epochs if provided
    if hasattr(args, 'epochs') and args.epochs:
        if 'pretrain' in config.get('training', {}):
            config['training']['pretrain']['epochs'] = args.epochs
        if 'finetune' in config.get('training', {}):
            config['training']['finetune']['epochs'] = args.epochs
    
    # Update resume path if provided
    if hasattr(args, 'resume_from') and args.resume_from:
        config['checkpoint']['resume_from'] = args.resume_from
    
    # Update GPU settings

    if hasattr(args, 'gpus') and args.gpus:
        if args.gpus > 1:
            config['distributed']['enabled'] = True
            config['distributed']['world_size'] = args.gpus
        else:
            config['distributed']['enabled'] = False
    
    return config


def create_resolution_aware_paths(config: Dict[str, Any], img_size: int) -> Dict[str, Any]:
    """
    Update checkpoint and logging paths to be resolution-specific.
    
    Args:
        config: Configuration dictionary
        img_size: Image size
        
    Returns:
        Updated configuration with resolution-specific paths
    """
    config = config.copy()
    
    # Update checkpoint directory
    if 'checkpoint' in config and 'save_dir' in config['checkpoint']:
        base_dir = config['checkpoint']['save_dir']
        config['checkpoint']['save_dir'] = os.path.join(base_dir, f"size_{img_size}")
    
    # Update logging directory
    if 'logging' in config and 'log_dir' in config['logging']:
        base_dir = config['logging']['log_dir']
        config['logging']['log_dir'] = os.path.join(base_dir, f"size_{img_size}")
    
    # Update wandb project name if enabled
    if (config.get('logging', {}).get('wandb', {}).get('enabled', False) and 
        'project' in config['logging']['wandb']):
        base_project = config['logging']['wandb']['project']
        config['logging']['wandb']['project'] = f"{base_project}-{img_size}px"
    
    return config


def validate_resolution_config(config: Dict[str, Any], img_size: int) -> None:
    """
    Validate that the configuration is appropriate for the given image size.
    
    Args:
        config: Configuration dictionary
        img_size: Image size
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check that img_size in config matches requested size
    config_img_size = config.get('img_size', [0, 0])
    if config_img_size != [img_size, img_size]:
        raise ValueError(f"Config img_size {config_img_size} doesn't match requested size [{img_size}, {img_size}]")
    
    # Validate memory settings for large images
    if img_size >= 256:
        if not config.get('mixed_precision', False):
            print("WARNING: Mixed precision is disabled for 256x256. This may cause memory issues.")
        
        batch_size = config.get('training', {}).get('finetune', {}).get('batch_size', 8)
        if batch_size > 4:
            print(f"WARNING: Batch size {batch_size} may be too large for 256x256 images.")
    
    # Validate small image settings
    if img_size <= 64:
        if config.get('mixed_precision', False):
            print("INFO: Mixed precision enabled for small images (may not be necessary).")


if __name__ == "__main__":
    # Test the configuration loading
    for size in [64, 128, 256]:
        try:
            config = load_resolution_config(size)
            config = create_resolution_aware_paths(config, size)
            validate_resolution_config(config, size)
            print(f"✓ Config for {size}x{size} loaded successfully")
            print(f"  - Batch size (finetune): {config['training']['finetune']['batch_size']}")
            print(f"  - Model dims: {config['dims']}")
            print(f"  - Checkpoint dir: {config['checkpoint']['save_dir']}")
        except Exception as e:
            print(f"✗ Error loading config for {size}x{size}: {e}")
