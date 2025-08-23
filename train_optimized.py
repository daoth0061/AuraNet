"""
AuraNet Optimized Training Script
Uses the optimized implementations for faster training
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import yaml
import time
import sys
from pathlib import Path
from datetime import datetime

# Import logging utilities
from src.logging_utils import setup_logging, get_logger

# Set up logging with file output
logger = setup_logging(log_dir='logs', level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description='AuraNet Optimized Training')
    
    # Basic arguments
    parser.add_argument('--config', type=str, default='config_celeb_df_memory_optimized.yaml',
                        help='Path to the config file')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Root directory for dataset (overrides config)')
    parser.add_argument('--mode', type=str, choices=['pretrain', 'finetune', 'both'], default='pretrain',
                        help='Training mode')
    parser.add_argument('--use_pretrained', type=str, choices=['yes', 'no'], default='yes',
                        help='Whether to use pretrained weights')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Path to pretrained model')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    
    # Performance optimization
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--memory_optimization', action='store_true',
                        help='Enable memory optimizations')
    parser.add_argument('--enable_optimized_modules', action='store_true',
                        help='Use optimized module implementations')
    
    # Kaggle-specific arguments
    parser.add_argument('--kaggle', action='store_true',
                        help='Running in Kaggle environment')
    parser.add_argument('--kaggle_working_dir', type=str, default=None,
                        help='Working directory for Kaggle')
    
    # Continue training
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    return args

def is_kaggle_environment() -> bool:
    """Check if running in Kaggle environment."""
    return os.path.exists('/kaggle/working')

def setup_kaggle_environment(working_dir: str) -> None:
    """Setup environment for Kaggle."""
    logger.info(f"Setting up Kaggle environment in {working_dir}")
    # Add current directory to path
    sys.path.append(working_dir)
    # Add src directory to path
    sys.path.append(os.path.join(working_dir, 'src'))

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_distributed(args, config):
    """Set up distributed training."""
    if args.gpus > 1:
        if not config['distributed']['enabled']:
            logger.warning("Distributed training not enabled in config, but multiple GPUs requested")
            config['distributed']['enabled'] = True
            
        # Initialize process group
        if torch.cuda.is_available():
            logger.info(f"Initializing distributed training with {args.gpus} GPUs")
            torch.cuda.set_device(args.local_rank)
            dist.init_process_group(backend=config['distributed']['backend'])
            config['distributed']['world_size'] = dist.get_world_size()
            config['distributed']['rank'] = dist.get_rank()
            logger.info(f"Process group initialized: rank {config['distributed']['rank']} of {config['distributed']['world_size']}")
        else:
            logger.warning("CUDA not available, falling back to CPU")
            args.gpus = 0
            config['distributed']['enabled'] = False

def create_model(config, args):
    """Create the optimized AuraNet model."""
    try:
        # Try to import the optimized implementation first
        from src.auranet_optimized import create_optimized_auranet
        logger.info("Using optimized AuraNet implementation")
        
        # Create model
        model = create_optimized_auranet(
            config=config,
            use_pretrained=(args.use_pretrained == 'yes'),
            pretrained_path=args.pretrained_path
        )
        
        return model
    except ImportError:
        # Fall back to standard implementation
        logger.warning("Optimized implementation not found, falling back to standard AuraNet")
        from src.auranet import create_auranet
        
        # Create model
        model = create_auranet(
            config=config,
            use_pretrained=(args.use_pretrained == 'yes'),
            pretrained_path=args.pretrained_path
        )
        
        return model

def create_data_loaders(config, args):
    """Create data loaders for training and validation."""
    try:
        from src.data_loader import create_dataloaders
        
        # Override batch size if specified
        if args.batch_size is not None:
            logger.info(f"Overriding batch size to {args.batch_size}")
            if args.mode == 'pretrain' or args.mode == 'both':
                config['training']['pretrain']['batch_size'] = args.batch_size
            if args.mode == 'finetune' or args.mode == 'both':
                config['training']['finetune']['batch_size'] = args.batch_size
        
        # Override data root if specified
        if args.data_root is not None:
            logger.info(f"Overriding data root to {args.data_root}")
            config['dataset']['data_root'] = args.data_root
        
        # Create data loaders
        train_loader, val_loader = create_dataloaders(config)
        
        return train_loader, val_loader
    except ImportError:
        logger.error("Failed to import data loader, make sure src/data_loader.py exists")
        sys.exit(1)

def train_model(model, train_loader, val_loader, config, args):
    """Train the model using the specified mode."""
    try:
        from src.training import AuraNetTrainer
        
        # Create trainer
        trainer = AuraNetTrainer(
            model=model,
            config=config,
            distributed=(args.gpus > 1),
            rank=config['distributed'].get('rank', 0) if args.gpus > 1 else 0
        )
        
        # Start time
        start_time = time.time()
        
        # Resume training if requested
        if args.resume and args.checkpoint:
            logger.info(f"Resuming training from checkpoint: {args.checkpoint}")
            trainer.load_checkpoint(args.checkpoint)
        
        # Train based on mode
        if args.mode == 'pretrain' or args.mode == 'both':
            logger.info("Starting pre-training")
            trainer.pretrain(train_loader, val_loader)
            
        if args.mode == 'finetune' or args.mode == 'both':
            logger.info("Starting fine-tuning")
            trainer.finetune(train_loader, val_loader)
        
        # End time
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
        
        return trainer
    except ImportError:
        logger.error("Failed to import trainer, make sure src/training.py exists")
        sys.exit(1)

def optimize_config(config, args):
    """Apply optimization settings to config."""
    if args.memory_optimization:
        logger.info("Applying memory optimizations to config")
        
        # Ensure memory_optimization section exists
        if 'memory_optimization' not in config:
            config['memory_optimization'] = {}
            
        # Apply optimizations
        config['memory_optimization']['gradient_checkpointing'] = True
        config['memory_optimization']['flash_attention'] = True
        config['memory_optimization']['periodic_cache_clearing'] = True
        config['memory_optimization']['enable_channels_last'] = True
        config['memory_optimization']['enable_cudnn_autotuner'] = True
        
    if args.enable_optimized_modules:
        logger.info("Enabling optimized module implementations")
        config['memory_optimization']['use_optimized_modules'] = True
        
    # Enable mixed precision by default
    config['mixed_precision'] = True
    
    return config

def main():
    # Parse arguments
    args = parse_args()
    
    # Check for Kaggle environment
    if args.kaggle or is_kaggle_environment():
        kaggle_working_dir = args.kaggle_working_dir if args.kaggle_working_dir else '/kaggle/working/AuraNet'
        setup_kaggle_environment(kaggle_working_dir)
        logger.info(f"Running in Kaggle environment with working directory: {kaggle_working_dir}")
        
        # Update paths for Kaggle
        if not os.path.isabs(args.config):
            args.config = os.path.join(kaggle_working_dir, args.config)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Apply optimizations
    config = optimize_config(config, args)
    
    # Set up distributed training
    setup_distributed(args, config)
    
    # Create model
    model = create_model(config, args)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config, args)
    
    # Train model
    trainer = train_model(model, train_loader, val_loader, config, args)
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    try:
        # Auto-detect Kaggle environment
        if is_kaggle_environment():
            logger.info("Detected Kaggle environment!")
            
        main()
    except Exception as e:
        if is_kaggle_environment():
            logger.error("=" * 80)
            logger.error("ERROR IN KAGGLE ENVIRONMENT:")
            logger.error(f"Error: {str(e)}")
            logger.error("Debugging tips:")
            logger.error("1. Check if --kaggle flag is set")
            logger.error("2. Verify the correct paths in your command")
            logger.error("3. Make sure all required files are in /kaggle/working/AuraNet/")
            logger.error("=" * 80)
        
        if "pretrain" in str(e).lower():
            logger.error("Pre-training failed!")
        elif "finetune" in str(e).lower():
            logger.error("Fine-tuning failed!")
        else:
            logger.error(f"Training failed with error: {str(e)}")
        
        # Print full traceback
        import traceback
        traceback.print_exc()
        
        sys.exit(1)
