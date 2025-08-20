#!/usr/bin/env python3
"""
Example training script for AuraNet
Demonstrates complete training pipeline
"""

import os
import sys
import argparse
import torch
import json
import yaml
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from auranet import create_auranet
from train import AuraNetTrainer
from data_loader import create_sample_annotations, create_data_loaders


def setup_sample_dataset(data_root):
    """Create sample dataset structure for testing."""
    print(f"Setting up sample dataset in {data_root}")
    
    # Create directories
    os.makedirs(os.path.join(data_root, 'real'), exist_ok=True)
    os.makedirs(os.path.join(data_root, 'fake'), exist_ok=True)
    
    # Note: In practice, you would copy actual images here
    print("Note: Please add actual images to the real/ and fake/ directories")
    print("This script assumes you have images in the specified directories")
    
    # Create sample annotations
    train_annotations_path = os.path.join(data_root, 'train_annotations.json')
    val_annotations_path = os.path.join(data_root, 'val_annotations.json')
    
    # Create minimal sample annotations for demonstration
    sample_annotations = [
        {
            "image": "real/sample1.jpg",
            "label": 0,
            "metadata": {"manipulation_type": "none", "source": "real"}
        },
        {
            "image": "fake/sample1.jpg", 
            "label": 1,
            "metadata": {"manipulation_type": "deepfake", "source": "fake"}
        }
    ]
    
    with open(train_annotations_path, 'w') as f:
        json.dump(sample_annotations, f, indent=2)
    
    with open(val_annotations_path, 'w') as f:
        json.dump(sample_annotations, f, indent=2)
    
    print(f"Sample annotations created:")
    print(f"  - {train_annotations_path}")
    print(f"  - {val_annotations_path}")
    
    return train_annotations_path, val_annotations_path


def main():
    parser = argparse.ArgumentParser(description='AuraNet Training Example')
    parser.add_argument('--data_root', type=str, default='./sample_data',
                       help='Root directory for training data')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--mode', type=str, choices=['pretrain', 'finetune', 'both'],
                       default='both', help='Training mode')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size (use small value for demo)')
    parser.add_argument('--epochs_pretrain', type=int, default=2,
                       help='Number of pre-training epochs (small for demo)')
    parser.add_argument('--epochs_finetune', type=int, default=2,
                       help='Number of fine-tuning epochs (small for demo)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')
    parser.add_argument('--setup_data', action='store_true',
                       help='Setup sample dataset structure')
    parser.add_argument('--demo_mode', action='store_true',
                       help='Run in demo mode with minimal epochs')
    
    args = parser.parse_args()
    
    print("AuraNet Training Example")
    print("=" * 50)
    print(f"Device: {args.device}")
    print(f"Data root: {args.data_root}")
    print(f"Config: {args.config}")
    print(f"Save dir: {args.save_dir}")
    print(f"Mode: {args.mode}")
    
    # Setup dataset if requested
    if args.setup_data:
        train_annotations, val_annotations = setup_sample_dataset(args.data_root)
    else:
        train_annotations = os.path.join(args.data_root, 'train_annotations.json')
        val_annotations = os.path.join(args.data_root, 'val_annotations.json')
    
    # Check if annotation files exist
    if not os.path.exists(train_annotations) or not os.path.exists(val_annotations):
        print(f"Error: Annotation files not found!")
        print(f"  Expected: {train_annotations}")
        print(f"  Expected: {val_annotations}")
        print("  Use --setup_data to create sample structure")
        return
    
    # Adjust epochs for demo mode
    if args.demo_mode:
        epochs_pretrain = min(args.epochs_pretrain, 1)
        epochs_finetune = min(args.epochs_finetune, 1)
        batch_size = min(args.batch_size, 2)
        print(f"Demo mode: Using {epochs_pretrain} pretrain epochs, {epochs_finetune} finetune epochs, batch size {batch_size}")
    else:
        epochs_pretrain = args.epochs_pretrain
        epochs_finetune = args.epochs_finetune
        batch_size = args.batch_size
    
    try:
        # Initialize trainer
        print("\nInitializing trainer...")
        trainer = AuraNetTrainer(
            config_path=args.config,
            save_dir=args.save_dir,
            device=args.device
        )
        
        # Pre-training stage
        if args.mode in ['pretrain', 'both']:
            print(f"\n{'='*20} PRE-TRAINING {'='*20}")
            
            # Create pre-training data loaders
            print("Creating pre-training data loaders...")
            train_loader, val_loader = create_data_loaders(
                train_data_root=args.data_root,
                val_data_root=args.data_root,
                train_annotations=train_annotations,
                val_annotations=val_annotations,
                batch_size=batch_size,
                num_workers=2,  # Reduced for demo
                mode='pretrain'
            )
            
            print(f"Train batches: {len(train_loader)}")
            print(f"Val batches: {len(val_loader)}")
            
            # Run pre-training
            trainer.pretrain(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=epochs_pretrain,
                save_every=1
            )
            
            pretrained_checkpoint = os.path.join(args.save_dir, 'pretrain_best.pth')
        else:
            pretrained_checkpoint = None
        
        # Fine-tuning stage
        if args.mode in ['finetune', 'both']:
            print(f"\n{'='*20} FINE-TUNING {'='*20}")
            
            # Create fine-tuning data loaders
            print("Creating fine-tuning data loaders...")
            train_loader, val_loader = create_data_loaders(
                train_data_root=args.data_root,
                val_data_root=args.data_root,
                train_annotations=train_annotations,
                val_annotations=val_annotations,
                batch_size=batch_size,
                num_workers=2,  # Reduced for demo
                mode='finetune'
            )
            
            print(f"Train batches: {len(train_loader)}")
            print(f"Val batches: {len(val_loader)}")
            
            # Run fine-tuning
            trainer.finetune(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=epochs_finetune,
                pretrained_checkpoint=pretrained_checkpoint,
                save_every=1
            )
        
        print(f"\n{'='*20} TRAINING COMPLETED {'='*20}")
        print(f"Checkpoints saved in: {args.save_dir}")
        print(f"TensorBoard logs: {os.path.join(args.save_dir, 'logs')}")
        print("\nTo view training logs, run:")
        print(f"  tensorboard --logdir {os.path.join(args.save_dir, 'logs')}")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
