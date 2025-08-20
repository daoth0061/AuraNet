#!/usr/bin/env python3
"""
Training Script Launcher for AuraNet on Celeb-DF
Handles multi-GPU setup and provides easy-to-use interface
"""

import os
import sys
import argparse
import subprocess
import yaml
from pathlib import Path
import torch


def check_gpu_availability():
    """Check GPU availability and return device info."""
    if not torch.cuda.is_available():
        return 0, []
    
    num_gpus = torch.cuda.device_count()
    gpu_info = []
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        gpu_info.append({
            'id': i,
            'name': props.name,
            'memory': props.total_memory // (1024**3),  # GB
            'compute_capability': f"{props.major}.{props.minor}"
        })
    
    return num_gpus, gpu_info


def validate_dataset(data_root):
    """Validate dataset structure."""
    data_root = Path(data_root)
    
    required_dirs = ['Celeb_DF_Real', 'Celeb_DF_Fake', 'Celeb_DF_Mask']
    missing_dirs = []
    
    for dir_name in required_dirs:
        if not (data_root / dir_name).exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"ERROR: Missing required directories: {missing_dirs}")
        print(f"Expected structure: {data_root}/")
        for dir_name in required_dirs:
            print(f"  └── {dir_name}/")
            print(f"      └── video_id_folders/")
            print(f"          └── cropped_face_X_frame_Y.jpg")
        return False
    
    return True


def update_config_for_training(config_path, updates):
    """Update configuration file with training-specific parameters."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply updates
    for key_path, value in updates.items():
        keys = key_path.split('.')
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    # Save updated config
    temp_config_path = config_path.replace('.yaml', '_temp.yaml')
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    return temp_config_path


def run_single_gpu_training(config_path, data_root, mode, gpu_id=0):
    """Run single GPU training."""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    cmd = [
        sys.executable, 'train_celeb_df.py',
        '--config', config_path,
        '--mode', mode,
        '--data_root', data_root,
        '--gpus', '1'
    ]
    
    print(f"Running single GPU training on GPU {gpu_id}")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, env=env)
    return result.returncode


def run_multi_gpu_training(config_path, data_root, mode, num_gpus):
    """Run multi-GPU training."""
    cmd = [
        sys.executable, 'train_celeb_df.py',
        '--config', config_path,
        '--mode', mode,
        '--data_root', data_root,
        '--gpus', str(num_gpus)
    ]
    
    print(f"Running multi-GPU training on {num_gpus} GPUs")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Launch AuraNet Training on Celeb-DF')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of Celeb-DF dataset')
    parser.add_argument('--config', type=str, default='config_celeb_df.yaml',
                       help='Configuration file path')
    parser.add_argument('--mode', type=str, choices=['pretrain', 'finetune', 'both'],
                       default='finetune', help='Training mode')
    parser.add_argument('--gpus', type=int, default=None,
                       help='Number of GPUs to use (auto-detect if not specified)')
    parser.add_argument('--gpu_ids', type=str, default=None,
                       help='Specific GPU IDs to use (comma-separated, e.g., "0,1,2")')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Override learning rate')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Resume training from checkpoint')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only analyze dataset without training')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show configuration and exit without training')
    
    args = parser.parse_args()
    
    # Validate dataset
    if not validate_dataset(args.data_root):
        return 1
    
    # Check GPU availability
    num_available_gpus, gpu_info = check_gpu_availability()
    
    if num_available_gpus == 0:
        print("ERROR: No GPUs available. AuraNet requires CUDA-enabled GPUs.")
        return 1
    
    print(f"Available GPUs ({num_available_gpus}):")
    for gpu in gpu_info:
        print(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['memory']}GB)")
    print()
    
    # Determine GPUs to use
    if args.gpu_ids is not None:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        if max(gpu_ids) >= num_available_gpus:
            print(f"ERROR: GPU ID {max(gpu_ids)} not available (max: {num_available_gpus-1})")
            return 1
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        effective_num_gpus = len(gpu_ids)
    elif args.gpus is not None:
        effective_num_gpus = min(args.gpus, num_available_gpus)
    else:
        effective_num_gpus = num_available_gpus
    
    # Analyze dataset if requested
    if args.analyze_only:
        print("Analyzing dataset...")
        analyzer_cmd = [
            sys.executable, 'analyze_celeb_df.py',
            '--data_root', args.data_root,
            '--config', args.config
        ]
        subprocess.run(analyzer_cmd)
        return 0
    
    # Prepare configuration updates
    config_updates = {}
    
    if args.batch_size is not None:
        if args.mode == 'pretrain' or args.mode == 'both':
            config_updates['training.pretrain.batch_size'] = args.batch_size
        if args.mode == 'finetune' or args.mode == 'both':
            config_updates['training.finetune.batch_size'] = args.batch_size
    
    if args.learning_rate is not None:
        if args.mode == 'pretrain' or args.mode == 'both':
            config_updates['training.pretrain.learning_rate'] = args.learning_rate
        if args.mode == 'finetune' or args.mode == 'both':
            config_updates['training.finetune.encoder_lr'] = args.learning_rate * 0.1
            config_updates['training.finetune.head_lr'] = args.learning_rate
    
    if args.epochs is not None:
        if args.mode == 'pretrain' or args.mode == 'both':
            config_updates['training.pretrain.epochs'] = args.epochs
        if args.mode == 'finetune' or args.mode == 'both':
            config_updates['training.finetune.epochs'] = args.epochs
    
    if args.resume_from is not None:
        config_updates['checkpoint.resume_from'] = args.resume_from
    
    # Set data root in config
    config_updates['dataset.data_root'] = args.data_root
    
    # Update distributed settings
    if effective_num_gpus > 1:
        config_updates['distributed.enabled'] = True
        config_updates['distributed.world_size'] = effective_num_gpus
    else:
        config_updates['distributed.enabled'] = False
    
    # Create temporary config with updates
    if config_updates:
        temp_config_path = update_config_for_training(args.config, config_updates)
    else:
        temp_config_path = args.config
    
    # Show configuration
    print(f"Training Configuration:")
    print(f"  Mode: {args.mode}")
    print(f"  GPUs: {effective_num_gpus}")
    print(f"  Data root: {args.data_root}")
    print(f"  Config: {temp_config_path}")
    
    if config_updates:
        print(f"  Configuration overrides:")
        for key, value in config_updates.items():
            print(f"    {key}: {value}")
    print()
    
    if args.dry_run:
        print("Dry run mode - exiting without training")
        if temp_config_path != args.config:
            os.remove(temp_config_path)
        return 0
    
    # Run training
    try:
        if args.mode == 'both':
            # Run pre-training first, then fine-tuning
            print("Starting pre-training phase...")
            if effective_num_gpus > 1:
                ret_code = run_multi_gpu_training(temp_config_path, args.data_root, 'pretrain', effective_num_gpus)
            else:
                ret_code = run_single_gpu_training(temp_config_path, args.data_root, 'pretrain')
            
            if ret_code != 0:
                print("Pre-training failed!")
                return ret_code
            
            print("\nStarting fine-tuning phase...")
            if effective_num_gpus > 1:
                ret_code = run_multi_gpu_training(temp_config_path, args.data_root, 'finetune', effective_num_gpus)
            else:
                ret_code = run_single_gpu_training(temp_config_path, args.data_root, 'finetune')
        else:
            # Run single mode
            if effective_num_gpus > 1:
                ret_code = run_multi_gpu_training(temp_config_path, args.data_root, args.mode, effective_num_gpus)
            else:
                ret_code = run_single_gpu_training(temp_config_path, args.data_root, args.mode)
        
        return ret_code
    
    finally:
        # Cleanup temporary config
        if temp_config_path != args.config and os.path.exists(temp_config_path):
            os.remove(temp_config_path)


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
