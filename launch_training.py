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
    
    required_dirs = ['celeb-df-real', 'celeb-df-fake', 'celeb-df-mask']
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


def run_single_gpu_training(config_path, data_root, mode, use_pretrained, pretrained_path, gpu_id=0, resume=False, checkpoint=None, mask_gt_dir=None):
    """Run single GPU training."""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    cmd = [
        sys.executable, 'train_celeb_df.py',
        '--config', config_path,
        '--mode', mode,
        '--data_root', data_root,
        '--gpus', '1',
        '--use_pretrained', use_pretrained,
        '--pretrained_path', pretrained_path
    ]
    
    if resume and checkpoint:
        cmd.extend(['--resume', '--checkpoint', checkpoint])
    
    if mask_gt_dir:
        cmd.extend(['--mask_gt_dir', mask_gt_dir])
    
    print(f"Running single GPU training on GPU {gpu_id}")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, env=env)
    return result.returncode


def run_multi_gpu_training(config_path, data_root, mode, use_pretrained, pretrained_path, num_gpus, resume=False, checkpoint=None, mask_gt_dir=None):
    """Run multi-GPU training."""
    cmd = [
        sys.executable, 'train_celeb_df.py',
        '--config', config_path,
        '--mode', mode,
        '--data_root', data_root,
        '--gpus', str(num_gpus),
        '--use_pretrained', use_pretrained,
        '--pretrained_path', pretrained_path
    ]
    
    if resume and checkpoint:
        cmd.extend(['--resume', '--checkpoint', checkpoint])
    
    if mask_gt_dir:
        cmd.extend(['--mask_gt_dir', mask_gt_dir])
    
    print(f"Running multi-GPU training on {num_gpus} GPUs")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    return result.returncode


def run_single_gpu_optimized_training(config_path, data_root, mode, use_pretrained, pretrained_path, 
                                    memory_optimization, enable_optimized_modules, gpu_id=0, 
                                    resume=False, checkpoint=None, mask_gt_dir=None):
    """Run single GPU optimized training."""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    cmd = [
        sys.executable, 'train_optimized.py',
        '--config', config_path,
        '--mode', mode,
        '--data_root', data_root,
        '--gpus', '1',
        '--use_pretrained', use_pretrained,
        '--pretrained_path', pretrained_path
    ]
    
    if memory_optimization:
        cmd.append('--memory_optimization')
    
    if enable_optimized_modules:
        cmd.append('--enable_optimized_modules')
    
    if resume and checkpoint:
        cmd.extend(['--resume', '--checkpoint', checkpoint])
    
    if mask_gt_dir:
        cmd.extend(['--mask_gt_dir', mask_gt_dir])
    
    print(f"Running single GPU optimized training on GPU {gpu_id}")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, env=env)
    return result.returncode


def run_multi_gpu_optimized_training(config_path, data_root, mode, use_pretrained, pretrained_path,
                                   memory_optimization, enable_optimized_modules, num_gpus, 
                                   resume=False, checkpoint=None, mask_gt_dir=None):
    """Run multi-GPU optimized training."""
    cmd = [
        sys.executable, 'train_optimized.py',
        '--config', config_path,
        '--mode', mode,
        '--data_root', data_root,
        '--gpus', str(num_gpus),
        '--use_pretrained', use_pretrained,
        '--pretrained_path', pretrained_path
    ]
    
    if memory_optimization:
        cmd.append('--memory_optimization')
    
    if enable_optimized_modules:
        cmd.append('--enable_optimized_modules')
    
    if resume and checkpoint:
        cmd.extend(['--resume', '--checkpoint', checkpoint])
    
    if mask_gt_dir:
        cmd.extend(['--mask_gt_dir', mask_gt_dir])
    
    print(f"Running multi-GPU optimized training on {num_gpus} GPUs")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Launch AuraNet Training on Celeb-DF')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of Celeb-DF dataset')
    parser.add_argument('--mask_gt_dir', type=str, default=None,
                       help='Directory containing ground truth masks for evaluation (e.g., /kaggle/input/ff-mask/)')
    parser.add_argument('--config', type=str, default='config_celeb_df.yaml',
                       help='Configuration file path')
    parser.add_argument('--mode', type=str, choices=['pretrain', 'finetune', 'both'],
                       default='finetune', help='Training mode')
    
    # Model variant selection
    parser.add_argument('--use_optimized', action='store_true',
                       help='Use optimized AuraNet implementation (train_optimized.py)')
    
    # ConvNeXt V2 pretrained weights for spatial stream (different from full model checkpoint)
    parser.add_argument('--use_pretrained', type=str, choices=['y', 'n', 'yes', 'no'],
                       default='n', help='Use ConvNeXt V2 pretrained weights for spatial stream')
    parser.add_argument('--pretrained_path', type=str, default='convnextv2_pico_1k_224_fcmae.pt',
                       help='Path to ConvNeXt V2 pretrained weights file')
    
    # Full model checkpoint for resuming training
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from full AuraNet checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to full AuraNet model checkpoint to resume from')
    
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
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only analyze dataset without training')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show configuration and exit without training')
    
    # Optimized-specific options
    parser.add_argument('--memory_optimization', action='store_true',
                       help='Enable memory optimizations (only with --use_optimized)')
    parser.add_argument('--enable_optimized_modules', action='store_true',
                       help='Use optimized module implementations (only with --use_optimized)')
    
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
    
    if args.resume and args.checkpoint is not None:
        config_updates['checkpoint.resume_from'] = args.checkpoint
    
    # Set data root in config
    config_updates['dataset.data_root'] = args.data_root
    
    # Process ConvNeXt V2 pretrained weights arguments
    use_pretrained = args.use_pretrained.lower() in ['y', 'yes']
    
    # Handle training mode logic for ConvNeXt V2 pretrained weights
    if args.mode == 'both':
        # For 'both' mode, only use pretrained weights in pretraining stage
        config_updates['model.use_pretrained_pretrain'] = use_pretrained
        config_updates['model.use_pretrained_finetune'] = False
    else:
        # For single mode, use pretrained weights if requested
        config_updates['model.use_pretrained_pretrain'] = use_pretrained if args.mode == 'pretrain' else False
        config_updates['model.use_pretrained_finetune'] = use_pretrained if args.mode == 'finetune' else False
    
    config_updates['model.pretrained_path'] = args.pretrained_path
    
    # Set mask GT directory (default to celeb-df-mask if not provided)
    if args.mask_gt_dir:
        config_updates['evaluation.mask_gt_dir'] = args.mask_gt_dir
    else:
        # Default to the standard celeb-df-mask directory
        config_updates['evaluation.mask_gt_dir'] = 'celeb-df-mask'
    
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
                if args.use_optimized:
                    ret_code = run_multi_gpu_optimized_training(temp_config_path, args.data_root, 'pretrain', 
                                                              args.use_pretrained, args.pretrained_path,
                                                              args.memory_optimization, args.enable_optimized_modules,
                                                              effective_num_gpus, args.resume, args.checkpoint, args.mask_gt_dir)
                else:
                    ret_code = run_multi_gpu_training(temp_config_path, args.data_root, 'pretrain', 
                                                    args.use_pretrained, args.pretrained_path, 
                                                    effective_num_gpus, args.resume, args.checkpoint, args.mask_gt_dir)
            else:
                if args.use_optimized:
                    ret_code = run_single_gpu_optimized_training(temp_config_path, args.data_root, 'pretrain',
                                                               args.use_pretrained, args.pretrained_path,
                                                               args.memory_optimization, args.enable_optimized_modules,
                                                               0, args.resume, args.checkpoint, args.mask_gt_dir)
                else:
                    ret_code = run_single_gpu_training(temp_config_path, args.data_root, 'pretrain',
                                                     args.use_pretrained, args.pretrained_path,
                                                     0, args.resume, args.checkpoint, args.mask_gt_dir)
            
            if ret_code != 0:
                print("Pre-training failed!")
                return ret_code
            
            print("\nStarting fine-tuning phase...")
            if effective_num_gpus > 1:
                if args.use_optimized:
                    ret_code = run_multi_gpu_optimized_training(temp_config_path, args.data_root, 'finetune',
                                                              'n', args.pretrained_path,  # Don't use ConvNeXt pretrained for finetune
                                                              args.memory_optimization, args.enable_optimized_modules,
                                                              effective_num_gpus, False, None, args.mask_gt_dir)
                else:
                    ret_code = run_multi_gpu_training(temp_config_path, args.data_root, 'finetune',
                                                    'n', args.pretrained_path,  # Don't use ConvNeXt pretrained for finetune
                                                    effective_num_gpus, False, None, args.mask_gt_dir)
            else:
                if args.use_optimized:
                    ret_code = run_single_gpu_optimized_training(temp_config_path, args.data_root, 'finetune',
                                                               'n', args.pretrained_path,  # Don't use ConvNeXt pretrained for finetune
                                                               args.memory_optimization, args.enable_optimized_modules,
                                                               0, False, None, args.mask_gt_dir)
                else:
                    ret_code = run_single_gpu_training(temp_config_path, args.data_root, 'finetune',
                                                     'n', args.pretrained_path,  # Don't use ConvNeXt pretrained for finetune
                                                     0, False, None, args.mask_gt_dir)
        else:
            # Run single mode
            if effective_num_gpus > 1:
                if args.use_optimized:
                    ret_code = run_multi_gpu_optimized_training(temp_config_path, args.data_root, args.mode,
                                                              args.use_pretrained, args.pretrained_path,
                                                              args.memory_optimization, args.enable_optimized_modules,
                                                              effective_num_gpus, args.resume, args.checkpoint, args.mask_gt_dir)
                else:
                    ret_code = run_multi_gpu_training(temp_config_path, args.data_root, args.mode,
                                                    args.use_pretrained, args.pretrained_path,
                                                    effective_num_gpus, args.resume, args.checkpoint, args.mask_gt_dir)
            else:
                if args.use_optimized:
                    ret_code = run_single_gpu_optimized_training(temp_config_path, args.data_root, args.mode,
                                                               args.use_pretrained, args.pretrained_path,
                                                               args.memory_optimization, args.enable_optimized_modules,
                                                               0, args.resume, args.checkpoint, args.mask_gt_dir)
                else:
                    ret_code = run_single_gpu_training(temp_config_path, args.data_root, args.mode,
                                                     args.use_pretrained, args.pretrained_path,
                                                     0, args.resume, args.checkpoint, args.mask_gt_dir)
        
        return ret_code
    
    finally:
        # Cleanup temporary config
        if temp_config_path != args.config and os.path.exists(temp_config_path):
            os.remove(temp_config_path)


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
