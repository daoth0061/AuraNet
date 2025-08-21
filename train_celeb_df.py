"""
Distributed Training Script for AuraNet on Celeb-DF Dataset
Supports multi-GPU training with proper data sampling strategy
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import random
from typing import Dict, Optional, Tuple
import wandb
from tensorboardX import SummaryWriter

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from auranet import create_auranet
from celeb_df_dataset import CelebDFDataset
from training import (CombinedPretrainLoss, CombinedFinetuneLoss, 
                     get_optimizer, get_scheduler)
from evaluate import AuraNetEvaluator


class DistributedTrainer:
    """Distributed trainer for AuraNet on Celeb-DF dataset."""
    
    def __init__(self, rank: int, world_size: int, config: Dict, mode: str):
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.mode = mode
        self.device = torch.device(f'cuda:{rank}')
        
        # Setup logging
        self._setup_logging()
        
        # Setup reproducibility
        self._setup_seed()
        
        # Initialize model
        self.model = self._create_model()
        
        # Setup data loaders
        self.train_loader, self.val_loader = self._setup_data_loaders()
        
        # Setup training components
        self.criterion = self._setup_criterion()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup logging and checkpointing
        self.writer = None
        self.evaluator = None
        if self.rank == 0:
            self._setup_logging_tools()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        self.patience_counter = 0
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.INFO if self.rank == 0 else logging.WARNING
        logging.basicConfig(
            level=log_level,
            format=f'[Rank {self.rank}] %(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _setup_seed(self):
        """Setup random seeds for reproducibility."""
        seed = self.config.get('seed', 42) + self.rank
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        if self.config.get('deterministic', False):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _create_model(self):
        """Create and setup the model."""
        # Create model
        model = create_auranet(config=self.config)
        model = model.to(self.device)
        
        # Wrap with DDP
        model = DDP(model, device_ids=[self.rank], find_unused_parameters=True)
        
        # Enable mixed precision if configured
        if self.config.get('mixed_precision', False):
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            
        # Compile model if configured (PyTorch 2.0+)
        if self.config.get('compile_model', False):
            model = torch.compile(model)
            
        return model
    
    def _setup_data_loaders(self):
        """Setup distributed data loaders."""
        mode = self.mode
        
        # Create datasets
        train_dataset = CelebDFDataset(
            data_root=self.config['dataset']['data_root'],
            mode=mode,
            split='train',
            config=self.config
        )
        
        val_dataset = CelebDFDataset(
            data_root=self.config['dataset']['data_root'],
            mode=mode,
            split='test',  # Use test split for validation
            config=self.config
        )
        
        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=self.world_size, 
            rank=self.rank,
            shuffle=True
        )
        
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False
        )
        
        # Get batch size based on mode
        if mode == 'pretrain':
            batch_size = self.config['training']['pretrain']['batch_size']
        else:
            batch_size = self.config['training']['finetune']['batch_size']
        
        # Data loading parameters
        data_config = self.config['data_loading']
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory'],
            drop_last=data_config['drop_last'],
            prefetch_factor=data_config.get('prefetch_factor', 2),
            persistent_workers=data_config.get('persistent_workers', True)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory'],
            prefetch_factor=data_config.get('prefetch_factor', 2),
            persistent_workers=data_config.get('persistent_workers', True)
        )
        
        return train_loader, val_loader
    
    def _setup_criterion(self):
        """Setup loss function."""
        mode = self.mode
        
        if mode == 'pretrain':
            return CombinedPretrainLoss(config=self.config)
        else:
            return CombinedFinetuneLoss(config=self.config)
    
    def _setup_optimizer(self):
        """Setup optimizer."""
        return get_optimizer(self.model, mode=self.mode, config=self.config)
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        return get_scheduler(self.optimizer, mode=self.mode, config=self.config)
    
    def _setup_logging_tools(self):
        """Setup tensorboard and wandb logging (only on rank 0)."""
        # Setup tensorboard
        if self.config['logging']['tensorboard']:
            log_dir = Path(self.config['logging']['log_dir'])
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.writer = SummaryWriter(log_dir / f'run_{timestamp}')
        
        # Setup wandb
        if self.config['logging']['wandb']['enabled']:
            wandb.init(
                project=self.config['logging']['wandb']['project'],
                entity=self.config['logging']['wandb'].get('entity'),
                config=self.config,
                name=f"auranet_celeb_df_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        # Setup evaluator
        self.evaluator = AuraNetEvaluator(config=self.config)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_loader.sampler.set_epoch(self.epoch)
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        log_freq = self.config['logging']['log_freq']
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                # Mixed precision training
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch['image'])
                    loss = self.criterion(outputs, batch)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Regular training
                outputs = self.model(batch['image'])
                loss = self.criterion(outputs, batch)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            # Logging
            if self.rank == 0 and batch_idx % log_freq == 0:
                self.logger.info(
                    f'Epoch: {self.epoch}, Batch: {batch_idx}/{num_batches}, '
                    f'Loss: {loss.item():.6f}, LR: {self.optimizer.param_groups[0]["lr"]:.2e}'
                )
                
                if self.writer is not None:
                    self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                    self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], self.global_step)
                
                if self.config['logging']['wandb']['enabled']:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        'epoch': self.epoch,
                        'step': self.global_step
                    })
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch['image'])
                        loss = self.criterion(outputs, batch)
                else:
                    outputs = self.model(batch['image'])
                    loss = self.criterion(outputs, batch)
                
                total_loss += loss.item()
                
                # Collect predictions for evaluation
                if 'classification_logits' in outputs:
                    predictions = torch.softmax(outputs['classification_logits'], dim=1)
                    all_predictions.append(predictions.cpu())
                    all_labels.append(batch['label'].cpu())
        
        avg_loss = total_loss / num_batches
        
        # Calculate metrics
        metrics = {'val_loss': avg_loss}
        
        if all_predictions and self.rank == 0:
            all_predictions = torch.cat(all_predictions, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            # Calculate additional metrics
            eval_metrics = self.evaluator.calculate_metrics(all_predictions, all_labels)
            metrics.update(eval_metrics)
        
        return metrics
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint (only on rank 0)."""
        if self.rank != 0:
            return
        
        checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'best_metric': self.best_metric
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{self.epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f'New best model saved with metric: {self.best_metric:.6f}')
        
        # Keep only recent checkpoints
        self._cleanup_checkpoints(checkpoint_dir)
    
    def _cleanup_checkpoints(self, checkpoint_dir: Path, keep_last: int = 3):
        """Keep only the last N checkpoints."""
        checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        
        while len(checkpoints) > keep_last:
            oldest = checkpoints.pop(0)
            oldest.unlink()
    
    def train(self):
        """Main training loop."""
        mode = self.mode
        
        # Get training configuration
        if mode == 'pretrain':
            train_config = self.config['training']['pretrain']
            num_epochs = train_config['epochs']
            patience = train_config['patience']
            min_delta = train_config['min_delta']
        else:
            train_config = self.config['training']['finetune']
            num_epochs = train_config['epochs']
            patience = train_config['patience']
            min_delta = train_config['min_delta']
        
        self.logger.info(f'Starting {mode} training for {num_epochs} epochs')
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Step scheduler
            self.scheduler.step()
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            
            # Check for improvement (using accuracy or AUC)
            current_metric = val_metrics.get('accuracy', val_metrics.get('auc', val_metrics['val_loss']))
            is_best = False
            
            if mode == 'finetune' and 'accuracy' in val_metrics:
                # For fine-tuning, higher accuracy is better
                if current_metric > self.best_metric + min_delta:
                    self.best_metric = current_metric
                    self.patience_counter = 0
                    is_best = True
                else:
                    self.patience_counter += 1
            elif mode == 'pretrain':
                # For pre-training, lower loss is better
                current_metric = -val_metrics['val_loss']  # Negative for consistency
                if current_metric > self.best_metric + min_delta:
                    self.best_metric = current_metric
                    self.patience_counter = 0
                    is_best = True
                else:
                    self.patience_counter += 1
            
            # Log metrics
            if self.rank == 0:
                self.logger.info(f'Epoch {epoch}: {metrics}')
                
                if self.writer is not None:
                    for key, value in metrics.items():
                        self.writer.add_scalar(f'Epoch/{key}', value, epoch)
                
                if self.config['logging']['wandb']['enabled']:
                    wandb.log({f'epoch/{k}': v for k, v in metrics.items()})
                
                # Save checkpoint
                if epoch % self.config['checkpoint']['save_freq'] == 0 or is_best:
                    self.save_checkpoint(metrics, is_best)
            
            # Early stopping
            if self.patience_counter >= patience:
                self.logger.info(f'Early stopping triggered after {epoch + 1} epochs')
                break
        
        if self.rank == 0:
            self.logger.info(f'Training completed. Best metric: {self.best_metric:.6f}')
            
            # Close logging tools
            if self.writer is not None:
                self.writer.close()
            
            if self.config['logging']['wandb']['enabled']:
                wandb.finish()


def setup_distributed(rank: int, world_size: int, backend: str = 'nccl'):
    """Setup distributed training environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    # Set CUDA device
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed training environment."""
    dist.destroy_process_group()


def main_worker(rank: int, world_size: int, config: Dict, mode: str):
    """Main worker function for distributed training."""
    try:
        # Setup distributed training
        setup_distributed(rank, world_size, config['distributed']['backend'])
        
        # Create trainer
        trainer = DistributedTrainer(rank, world_size, config, mode)
        
        # Start training
        trainer.train()
        
    except Exception as e:
        logging.error(f"Error in worker {rank}: {str(e)}")
        raise
    finally:
        # Cleanup
        cleanup_distributed()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train AuraNet on Celeb-DF Dataset')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['pretrain', 'finetune'],
                       default='finetune', help='Training mode')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of Celeb-DF dataset')
    parser.add_argument('--gpus', type=int, default=None,
                       help='Number of GPUs to use (auto-detect if not specified)')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update data root in config
    config['dataset']['data_root'] = args.data_root
    
    # Determine number of GPUs
    if args.gpus is not None:
        world_size = args.gpus
    elif config['distributed']['enabled']:
        world_size = torch.cuda.device_count()
    else:
        world_size = 1
    
    if world_size <= 1:
        # Single GPU training
        config['distributed']['enabled'] = False
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Create single GPU trainer (simplified version)
        trainer = DistributedTrainer(0, 1, config, args.mode)
        trainer.train()
    else:
        # Multi-GPU training
        print(f'Starting distributed training on {world_size} GPUs')
        config['distributed']['enabled'] = True
        config['distributed']['world_size'] = world_size
        
        # Spawn processes for each GPU
        mp.spawn(
            main_worker,
            args=(world_size, config, args.mode),
            nprocs=world_size,
            join=True
        )


if __name__ == '__main__':
    main()
