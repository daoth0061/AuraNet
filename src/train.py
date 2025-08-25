"""
Training scripts for AuraNet
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import os
import sys

# Add the project root directory to Python path for absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
from src.auranet import create_auranet
from src.training import CombinedPretrainLoss, CombinedFinetuneLoss, get_optimizer, get_scheduler, create_random_mask
from src.data_loader import create_data_loaders


class AuraNetTrainer:
    """Trainer class for AuraNet model."""
    
    def __init__(self, config_path, save_dir, device='cuda'):
        """
        Args:
            config_path: str, path to configuration file
            save_dir: str, directory to save model checkpoints and logs
            device: str, device to use for training
        """
        self.config_path = config_path
        self.save_dir = save_dir
        self.device = device
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize model
        self.model = create_auranet(config_path=config_path)
        self.model.to(device)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {'train_loss': [], 'val_loss': []}
    
    def pretrain(self, train_loader, val_loader, num_epochs=100, save_every=10):
        """
        Pre-training stage with self-supervised learning.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: int, number of training epochs
            save_every: int, save checkpoint every N epochs
        """
        print("Starting pre-training...")
        
        # Set model to pre-training mode
        self.model.train()
        
        # Initialize loss function and optimizer
        criterion = CombinedPretrainLoss()
        optimizer = get_optimizer(self.model, mode='pretrain')
        scheduler = get_scheduler(optimizer, num_epochs)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_loss = self._train_pretrain_epoch(train_loader, criterion, optimizer)
            
            # Validation
            val_loss = self._validate_pretrain_epoch(val_loader, criterion)
            
            # Update learning rate
            scheduler.step()
            
            # Log metrics
            self.writer.add_scalar('PreTrain/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('PreTrain/Val_Loss', val_loss, epoch)
            self.writer.add_scalar('PreTrain/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0 or val_loss < self.best_val_loss:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                self._save_checkpoint(epoch, 'pretrain', is_best=(val_loss < self.best_val_loss))
        
        print("Pre-training completed!")
    
    def finetune(self, train_loader, val_loader, num_epochs=50, pretrained_checkpoint=None, save_every=5):
        """
        Fine-tuning stage with supervised learning.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: int, number of training epochs
            pretrained_checkpoint: str, path to pre-trained checkpoint
            save_every: int, save checkpoint every N epochs
        """
        print("Starting fine-tuning...")
        
        # Load pre-trained weights if provided
        if pretrained_checkpoint:
            self._load_checkpoint(pretrained_checkpoint, mode='pretrain')
            print(f"Loaded pre-trained weights from {pretrained_checkpoint}")
        
        # Set model to fine-tuning mode
        self.model.train()
        
        # Initialize loss function and optimizer
        criterion = CombinedFinetuneLoss()
        optimizer = get_optimizer(self.model, mode='finetune')
        scheduler = get_scheduler(optimizer, num_epochs)
        
        # Reset training state for fine-tuning
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_loss, train_metrics = self._train_finetune_epoch(train_loader, criterion, optimizer)
            
            # Validation
            val_loss, val_metrics = self._validate_finetune_epoch(val_loader, criterion)
            
            # Update learning rate
            scheduler.step()
            
            # Log metrics
            self._log_finetune_metrics(train_loss, train_metrics, val_loss, val_metrics, epoch)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_metrics['accuracy']:.3f}")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0 or val_loss < self.best_val_loss:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                self._save_checkpoint(epoch, 'finetune', is_best=(val_loss < self.best_val_loss))
        
        print("Fine-tuning completed!")
    
    def _train_pretrain_epoch(self, train_loader, criterion, optimizer):
        """Train one epoch during pre-training."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f"Pre-train Epoch {self.current_epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Get model outputs
            outputs = self.model(batch['image'], mode='pretrain')
            
            # Prepare targets for loss computation
            targets = {
                'original_image': batch['original_image'],
                'mask': batch['mask'],
                'ground_truth_mask': batch['ground_truth_mask'],
                'labels': batch['label']
            }
            
            # Compute loss
            loss, loss_dict = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        return total_loss / num_batches
    
    def _validate_pretrain_epoch(self, val_loader, criterion):
        """Validate one epoch during pre-training."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(self.device)
                
                # Forward pass
                outputs = self.model(batch['image'], mode='pretrain')
                
                # Prepare targets
                targets = {
                    'original_image': batch['original_image'],
                    'mask': batch['mask'],
                    'ground_truth_mask': batch['ground_truth_mask'],
                    'labels': batch['label']
                }
                
                # Compute loss
                loss, _ = criterion(outputs, targets)
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def _train_finetune_epoch(self, train_loader, criterion, optimizer):
        """Train one epoch during fine-tuning."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f"Finetune Epoch {self.current_epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Get model outputs
            outputs = self.model(batch['image'], mode='finetune')
            
            # Prepare targets
            targets = {
                'mask': batch['mask'],
                'labels': batch['label']
            }
            
            # Compute loss
            loss, loss_dict = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Collect predictions for metrics
            preds = torch.argmax(outputs['classification_logits'], dim=1).cpu().numpy()
            labels = batch['label'].cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Compute epoch metrics
        metrics = self._compute_metrics(np.array(all_preds), np.array(all_labels))
        
        return total_loss / num_batches, metrics
    
    def _validate_finetune_epoch(self, val_loader, criterion):
        """Validate one epoch during fine-tuning."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(self.device)
                
                # Forward pass
                outputs = self.model(batch['image'], mode='finetune')
                
                # Prepare targets
                targets = {
                    'mask': batch['mask'],
                    'labels': batch['label']
                }
                
                # Compute loss
                loss, _ = criterion(outputs, targets)
                total_loss += loss.item()
                
                # Collect predictions
                logits = outputs['classification_logits']
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(batch['label'].cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Compute metrics
        metrics = self._compute_metrics(np.array(all_preds), np.array(all_labels), np.array(all_probs))
        
        return total_loss / num_batches, metrics
    
    def _compute_metrics(self, preds, labels, probs=None):
        """Compute classification metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        
        # AUC if probabilities available
        if probs is not None and probs.shape[1] > 1:
            metrics['auc'] = roc_auc_score(labels, probs[:, 1])
        
        return metrics
    
    def _log_finetune_metrics(self, train_loss, train_metrics, val_loss, val_metrics, epoch):
        """Log fine-tuning metrics to tensorboard."""
        # Losses
        self.writer.add_scalar('FineTune/Train_Loss', train_loss, epoch)
        self.writer.add_scalar('FineTune/Val_Loss', val_loss, epoch)
        
        # Training metrics
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'FineTune/Train_{key.capitalize()}', value, epoch)
        
        # Validation metrics
        for key, value in val_metrics.items():
            self.writer.add_scalar(f'FineTune/Val_{key.capitalize()}', value, epoch)
    
    def _save_checkpoint(self, epoch, mode, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'config_path': self.config_path,
            'mode': mode,
            'training_history': self.training_history
        }
        
        # Regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'{mode}_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Best checkpoint
        if is_best:
            best_path = os.path.join(self.save_dir, f'{mode}_best.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def _load_checkpoint(self, checkpoint_path, mode='finetune'):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if mode == checkpoint['mode']:
            self.current_epoch = checkpoint['epoch']
            self.training_history = checkpoint.get('training_history', {'train_loss': [], 'val_loss': []})
        
        print(f"Loaded checkpoint from {checkpoint_path}")


def main():
    """Main training function - example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train AuraNet')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['pretrain', 'finetune', 'both'], 
                       default='both', help='Training mode')
    parser.add_argument('--data_root', type=str, required=True, help='Data root directory')
    parser.add_argument('--train_annotations', type=str, required=True, help='Training annotations file')
    parser.add_argument('--val_annotations', type=str, required=True, help='Validation annotations file')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save checkpoints')
    parser.add_argument('--pretrained_checkpoint', type=str, help='Pre-trained checkpoint path')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data workers')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = AuraNetTrainer(args.config, args.save_dir, args.device)
    
    if args.mode in ['pretrain', 'both']:
        # Pre-training data loaders
        train_loader, val_loader = create_data_loaders(
            args.data_root, args.data_root, args.train_annotations, args.val_annotations,
            batch_size=args.batch_size, num_workers=args.num_workers, mode='pretrain'
        )
        
        # Run pre-training
        trainer.pretrain(train_loader, val_loader, num_epochs=100)
        pretrained_checkpoint = os.path.join(args.save_dir, 'pretrain_best.pth')
    else:
        pretrained_checkpoint = args.pretrained_checkpoint
    
    if args.mode in ['finetune', 'both']:
        # Fine-tuning data loaders
        train_loader, val_loader = create_data_loaders(
            args.data_root, args.data_root, args.train_annotations, args.val_annotations,
            batch_size=args.batch_size, num_workers=args.num_workers, mode='finetune'
        )
        
        # Run fine-tuning
        trainer.finetune(train_loader, val_loader, num_epochs=50, 
                        pretrained_checkpoint=pretrained_checkpoint)


if __name__ == '__main__':
    main()
