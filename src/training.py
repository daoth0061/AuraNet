"""
Training utilities and loss functions for AuraNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pytorch_metric_learning import losses
from pytorch_msssim import ssim
import yaml
import os


class ImageReconstructionLoss(nn.Module):
    """L2 loss for image reconstruction on masked patches."""
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred, target, mask):
        """
        Args:
            pred: (B, 3, H, W) predicted image
            target: (B, 3, H, W) original image
            mask: (B, 1, H, W) binary mask (1 for masked regions)
            
        Returns:
            loss: scalar tensor
        """
        # Apply mask - only compute loss on masked regions
        pred_masked = pred * mask
        target_masked = target * mask
        
        # Compute MSE loss
        loss = self.mse_loss(pred_masked, target_masked)
        
        # Normalize by the number of masked pixels
        num_masked = mask.sum()
        if num_masked > 0:
            loss = loss * (mask.numel() / num_masked)
        
        return loss


class MaskReconstructionLoss(nn.Module):
    """Combined L1 and SSIM loss for mask reconstruction."""
    
    def __init__(self, config=None):
        super().__init__()
        
        # Load config if provided
        if config is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        # Extract weights from config
        self.l1_weight = config['training']['pretrain']['l1_weight']
        self.ssim_weight = config['training']['pretrain']['ssim_weight']
        self.l1_loss = nn.L1Loss()
    
    def forward(self, pred_mask, target_mask):
        """
        Args:
            pred_mask: (B, 1, H, W) predicted mask
            target_mask: (B, 1, H, W) ground truth mask
            
        Returns:
            loss: scalar tensor
        """
        # L1 loss
        l1_loss = self.l1_loss(pred_mask, target_mask)
        
        # SSIM loss (1 - SSIM for minimization)
        ssim_val = ssim(pred_mask, target_mask, data_range=1.0, size_average=True)
        ssim_loss = 1.0 - ssim_val
        
        # Combined loss
        total_loss = self.l1_weight * l1_loss + self.ssim_weight * ssim_loss
        
        return total_loss


class AMSoftmaxLoss(nn.Module):
    """Additive Margin Softmax Loss for classification."""
    
    def __init__(self, in_features, out_features, config=None):
        super().__init__()
        
        # Load config if provided
        if config is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        self.in_features = in_features
        self.out_features = out_features
        self.margin = config['model']['am_softmax_margin']
        self.scale = config['model']['am_softmax_scale']
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, features, labels):
        """
        Args:
            features: (B, in_features)
            labels: (B,) ground truth labels
            
        Returns:
            loss: scalar tensor
        """
        # Normalize features and weights
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        
        # Apply margin to the target class
        phi = cosine - self.margin
        
        # Create one-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        # Apply margin
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(output, labels)
        
        return loss


class CombinedPretrainLoss(nn.Module):
    """Combined loss for self-supervised pre-training."""
    
    def __init__(self, config=None):
        super().__init__()
        
        # Load config if provided
        if config is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        # Extract weights from config
        self.image_weight = config['training']['pretrain']['image_loss_weight']
        self.mask_weight = config['training']['pretrain']['mask_loss_weight']
        self.supcon_weight = config['training']['pretrain']['supcon_loss_weight']
        
        self.image_loss = ImageReconstructionLoss()
        self.mask_loss = MaskReconstructionLoss(config=config)
        self.supcon_loss = losses.SupConLoss(temperature=config['training']['pretrain']['supcon_temperature'])
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict containing model outputs
            targets: dict containing ground truth data
            
        Returns:
            loss: scalar tensor
            loss_dict: dict of individual losses
        """
        # Image reconstruction loss
        img_loss = self.image_loss(
            outputs['reconstructed_image'],
            targets['original_image'],
            targets['mask']
        )
        
        # Mask reconstruction loss
        mask_loss_val = self.mask_loss(
            outputs['reconstructed_mask'],
            targets['ground_truth_mask']
        )
        
        # Supervised contrastive loss
        embeddings = outputs['contrastive_embedding']
        labels = targets.get('labels', None)
        
        if labels is not None:
            # Create pseudo-labels for contrastive learning
            supcon_loss_val = self.supcon_loss(embeddings, labels)
        else:
            supcon_loss_val = torch.tensor(0.0, device=embeddings.device)
        
        # Combined loss
        total_loss = (self.image_weight * img_loss + 
                     self.mask_weight * mask_loss_val + 
                     self.supcon_weight * supcon_loss_val)
        
        loss_dict = {
            'image_loss': img_loss,
            'mask_loss': mask_loss_val,
            'supcon_loss': supcon_loss_val,
            'total_loss': total_loss
        }
        
        return total_loss, loss_dict


class CombinedFinetuneLoss(nn.Module):
    """Combined loss for supervised fine-tuning."""
    
    def __init__(self, config=None):
        super().__init__()
        
        # Load config if provided
        if config is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        # Extract weights from config
        self.seg_weight = config['training']['finetune']['seg_loss_weight']
        self.class_weight = config['training']['finetune']['class_loss_weight']
        
        # Use segmentation-specific config for mask loss
        seg_config = config.copy()
        seg_config['training']['pretrain']['l1_weight'] = config['training']['finetune']['seg_l1_weight']
        seg_config['training']['pretrain']['ssim_weight'] = config['training']['finetune']['seg_ssim_weight']
        
        self.seg_loss = MaskReconstructionLoss(config=seg_config)
        # Initialize AM-Softmax loss (will be set up during first forward pass)
        self.class_loss = None
        self.num_classes = config['num_classes']
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict containing model outputs
            targets: dict containing ground truth data
            
        Returns:
            loss: scalar tensor
            loss_dict: dict of individual losses
        """
        # Segmentation loss
        seg_loss_val = self.seg_loss(
            outputs['segmentation_mask'],
            targets['mask']
        )
        
        # Classification loss
        logits = outputs['classification_logits']
        labels = targets['labels']
        
        # Initialize AM-Softmax loss if not done yet
        if self.class_loss is None:
            feature_dim = logits.shape[1]  # This might need adjustment if using features instead of logits
            self.class_loss = AMSoftmaxLoss(feature_dim, self.num_classes)
            self.class_loss = self.class_loss.to(logits.device)
        
        # For now, use simple cross-entropy (can replace with AM-Softmax if needed)
        class_loss_val = F.cross_entropy(logits, labels)
        
        # Combined loss
        total_loss = (self.seg_weight * seg_loss_val + 
                     self.class_weight * class_loss_val)
        
        loss_dict = {
            'seg_loss': seg_loss_val,
            'class_loss': class_loss_val,
            'total_loss': total_loss
        }
        
        return total_loss, loss_dict


def create_random_mask(shape, config=None):
    """
    Create random mask for self-supervised pre-training.
    
    Args:
        shape: (B, H, W) shape of the mask
        config: configuration dict or path to config file
        
    Returns:
        mask: (B, 1, H, W) binary mask
    """
    # Load config if provided
    if config is None:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif isinstance(config, str):
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
    
    mask_ratio = config['training']['pretrain']['mask_ratio']
    
    B, H, W = shape
    
    # Create random mask
    mask = torch.rand(B, H, W) < mask_ratio
    mask = mask.float().unsqueeze(1)  # (B, 1, H, W)
    
    return mask


def get_optimizer(model, mode='finetune', config=None):
    """
    Create optimizer with differential learning rates.
    
    Args:
        model: AuraNet model
        mode: 'pretrain' or 'finetune'
        config: configuration dict or path to config file
        
    Returns:
        optimizer: PyTorch optimizer
    """
    # Load config if provided
    if config is None:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif isinstance(config, str):
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
    
    if mode == 'pretrain':
        # Single learning rate for pre-training
        pretrain_lr = config['training']['pretrain']['learning_rate']
        weight_decay = config['training']['pretrain']['weight_decay']
        optimizer = torch.optim.AdamW(model.parameters(), lr=pretrain_lr, weight_decay=weight_decay)
    
    elif mode == 'finetune':
        # Differential learning rates for fine-tuning
        encoder_lr = config['training']['finetune']['encoder_lr']
        head_lr = config['training']['finetune']['head_lr']
        weight_decay = config['training']['finetune']['weight_decay']
        
        encoder_params = []
        head_params = []
        
        # Encoder parameters (everything except final heads)
        for name, param in model.named_parameters():
            if any(head_name in name for head_name in ['dsf', 'classification_head', 'segmentation_head']):
                head_params.append(param)
            else:
                encoder_params.append(param)
        
        # Create parameter groups
        param_groups = [
            {'params': encoder_params, 'lr': encoder_lr},
            {'params': head_params, 'lr': head_lr}
        ]
        
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return optimizer


def get_scheduler(optimizer, mode, config=None):
    """
    Create learning rate scheduler with warmup.
    
    Args:
        optimizer: PyTorch optimizer
        mode: 'pretrain' or 'finetune'
        config: configuration dict or path to config file
        
    Returns:
        scheduler: PyTorch scheduler
    """
    # Load config if provided
    if config is None:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif isinstance(config, str):
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Get parameters based on mode
    if mode == 'pretrain':
        num_epochs = config['training']['pretrain']['epochs']
        warmup_epochs = config['training']['pretrain']['warmup_epochs']
    else:  # finetune
        num_epochs = config['training']['finetune']['epochs']
        warmup_epochs = config['training']['finetune']['warmup_epochs']
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    
    # Check if optimizer has multiple parameter groups
    if len(optimizer.param_groups) > 1:
        # Create separate lambda functions for each parameter group
        def make_lr_lambda():
            def _lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                else:
                    return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
            return _lr_lambda
        
        lr_lambda_list = [make_lr_lambda() for _ in range(len(optimizer.param_groups))]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_list)
    else:
        # Single parameter group
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return scheduler
