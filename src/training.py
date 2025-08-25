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
import sys

# Add the project root directory to Python path for absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config_utils import safe_load_config, load_config_safe


class ImageReconstructionLoss(nn.Module):
    """L2 loss for image reconstruction on masked patches, FCMAE style."""
    
    def __init__(self, patch_size=32, norm_pix_loss=False):
        super().__init__()
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
        self.mse_loss = nn.MSELoss()
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        Following FCMAE exactly
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
    
    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: (B, patch_size^2 * 3, H, W) predicted patches from decoder
            target: (B, 3, H_full, W_full) original image 
            mask: (B, L) binary mask (1 for removed patches, 0 for kept), optional
                  where L = (H*W) - number of patches in decoder output
            
        Returns:
            loss: scalar tensor
        """
        # Convert 4D prediction to 3D following FCMAE forward_loss exactly
        if len(pred.shape) == 4:
            n, c, _, _ = pred.shape
            pred = pred.reshape(n, c, -1)
            pred = torch.einsum('ncl->nlc', pred)  # (B, L, patch_size^2 * 3)

        # Patchify target image - CRITICAL: must match decoder spatial resolution
        # Target must be resized to match decoder output spatial dimensions
        B, _, target_H, target_W = target.shape
        _, pred_L, _ = pred.shape
        
        # Calculate decoder spatial resolution from number of patches
        decoder_spatial_dim = int(pred_L ** 0.5)  # sqrt(L) = H = W of decoder
        expected_target_size = decoder_spatial_dim * self.patch_size
        
        # Resize target to match expected size if needed
        if target_H != expected_target_size or target_W != expected_target_size:
            target = F.interpolate(target, size=(expected_target_size, expected_target_size), 
                                 mode='bilinear', align_corners=False)
        
        target = self.patchify(target)  # (B, L, patch_size^2 * 3)
        
        # Apply pixel normalization if enabled (following FCMAE)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        # Compute loss (following FCMAE exactly)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [B, L], mean loss per patch

        # If mask is provided, only compute loss on removed patches (following FCMAE)
        if mask is not None:
            # Ensure mask shape matches exactly: must be (B, L) where L matches pred
            if mask.shape[1] != pred_L:
                raise ValueError(f"Mask shape {mask.shape} doesn't match pred patches {pred_L}")
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)  # mean loss on removed patches
        else:
            loss = loss.mean()  # mean loss on all patches
        
        return loss


class MaskReconstructionLoss(nn.Module):
    """Combined L1 and SSIM loss for mask reconstruction, FCMAE style."""
    
    def __init__(self, patch_size=32, config=None):
        super().__init__()
        self.patch_size = patch_size
        
        # Load config if provided
        if config is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        # Extract weights from config
        self.l1_weight = config['training']['pretrain']['l1_weight']
        self.ssim_weight = config['training']['pretrain']['ssim_weight']
        self.l1_loss = nn.L1Loss()
        
    def patchify_mask(self, masks):
        """
        masks: (N, 1, H, W)
        x: (N, L, patch_size**2 * 1)
        """
        p = self.patch_size
        assert masks.shape[2] == masks.shape[3] and masks.shape[2] % p == 0

        h = w = masks.shape[2] // p
        x = masks.reshape(shape=(masks.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(masks.shape[0], h * w, p**2 * 1))
        return x
        
    def unpatchify_mask(self, x):
        """
        x: (N, L, patch_size**2 * 1)
        masks: (N, 1, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        masks = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return masks
    
    def forward(self, pred_mask, target_mask, mask=None):
        """
        Args:
            pred_mask: (B, patch_size^2 * 1, H, W) predicted mask patches from decoder
            target_mask: (B, 1, H_full, W_full) ground truth mask
            mask: (B, L) binary mask (1 for removed patches, 0 for kept), optional
            
        Returns:
            loss: scalar tensor
        """
        # Convert 4D prediction to 3D following FCMAE pattern
        if len(pred_mask.shape) == 4:
            n, c, _, _ = pred_mask.shape
            pred_mask = pred_mask.reshape(n, c, -1)
            pred_mask = torch.einsum('ncl->nlc', pred_mask)  # (B, L, patch_size^2 * 1)

        # Patchify target mask - CRITICAL: must match decoder spatial resolution
        B, _, target_H, target_W = target_mask.shape
        _, pred_L, _ = pred_mask.shape
        
        # Calculate decoder spatial resolution from number of patches
        decoder_spatial_dim = int(pred_L ** 0.5)
        expected_target_size = decoder_spatial_dim * self.patch_size
        
        # Resize target to match expected size if needed
        if target_H != expected_target_size or target_W != expected_target_size:
            target_mask = F.interpolate(target_mask, size=(expected_target_size, expected_target_size), 
                                      mode='bilinear', align_corners=False)
        
        target = self.patchify_mask(target_mask)  # (B, L, patch_size^2 * 1)
        
        # Compute patch-wise L1 loss 
        l1_loss = (pred_mask - target).abs().mean(dim=-1)  # [B, L], mean loss per patch
        
        # If mask is provided, only compute loss on removed patches
        if mask is not None:
            # Ensure mask shape matches exactly
            if mask.shape[1] != pred_L:
                raise ValueError(f"Mask shape {mask.shape} doesn't match pred patches {pred_L}")
            l1_loss = (l1_loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            l1_loss = l1_loss.mean()
        
        # For SSIM, we need to convert back to full resolution
        # Unpatchify predictions for SSIM calculation
        pred_full = self.unpatchify_mask(pred_mask)  # (B, 1, H, W)
        pred_full = torch.sigmoid(pred_full)  # Apply sigmoid for proper range
        
        # SSIM loss (1 - SSIM for minimization)
        ssim_val = ssim(pred_full, target_mask, data_range=1.0, size_average=True)
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
        
        # Get patch size from config or default to 32
        patch_size = config['model'].get('patch_size', 32)
        
        self.image_loss = ImageReconstructionLoss(patch_size=patch_size)
        self.mask_loss = MaskReconstructionLoss(patch_size=patch_size, config=config)
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
            targets.get('mask', None)  # mask for removed patches (optional)
        )
        
        # Mask reconstruction loss
        mask_loss_val = self.mask_loss(
            outputs['reconstructed_mask'],
            targets['ground_truth_mask'],
            targets.get('mask', None)  # mask for removed patches (optional)
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


def create_random_mask(encoder_output_shape, mask_ratio, patch_size=32, device=None):
    """
    Create random mask for self-supervised pre-training following FCMAE exactly.
    
    Args:
        encoder_output_shape: (B, C, H, W) - shape of encoder output (e.g., (2, 512, 8, 8))
        mask_ratio: float - ratio of patches to mask
        patch_size: int - patch size for reconstruction (relates to input image, not encoder)
        device: torch device
        
    Returns:
        mask: (B, L) binary mask where L = H * W (encoder spatial resolution)
              0 = keep, 1 = remove (following FCMAE convention)
    """
    B, C, H, W = encoder_output_shape
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # CRITICAL: L is the number of spatial positions in encoder output
    # Each spatial position will reconstruct one patch_size×patch_size region of input
    # This follows FCMAE's logic but for AuraNet's architecture
    L = H * W  # e.g., 8*8 = 64 positions/patches
    len_keep = int(L * (1 - mask_ratio))

    # Generate random mask following FCMAE exactly (lines 125-135)
    noise = torch.randn(B, L, device=device)

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # generate the binary mask: 0 is keep 1 is remove
    mask = torch.ones([B, L], device=device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    return mask


def create_random_mask_legacy(shape, config=None):
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
        config = safe_load_config(config_path)
    elif isinstance(config, str):
        config = safe_load_config(config)
    else:
        config = load_config_safe(config)
    
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
        config = safe_load_config(config_path)
    elif isinstance(config, str):
        config = safe_load_config(config)
    else:
        config = load_config_safe(config)
    print(f"My mode is: {mode}")
    if mode == 'pretrain':
        # Single learning rate for pre-training
        pretrain_lr = config['training']['pretrain']['learning_rate']
        weight_decay = config['training']['pretrain']['weight_decay']
        print(f"Pretrain lr: {type(pretrain_lr)}, weight decay: {type(weight_decay)} ")
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
            {'params': encoder_params},
            {'params': head_params}
        ]
        
        # Elegant fix: Initialize with a placeholder LR, then set group-specific LRs.
        # This ensures group['initial_lr'] is a float, not a list, preventing a TypeError
        # in the LambdaLR scheduler.
        optimizer = torch.optim.AdamW(param_groups, lr=head_lr, weight_decay=weight_decay)
        optimizer.param_groups[0]['lr'] = encoder_lr
        optimizer.param_groups[1]['lr'] = head_lr
    
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
        config = safe_load_config(config_path)
    elif isinstance(config, str):
        config = safe_load_config(config)
    else:
        config = load_config_safe(config)
    
    # Validate mode
    if mode not in ['pretrain', 'finetune']:
        raise ValueError(f"Invalid mode '{mode}'. Must be 'pretrain' or 'finetune'.")

    # Get parameters based on mode
    if mode == 'pretrain':
        num_epochs = config['training']['pretrain']['epochs']
        warmup_epochs = config['training']['pretrain']['warmup_epochs']
    else:  # finetune
        num_epochs = config['training']['finetune']['epochs']
        warmup_epochs = config['training']['finetune']['warmup_epochs']
    def lr_lambda(epoch):
        """Cosine decay with warmup."""
        if epoch < warmup_epochs:
            # Linear warmup
            return float(epoch + 1) / float(max(1, warmup_epochs))
        else:
            # Cosine decay
            progress = float(epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    # LambdaLR will apply the same lr_lambda to all parameter groups
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return scheduler
