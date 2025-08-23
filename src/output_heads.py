"""
Final Fusion and Output Heads
Implements DSF, ClassificationHead, and SegmentationHead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add src directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from utils import LayerNorm, Block, CBAMChannelAttention
try:
    from timm.layers import trunc_normal_  # New import path
except ImportError:
    from timm.models.layers import trunc_normal_  # Fallback to old path
import yaml


class DSF(nn.Module):
    """Dynamic Selection Fusion Module."""
    
    def __init__(self, dim, config=None):
        super().__init__()
        self.dim = dim
        
        # Load config if provided
        if config is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        # Dynamic selection network
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.selection_fc = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 2),  # 2 weights for spatial and frequency
            nn.Softmax(dim=1)
        )
        
        # Channel attention
        reduction = config['model']['cbam_reduction']
        self.channel_attention = CBAMChannelAttention(dim, reduction=reduction)
        
    def forward(self, spatial_feat, freq_feat):
        """
        Args:
            spatial_feat: (B, C, H, W)
            freq_feat: (B, C, H, W)
            
        Returns:
            fused_output: (B, C, H, W)
            pooled_output: (B, C)
        """
        B, C, H, W = spatial_feat.shape
        
        # Combine features
        combined_feat = spatial_feat + freq_feat  # (B, C, H, W)
        
        # Dynamic selection
        pooled_combined = self.global_pool(combined_feat).squeeze(-1).squeeze(-1)  # (B, C)
        weights = self.selection_fc(pooled_combined)  # (B, 2)
        
        w_spatial = weights[:, 0:1].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
        w_freq = weights[:, 1:2].unsqueeze(-1).unsqueeze(-1)    # (B, 1, 1, 1)
        
        # Weighted combination
        selected_feat = w_spatial * spatial_feat + w_freq * freq_feat
        
        # Channel attention
        channel_att = self.channel_attention(selected_feat)
        fused_output = selected_feat * channel_att
        
        # Global pooled output
        pooled_output = self.global_pool(fused_output).squeeze(-1).squeeze(-1)  # (B, C)
        
        return fused_output, pooled_output


class ClassificationHead(nn.Module):
    """Classification head for binary real/fake prediction."""
    
    def __init__(self, input_dim, config=None):
        super().__init__()
        
        # Load config if provided
        if config is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        # Extract parameters from config
        num_classes = config['num_classes']
        hidden_dim = config['model']['mlp_hidden_dim']
        dropout = config['model']['classification_dropout']
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, pooled_features):
        """
        Args:
            pooled_features: (B, input_dim)
            
        Returns:
            logits: (B, num_classes)
        """
        return self.classifier(pooled_features)


class SegmentationHead(nn.Module):
    """Segmentation head for grayscale manipulation mask prediction."""
    
    def __init__(self, input_dim, decoder_embed_dim=342, decoder_depth=1):
        super().__init__()
        
        # Project input channels to decoder embedding dimension
        self.proj = nn.Conv2d(input_dim, decoder_embed_dim, kernel_size=1)
        
        # Stack of ConvNeXt V2 blocks
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim) for _ in range(decoder_depth)
        ])
        
        # Upsampling layers to restore original resolution
        # From 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256
        self.upsample_layers = nn.ModuleList([
            nn.ConvTranspose2d(decoder_embed_dim, decoder_embed_dim // 2, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.ConvTranspose2d(decoder_embed_dim // 2, decoder_embed_dim // 4, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.ConvTranspose2d(decoder_embed_dim // 4, decoder_embed_dim // 8, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.ConvTranspose2d(decoder_embed_dim // 8, decoder_embed_dim // 16, kernel_size=4, stride=2, padding=1),  # 64x64 -> 128x128
            nn.ConvTranspose2d(decoder_embed_dim // 16, decoder_embed_dim // 32, kernel_size=4, stride=2, padding=1),  # 128x128 -> 256x256
        ])
        
        # Final prediction layer
        self.pred = nn.Conv2d(decoder_embed_dim // 32, 1, kernel_size=1)
        
        # Final activation
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, fused_features):
        """
        Args:
            fused_features: (B, input_dim, H, W) - typically (B, 512, 8, 8)
            
        Returns:
            mask: (B, 1, 256, 256) - full resolution grayscale segmentation mask
        """
        # Project to decoder dimension
        x = self.proj(fused_features)  # (B, decoder_embed_dim, 8, 8)
        
        # Apply decoder blocks
        for block in self.decoder_blocks:
            x = block(x)
        
        # Progressive upsampling with activation
        for upsample_layer in self.upsample_layers:
            x = upsample_layer(x)
            x = F.gelu(x)
        
        # Final prediction
        x = self.pred(x)  # (B, 1, 256, 256)
        
        # Apply sigmoid activation
        mask = self.sigmoid(x)
        
        return mask


# Additional heads for pre-training stage
class ImageDecoder(nn.Module):
    """Image decoder for self-supervised pre-training following FCMAE pattern."""
    
    def __init__(self, input_dim, output_channels=3, decoder_embed_dim=512, decoder_depth=1, patch_size=32):
        super().__init__()
        self.patch_size = patch_size
        
        # Following FCMAE implementation exactly
        # proj: input_dim -> decoder_embed_dim
        self.proj = nn.Conv2d(input_dim, decoder_embed_dim, kernel_size=1)
        
        # mask token for reconstruction
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim, 1, 1))
        
        # decoder: Stack of ConvNeXt V2 blocks 
        decoder = [Block(decoder_embed_dim, drop_path=0.) for _ in range(decoder_depth)]
        self.decoder = nn.Sequential(*decoder)
        
        # pred: decoder_embed_dim -> patch_size^2 * output_channels (following FCMAE exactly)
        self.pred = nn.Conv2d(decoder_embed_dim, patch_size ** 2 * output_channels, kernel_size=1)
        
        # Initialize weights following FCMAE pattern
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Initialize weights following FCMAE pattern."""
        if isinstance(m, nn.Conv2d):
            w = m.weight.data
            trunc_normal_(w.view([w.shape[0], -1]))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        # Initialize mask token
        if hasattr(self, 'mask_token'):
            torch.nn.init.normal_(self.mask_token, std=.02)
        
    def forward(self, spatial_features, mask=None):
        """
        Args:
            spatial_features: (B, input_dim, H, W) - typically (B, 512, 8, 8)
            mask: (B, H*W) - optional mask for reconstruction
            
        Returns:
            pred: (B, patch_size^2 * output_channels, H, W) - patch-wise predictions
        """
        # Project to decoder dimension (following FCMAE forward_decoder)
        x = self.proj(spatial_features)
        
        # Append mask tokens if mask is provided
        if mask is not None:
            n, c, h, w = x.shape
            mask = mask.reshape(-1, h, w).unsqueeze(1).type_as(x)
            mask_token = self.mask_token.repeat(x.shape[0], 1, x.shape[2], x.shape[3])
            x = x * (1. - mask) + mask_token * mask
        
        # Decoder blocks
        x = self.decoder(x)
        
        # Prediction (patch-wise output in channels)
        pred = self.pred(x)
        
        return pred


class MaskDecoder(nn.Module):
    """Mask decoder for self-supervised pre-training following FCMAE pattern."""
    
    def __init__(self, input_dim, decoder_embed_dim=512, decoder_depth=1, patch_size=32):
        super().__init__()
        self.patch_size = patch_size
        
        # Following FCMAE implementation exactly
        # proj: input_dim -> decoder_embed_dim  
        self.proj = nn.Conv2d(input_dim, decoder_embed_dim, kernel_size=1)
        
        # mask token for reconstruction
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim, 1, 1))
        
        # decoder: Stack of ConvNeXt V2 blocks
        decoder = [Block(decoder_embed_dim, drop_path=0.) for _ in range(decoder_depth)]
        self.decoder = nn.Sequential(*decoder)
        
        # pred: decoder_embed_dim -> patch_size^2 * 1 (for mask reconstruction following FCMAE)
        self.pred = nn.Conv2d(decoder_embed_dim, patch_size ** 2 * 1, kernel_size=1)
        
        # Initialize weights following FCMAE pattern
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Initialize weights following FCMAE pattern."""
        if isinstance(m, nn.Conv2d):
            w = m.weight.data
            trunc_normal_(w.view([w.shape[0], -1]))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        # Initialize mask token
        if hasattr(self, 'mask_token'):
            torch.nn.init.normal_(self.mask_token, std=.02)
        
    def forward(self, freq_features, mask=None):
        """
        Args:
            freq_features: (B, input_dim, H, W) - typically (B, 512, 8, 8)
            mask: (B, H*W) - optional mask for reconstruction
            
        Returns:
            pred: (B, patch_size^2 * 1, H, W) - patch-wise mask predictions
        """
        # Project to decoder dimension (following FCMAE forward_decoder)
        x = self.proj(freq_features)
        
        # Append mask tokens if mask is provided
        if mask is not None:
            n, c, h, w = x.shape
            mask = mask.reshape(-1, h, w).unsqueeze(1).type_as(x)
            mask_token = self.mask_token.repeat(x.shape[0], 1, x.shape[2], x.shape[3])
            x = x * (1. - mask) + mask_token * mask
        
        # Decoder blocks
        x = self.decoder(x)
        
        # Prediction (patch-wise output in channels)
        pred = self.pred(x)
        
        return pred


class ContrastiveProjectionHead(nn.Module):
    """Projection head for contrastive learning during pre-training."""
    
    def __init__(self, input_dim, projection_dim=128, hidden_dim=512):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )
        
    def forward(self, pooled_features):
        """
        Args:
            pooled_features: (B, input_dim)
            
        Returns:
            projected: (B, projection_dim) L2-normalized embeddings
        """
        projected = self.projection(pooled_features)
        # L2 normalize
        projected = F.normalize(projected, p=2, dim=1)
        return projected
