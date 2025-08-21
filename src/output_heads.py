"""
Final Fusion and Output Heads
Implements DSF, ClassificationHead, and SegmentationHead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import LayerNorm, Block, CBAMChannelAttention
import yaml
import os


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
        
        # Final prediction layer
        self.pred = nn.Conv2d(decoder_embed_dim, 1, kernel_size=1)
        
        # Final activation
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, fused_features):
        """
        Args:
            fused_features: (B, input_dim, H, W)
            
        Returns:
            mask: (B, 1, H, W) grayscale segmentation mask
        """
        # Project to decoder dimension
        x = self.proj(fused_features)  # (B, decoder_embed_dim, H, W)
        
        # Apply decoder blocks
        for block in self.decoder_blocks:
            x = block(x)
        
        # Final prediction
        x = self.pred(x)  # (B, 1, H, W)
        
        # Apply sigmoid activation
        mask = self.sigmoid(x)
        
        return mask


# Additional heads for pre-training stage
class ImageDecoder(nn.Module):
    """Image decoder for self-supervised pre-training."""
    
    def __init__(self, input_dim, output_channels=3, decoder_embed_dim=342, decoder_depth=1):
        super().__init__()
        
        # Project input channels to decoder embedding dimension
        self.proj = nn.Conv2d(input_dim, decoder_embed_dim, kernel_size=1)
        
        # Stack of ConvNeXt V2 blocks
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim) for _ in range(decoder_depth)
        ])
        
        # Final prediction layer for RGB reconstruction
        self.pred = nn.Conv2d(decoder_embed_dim, output_channels, kernel_size=1)
        
    def forward(self, spatial_features):
        """
        Args:
            spatial_features: (B, input_dim, H, W)
            
        Returns:
            reconstructed: (B, output_channels, H, W)
        """
        # Project to decoder dimension
        x = self.proj(spatial_features)
        
        # Apply decoder blocks
        for block in self.decoder_blocks:
            x = block(x)
        
        # Final prediction
        reconstructed = self.pred(x)
        
        return reconstructed


class MaskDecoder(nn.Module):
    """Mask decoder for self-supervised pre-training."""
    
    def __init__(self, input_dim, decoder_embed_dim=342, decoder_depth=1):
        super().__init__()
        
        # Project input channels to decoder embedding dimension
        self.proj = nn.Conv2d(input_dim, decoder_embed_dim, kernel_size=1)
        
        # Stack of ConvNeXt V2 blocks
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim) for _ in range(decoder_depth)
        ])
        
        # Final prediction layer for mask reconstruction
        self.pred = nn.Conv2d(decoder_embed_dim, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, freq_features):
        """
        Args:
            freq_features: (B, input_dim, H, W)
            
        Returns:
            reconstructed_mask: (B, 1, H, W)
        """
        # Project to decoder dimension
        x = self.proj(freq_features)
        
        # Apply decoder blocks
        for block in self.decoder_blocks:
            x = block(x)
        
        # Final prediction
        x = self.pred(x)
        reconstructed_mask = self.sigmoid(x)
        
        return reconstructed_mask


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
