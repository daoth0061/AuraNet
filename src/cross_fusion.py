"""
Cross-Fusion Block Implementation
Implements Deformable and Standard Cross-Attention mechanisms
"""

import os
import sys

# Add the project root directory to Python path for absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import LayerNorm, create_grid_like, normalize_grid, ContinuousPositionalBias
import yaml
import os


class DeformableCrossAttention(nn.Module):
    """Deformable Cross-Attention for high-resolution stages (2 & 3) - Optimized for Speed."""
    
    def __init__(self, dim, heads=8, num_offsets=9, config=None):
        super().__init__()
        
        # Load config if provided
        if config is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        self.dim = dim
        self.heads = heads
        self.num_offsets = num_offsets
        self.scale = (dim // heads) ** -0.5
        
        # Offset scale from config
        self.offset_scale = config['model']['deformable_offset_scale']
        
        inner_dim = dim
        
        # Query, Key, Value projections - use Conv2d for efficiency
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_k = nn.Conv2d(dim, inner_dim, 1, bias=False)  
        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias=False)
        
        # SPEED OPTIMIZATION: Restore sophisticated offset prediction
        self.to_offsets = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, 2 * num_offsets, 1)  # Predict multiple offsets per position
        )
        
        # Restore relative positional bias
        self.rel_pos_bias = ContinuousPositionalBias(dim // heads, heads)
        
        # Output projection
        self.to_out = nn.Conv2d(inner_dim, dim, 1)
        
    def forward(self, query_map, kv_map):
        """
        Args:
            query_map: (B, C, H, W)
            kv_map: (B, C, H, W)
            
        Returns:
            out: (B, C, H, W)
        """
        B, C, H, W = query_map.shape
        
        # SPEED OPTIMIZATION: Restore sophisticated deformable attention
        # Generate queries
        q = self.to_q(query_map)  # (B, C, H, W)
        
        # Restore sophisticated offset prediction - multiple offsets per position
        offsets = self.to_offsets(q)  # (B, 2*num_offsets, H, W)
        offsets = offsets.view(B, 2, self.num_offsets, H, W)  # (B, 2, num_offsets, H, W)
        
        # Create base sampling grid
        device = query_map.device
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        base_grid = torch.stack([x_coords, y_coords], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # (B, 2, H, W)
        
        # Apply scaled offsets for each offset position
        offset_grids = []
        for i in range(self.num_offsets):
            offset = offsets[:, :, i] * self.offset_scale  # (B, 2, H, W)
            vgrid = base_grid + offset  # (B, 2, H, W)
            vgrid = vgrid.permute(0, 2, 3, 1)  # (B, H, W, 2)
            offset_grids.append(vgrid)
        
        # SPEED OPTIMIZATION: Efficient grid sampling for multiple offsets
        kv_sampled_list = []
        for vgrid in offset_grids:
            kv_sampled = F.grid_sample(kv_map, vgrid, mode='bilinear', 
                                       padding_mode='border', align_corners=True)  # (B, C, H, W)
            kv_sampled_list.append(kv_sampled)
        
        # Concatenate sampled features
        kv_multi = torch.cat(kv_sampled_list, dim=1)  # (B, C*num_offsets, H, W)
        
        # Generate keys and values from multi-sampled features
        k = self.to_k(kv_multi)  # (B, C*num_offsets, H, W)
        v = self.to_v(kv_multi)  # (B, C*num_offsets, H, W)
        
        # SPEED OPTIMIZATION: Use efficient attention computation
        # Reshape for attention: (B, heads, C//heads, H*W) -> (B, heads, H*W, C//heads)
        q_attn = q.view(B, self.heads, C // self.heads, H * W).transpose(-2, -1)
        k_attn = k.view(B, self.heads, C // self.heads, H * W).transpose(-2, -1)
        v_attn = v.view(B, self.heads, C // self.heads, H * W).transpose(-2, -1)
        
        # Restore sophisticated relative positional bias
        rel_pos_bias = self.rel_pos_bias(H, W, device=device)  # (num_heads, H*W, H*W)
        
        # SPEED OPTIMIZATION: Use Flash Attention when available
        try:
            # Try Flash Attention first (fastest)
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                out_attn = F.scaled_dot_product_attention(
                    q_attn, k_attn, v_attn,
                    attn_mask=rel_pos_bias if rel_pos_bias is not None else None,
                    dropout_p=0.0, is_causal=False
                )
        except:
            # Fallback to efficient matrix multiplication
            # Compute attention scores efficiently
            attn = torch.matmul(q_attn, k_attn.transpose(-2, -1)) * self.scale  # (B, heads, H*W, H*W)
            if rel_pos_bias is not None:
                attn = attn + rel_pos_bias.unsqueeze(0)  # Add relative positional bias
            attn = F.softmax(attn, dim=-1)
            
            # Apply attention to values
            out_attn = torch.matmul(attn, v_attn)  # (B, heads, H*W, C//heads)
        
        # Reshape back to feature map
        out = out_attn.transpose(-2, -1).reshape(B, C, H, W)  # (B, C, H, W)
        
        # Final projection
        out = self.to_out(out)
        
        return out


class StandardCrossAttention(nn.Module):
    """Standard Cross-Attention for low-resolution stages (4 & 5) - Optimized for Speed."""
    
    def __init__(self, dim, heads=8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        inner_dim = dim
        
        # Linear projections - optimized for speed
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        
        # Output projection
        self.to_out = nn.Linear(inner_dim, dim)
        
    def forward(self, query_map, kv_map):
        """
        Args:
            query_map: (B, C, H, W)
            kv_map: (B, C, H, W)
            
        Returns:
            out: (B, C, H, W)
        """
        B, C, H, W = query_map.shape
        HW = H * W
        
        # SPEED OPTIMIZATION: Efficient sequence processing
        # Flatten to sequences
        query_seq = query_map.flatten(2).transpose(1, 2)  # (B, H*W, C)
        kv_seq = kv_map.flatten(2).transpose(1, 2)        # (B, H*W, C)
        
        # Generate Q, K, V
        q = self.to_q(query_seq)  # (B, H*W, C)
        k = self.to_k(kv_seq)     # (B, H*W, C)
        v = self.to_v(kv_seq)     # (B, H*W, C)
        
        # SPEED OPTIMIZATION: Reshape for efficient attention
        # (B, H*W, heads, C//heads) -> (B, heads, H*W, C//heads)
        q_attn = q.view(B, HW, self.heads, C // self.heads).transpose(1, 2)
        k_attn = k.view(B, HW, self.heads, C // self.heads).transpose(1, 2)
        v_attn = v.view(B, HW, self.heads, C // self.heads).transpose(1, 2)
        
        # SPEED OPTIMIZATION: Use Flash Attention when available
        try:
            # Try Flash Attention first (fastest)
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                out_attn = F.scaled_dot_product_attention(
                    q_attn, k_attn, v_attn,
                    attn_mask=None, dropout_p=0.0, is_causal=False
                )
        except:
            # Fallback to efficient matrix multiplication
            # Compute attention scores efficiently
            attn = torch.matmul(q_attn, k_attn.transpose(-2, -1)) * self.scale  # (B, heads, H*W, H*W)
            attn = F.softmax(attn, dim=-1)
            
            # Apply attention to values
            out_attn = torch.matmul(attn, v_attn)  # (B, heads, H*W, C//heads)
        
        # SPEED OPTIMIZATION: Efficient reshape back
        # (B, heads, H*W, C//heads) -> (B, H*W, C)
        out = out_attn.transpose(1, 2).reshape(B, HW, C)
        
        # Final projection
        out = self.to_out(out)  # (B, H*W, C)
        
        # Reshape back to feature map
        out = out.transpose(1, 2).view(B, C, H, W)  # (B, C, H, W)
        
        return out


class ConvFFN(nn.Module):
    """Convolutional Feed-Forward Network."""
    
    def __init__(self, dim, hidden_dim=None, config=None):
        super().__init__()
        
        # Load config if provided
        if config is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        # Extract parameters from config
        hidden_ratio = config['model']['ffn_hidden_ratio']
        dropout = config['model']['cross_fusion_dropout']
        
        hidden_dim = hidden_dim or dim * hidden_ratio
        
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


class CrossFusionBlock(nn.Module):
    """Cross-Fusion Block for bidirectional stream dialogue."""
    
    def __init__(self, dim, use_deformable=True, heads=8, dropout=0.0):
        super().__init__()
        self.dim = dim
        
        # Attention modules
        if use_deformable:
            self.spatial_enhancer = DeformableCrossAttention(dim, heads)
            self.frequency_enhancer = DeformableCrossAttention(dim, heads)
        else:
            self.spatial_enhancer = StandardCrossAttention(dim, heads)
            self.frequency_enhancer = StandardCrossAttention(dim, heads)
        
        # Normalization layers
        self.spatial_norm1 = LayerNorm(dim, data_format="channels_first")
        self.spatial_norm2 = LayerNorm(dim, data_format="channels_first")
        self.freq_norm1 = LayerNorm(dim, data_format="channels_first")
        self.freq_norm2 = LayerNorm(dim, data_format="channels_first")
        
        # Feed-forward networks
        self.spatial_ffn = ConvFFN(dim)
        self.freq_ffn = ConvFFN(dim)
        
    def forward(self, spatial_feat, freq_feat):
        """
        Args:
            spatial_feat: (B, C, H, W)
            freq_feat: (B, C, H, W)
            
        Returns:
            enhanced_spatial: (B, C, H, W)
            enhanced_freq: (B, C, H, W)
        """
        # Bidirectional cross-attention
        enhanced_spatial = self.spatial_enhancer(spatial_feat, freq_feat)
        enhanced_freq = self.frequency_enhancer(freq_feat, spatial_feat)
        
        # Spatial stream: Residual -> Norm -> FFN -> Residual -> Norm
        spatial_out = spatial_feat + enhanced_spatial
        spatial_out = self.spatial_norm1(spatial_out)
        spatial_out = spatial_out + self.spatial_ffn(spatial_out)
        spatial_out = self.spatial_norm2(spatial_out)
        
        # Frequency stream: Residual -> Norm -> FFN -> Residual -> Norm  
        freq_out = freq_feat + enhanced_freq
        freq_out = self.freq_norm1(freq_out)
        freq_out = freq_out + self.freq_ffn(freq_out)
        freq_out = self.freq_norm2(freq_out)
        
        return spatial_out, freq_out
