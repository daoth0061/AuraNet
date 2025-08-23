"""
Cross-Fusion Block Implementation
Implements Deformable and Standard Cross-Attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import LayerNorm, create_grid_like, normalize_grid, ContinuousPositionalBias
import yaml
import os


class DeformableCrossAttention(nn.Module):
    """Deformable Cross-Attention for high-resolution stages (2 & 3)."""
    
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
        
        # Query, Key, Value projections
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_k = nn.Conv2d(dim, inner_dim, 1, bias=False)  
        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias=False)
        
        # Offset prediction network
        self.to_offsets = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, num_offsets * 2, 1)  # 2D offsets
        )
        
        # Relative positional bias
        self.rel_pos_bias = ContinuousPositionalBias(dim, heads)
        
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
        
        # Generate queries and predict offsets
        q = self.to_q(query_map)  # (B, C, H, W)
        offsets = self.to_offsets(q)  # (B, num_offsets*2, H, W)
        
        # Reshape offsets
        offsets = offsets.view(B, self.num_offsets, 2, H, W)  # (B, num_offsets, 2, H, W)
        
        # Create base sampling grid
        base_grid = create_grid_like(query_map)  # (B, 2, H, W)
        base_grid = normalize_grid(base_grid)    # Normalize to [-1, 1]
        
        # Apply offsets to create sampling locations
        # For simplicity, we'll use the first offset for each query point
        offset = offsets[:, 0]  # (B, 2, H, W) - use first offset
        
        # Scale offsets (they should be small perturbations)
        offset = offset * self.offset_scale  
        
        # Create sampling grid
        vgrid = base_grid + offset  # (B, 2, H, W)
        vgrid = vgrid.permute(0, 2, 3, 1)  # (B, H, W, 2)
        
        # Sample key-value features
        kv_sampled = F.grid_sample(kv_map, vgrid, mode='bilinear', 
                                   padding_mode='border', align_corners=True)  # (B, C, H, W)
        
        # Generate keys and values from sampled features
        k = self.to_k(kv_sampled)  # (B, C, H, W)
        v = self.to_v(kv_sampled)  # (B, C, H, W)
        
        # Reshape for attention computation
        q = q.view(B, self.heads, C // self.heads, H * W).transpose(-2, -1)  # (B, heads, HW, C//heads)
        k = k.view(B, self.heads, C // self.heads, H * W).transpose(-2, -1)  # (B, heads, HW, C//heads)
        v = v.view(B, self.heads, C // self.heads, H * W).transpose(-2, -1)  # (B, heads, HW, C//heads)
        
        # MEMORY OPTIMIZATION: Use memory-efficient attention
        try:
            # Try to use Flash Attention if available (PyTorch 2.0+)
            try:
                # New API
                with torch.nn.attention.sdpa_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                    # Reshape for proper stride: (B, HW, heads, C//heads) -> (B*HW, heads, C//heads)
                    B, heads, HW, dim = q.shape
                    q_sdpa = q.transpose(1, 2).contiguous().view(B * HW, heads, dim)  
                    k_sdpa = k.transpose(1, 2).contiguous().view(B * HW, heads, dim)   
                    v_sdpa = v.transpose(1, 2).contiguous().view(B * HW, heads, dim)   
                    
                    # Use scaled_dot_product_attention for memory efficiency
                    out_sdpa = F.scaled_dot_product_attention(
                        q_sdpa, k_sdpa, v_sdpa,
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=False
                    )
                    # Reshape back: (B*HW, heads, C//heads) -> (B, heads, HW, C//heads)
                    out = out_sdpa.view(B, HW, heads, dim).transpose(1, 2)
            except AttributeError:
                # Fallback to old API
                with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                    # Reshape for proper stride: (B, HW, heads, C//heads) -> (B*HW, heads, C//heads)
                    B, heads, HW, dim = q.shape
                    q_sdpa = q.transpose(1, 2).contiguous().view(B * HW, heads, dim)  
                    k_sdpa = k.transpose(1, 2).contiguous().view(B * HW, heads, dim)   
                    v_sdpa = v.transpose(1, 2).contiguous().view(B * HW, heads, dim)   
                    
                    # Use scaled_dot_product_attention for memory efficiency
                    out_sdpa = F.scaled_dot_product_attention(
                        q_sdpa, k_sdpa, v_sdpa,
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=False
                    )
                    # Reshape back: (B*HW, heads, C//heads) -> (B, heads, HW, C//heads)
                    out = out_sdpa.view(B, HW, heads, dim).transpose(1, 2)
        except:
            # Fallback to standard attention computation
            # Compute attention scores
            attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, HW, HW)
            
            # Add relative positional bias
            # Create coordinate grids for bias computation
            y_coords, x_coords = torch.meshgrid(
                torch.arange(H, device=query_map.device),
                torch.arange(W, device=query_map.device),
                indexing='ij'
            )
            query_coords = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=-1)  # (HW, 2)
            query_coords = query_coords.unsqueeze(0).repeat(B, 1, 1).float()  # (B, HW, 2)
            
            # For simplicity, use the same coordinates for keys (this can be refined)
            key_coords = query_coords  
            
            rel_bias = self.rel_pos_bias(query_coords, key_coords)  # (B, heads, HW, HW)
            attn = attn + rel_bias
            
            # Apply softmax
            attn = F.softmax(attn, dim=-1)
            
            # Apply attention to values
            out = attn @ v  # (B, heads, HW, C//heads)
        
        # Reshape back to feature map format
        out = out.transpose(-2, -1).reshape(B, C, H, W)  # (B, C, H, W)
        
        # Final projection
        out = self.to_out(out)
        
        return out


class StandardCrossAttention(nn.Module):
    """Standard Cross-Attention for low-resolution stages (4 & 5)."""
    
    def __init__(self, dim, heads=8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        inner_dim = dim
        
        # Linear projections
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
        
        # Flatten to sequences
        query_seq = query_map.flatten(2).transpose(1, 2)  # (B, H*W, C)
        kv_seq = kv_map.flatten(2).transpose(1, 2)        # (B, H*W, C)
        
        # Generate Q, K, V
        q = self.to_q(query_seq)  # (B, H*W, C)
        k = self.to_k(kv_seq)     # (B, H*W, C)
        v = self.to_v(kv_seq)     # (B, H*W, C)
        
        # Reshape for multi-head attention
        q = q.view(B, H*W, self.heads, C // self.heads).transpose(1, 2)  # (B, heads, H*W, C//heads)
        k = k.view(B, H*W, self.heads, C // self.heads).transpose(1, 2)  # (B, heads, H*W, C//heads)
        v = v.view(B, H*W, self.heads, C // self.heads).transpose(1, 2)  # (B, heads, H*W, C//heads)
        
        # MEMORY OPTIMIZATION: Use memory-efficient attention
        try:
            # Try to use Flash Attention if available (PyTorch 2.0+)
            try:
                # New API
                with torch.nn.attention.sdpa_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                    # Use scaled_dot_product_attention for memory efficiency
                    out = F.scaled_dot_product_attention(
                        q.transpose(1, 2),  # (B, H*W, heads, C//heads)
                        k.transpose(1, 2),  # (B, H*W, heads, C//heads)
                        v.transpose(1, 2),  # (B, H*W, heads, C//heads)
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=False
                    )
                    out = out.transpose(1, 2)  # (B, heads, H*W, C//heads)
            except AttributeError:
                # Fallback to old API
                with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                    # Use scaled_dot_product_attention for memory efficiency
                    out = F.scaled_dot_product_attention(
                        q.transpose(1, 2),  # (B, H*W, heads, C//heads)
                        k.transpose(1, 2),  # (B, H*W, heads, C//heads)
                        v.transpose(1, 2),  # (B, H*W, heads, C//heads)
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=False
                    )
                    out = out.transpose(1, 2)  # (B, heads, H*W, C//heads)
        except:
            # Fallback to standard attention computation
            # Compute attention
            attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, H*W, H*W)
            attn = F.softmax(attn, dim=-1)
            
            # Apply attention to values
            out = attn @ v  # (B, heads, H*W, C//heads)
        
        # Reshape back
        out = out.transpose(1, 2).reshape(B, H*W, C)  # (B, H*W, C)
        
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
