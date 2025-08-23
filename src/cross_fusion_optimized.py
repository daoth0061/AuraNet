"""
Cross-Fusion Block Implementation (Optimized)
Implements Deformable and Standard Cross-Attention mechanisms with parallel processing optimizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
import sys
from functools import partial

# Add src directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from utils import LayerNorm, create_grid_like, normalize_grid, ContinuousPositionalBias


class ParallelDeformableCrossAttention(nn.Module):
    """Optimized Deformable Cross-Attention for high-resolution stages (2 & 3)."""
    
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

        # Cache grid coordinates for efficiency
        self._cached_coords = {}
        
    def forward(self, query_maps):
        """Process multiple query maps in parallel
        
        Args:
            query_maps: List of (query_map, kv_map) pairs
                Each query_map and kv_map has shape (B, C, H, W)
            
        Returns:
            outputs: List of outputs with same shape as inputs
        """
        if not isinstance(query_maps, list):
            # If single pair is provided, convert to list
            return self._forward_single(*query_maps)
            
        results = []
        batch_queries = []
        batch_kvs = []
        batch_indices = []
        
        # Prepare batched processing
        for i, (query_map, kv_map) in enumerate(query_maps):
            batch_queries.append(query_map)
            batch_kvs.append(kv_map)
            batch_indices.append(i)
            
        # Batch all queries and kvs
        batched_queries = torch.cat(batch_queries, dim=0)
        batched_kvs = torch.cat(batch_kvs, dim=0)
        
        # Process in batch
        batched_output = self._forward_single(batched_queries, batched_kvs)
        
        # Split results back into original batch sizes
        start_idx = 0
        for i, (query_map, _) in enumerate(query_maps):
            batch_size = query_map.size(0)
            results.append(batched_output[start_idx:start_idx + batch_size])
            start_idx += batch_size
            
        return results
    
    def _forward_single(self, query_map, kv_map):
        """Original forward pass for a single query/kv pair"""
        B, C, H, W = query_map.shape
        
        # Generate queries and predict offsets
        q = self.to_q(query_map)  # (B, C, H, W)
        offsets = self.to_offsets(q)  # (B, num_offsets*2, H, W)
        
        # Reshape offsets
        offsets = offsets.view(B, self.num_offsets, 2, H, W)  # (B, num_offsets, 2, H, W)
        
        # Create base sampling grid - use cached grid if available
        grid_key = f"{H}_{W}_{query_map.device}"
        if grid_key not in self._cached_coords:
            base_grid = create_grid_like(query_map)  # (B, 2, H, W)
            self._cached_coords[grid_key] = normalize_grid(base_grid)
        
        base_grid = self._cached_coords[grid_key]
        
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
        
        # Use Flash Attention if available
        try:
            # Try to use Flash Attention if available (PyTorch 2.0+)
            if hasattr(F, 'scaled_dot_product_attention'):
                out = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False
                )
            else:
                # Compute attention scores
                attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, HW, HW)
                
                # Add relative positional bias
                rel_coords_key = f"rel_pos_{H}_{W}_{query_map.device}"
                if rel_coords_key not in self._cached_coords:
                    # Create coordinate grids for bias computation
                    y_coords, x_coords = torch.meshgrid(
                        torch.arange(H, device=query_map.device),
                        torch.arange(W, device=query_map.device),
                        indexing='ij'
                    )
                    query_coords = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=-1)  # (HW, 2)
                    query_coords = query_coords.unsqueeze(0).float()  # (1, HW, 2)
                    self._cached_coords[rel_coords_key] = query_coords
                
                query_coords = self._cached_coords[rel_coords_key].repeat(B, 1, 1)  # (B, HW, 2)
                key_coords = query_coords  # For simplicity, use same coords
                
                rel_bias = self.rel_pos_bias(query_coords, key_coords)  # (B, heads, HW, HW)
                attn = attn + rel_bias
                
                # Apply softmax
                attn = F.softmax(attn, dim=-1)
                
                # Apply attention to values
                out = attn @ v  # (B, heads, HW, C//heads)
        except Exception as e:
            # Fallback to standard attention computation
            # Compute attention scores
            attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, HW, HW)
            attn = F.softmax(attn, dim=-1)
            out = attn @ v  # (B, heads, HW, C//heads)
        
        # Reshape back to feature map format
        out = out.transpose(-2, -1).reshape(B, C, H, W)  # (B, C, H, W)
        
        # Final projection
        out = self.to_out(out)
        
        return out


class OptimizedStandardCrossAttention(nn.Module):
    """Optimized Standard Cross-Attention for low-resolution stages (4 & 5)."""
    
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
        
    def forward(self, query_maps):
        """Process multiple query maps in parallel
        
        Args:
            query_maps: List of (query_map, kv_map) pairs
                Each query_map and kv_map has shape (B, C, H, W)
            
        Returns:
            outputs: List of outputs with same shape as inputs
        """
        if not isinstance(query_maps, list):
            # If single pair is provided, convert to list
            return self._forward_single(*query_maps)
            
        results = []
        batch_queries = []
        batch_kvs = []
        shapes = []
        
        # Prepare batched processing
        for query_map, kv_map in query_maps:
            B, C, H, W = query_map.shape
            batch_queries.append(query_map)
            batch_kvs.append(kv_map)
            shapes.append((B, C, H, W))
            
        # Batch all queries and kvs
        batched_queries = torch.cat(batch_queries, dim=0)
        batched_kvs = torch.cat(batch_kvs, dim=0)
        
        # Process in batch
        batched_output = self._forward_single(batched_queries, batched_kvs)
        
        # Split results back into original batch sizes
        start_idx = 0
        for i, (B, C, H, W) in enumerate(shapes):
            results.append(batched_output[start_idx:start_idx + B])
            start_idx += B
            
        return results
    
    def _forward_single(self, query_map, kv_map):
        """Original forward pass for a single query/kv pair"""
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
        
        # Use Flash Attention if available
        try:
            if hasattr(F, 'scaled_dot_product_attention'):
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False
                )
            else:
                # Fallback to standard attention
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = F.softmax(attn, dim=-1)
                out = attn @ v
        except Exception as e:
            # Fallback to standard attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = attn @ v
        
        # Reshape back
        out = out.transpose(1, 2).reshape(B, H*W, C)  # (B, H*W, C)
        
        # Final projection
        out = self.to_out(out)  # (B, H*W, C)
        
        # Reshape back to feature map
        out = out.transpose(1, 2).view(B, C, H, W)  # (B, C, H, W)
        
        return out


class ParallelConvFFN(nn.Module):
    """Optimized Convolutional Feed-Forward Network."""
    
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
        
    def forward(self, x_list):
        """Process multiple inputs in parallel
        
        Args:
            x_list: List of input tensors, each of shape (B, C, H, W)
            
        Returns:
            outputs: List of output tensors with same shape as inputs
        """
        if not isinstance(x_list, list):
            # If single tensor is provided
            return self.net(x_list)
            
        # Concatenate all inputs
        shapes = [x.shape for x in x_list]
        batch_sizes = [shape[0] for shape in shapes]
        total_batch = sum(batch_sizes)
        
        # Process all inputs at once
        combined = torch.cat(x_list, dim=0)
        combined_output = self.net(combined)
        
        # Split back into original batches
        outputs = []
        start_idx = 0
        for batch_size in batch_sizes:
            outputs.append(combined_output[start_idx:start_idx + batch_size])
            start_idx += batch_size
            
        return outputs


class OptimizedCrossFusionBlock(nn.Module):
    """Optimized Cross-Fusion Block for bidirectional stream dialogue."""
    
    def __init__(self, dim, use_deformable=True, heads=8, dropout=0.0):
        super().__init__()
        self.dim = dim
        
        # Attention modules
        if use_deformable:
            self.spatial_enhancer = ParallelDeformableCrossAttention(dim, heads)
            self.frequency_enhancer = ParallelDeformableCrossAttention(dim, heads)
        else:
            self.spatial_enhancer = OptimizedStandardCrossAttention(dim, heads)
            self.frequency_enhancer = OptimizedStandardCrossAttention(dim, heads)
        
        # Normalization layers
        self.spatial_norm1 = LayerNorm(dim, data_format="channels_first")
        self.spatial_norm2 = LayerNorm(dim, data_format="channels_first")
        self.freq_norm1 = LayerNorm(dim, data_format="channels_first")
        self.freq_norm2 = LayerNorm(dim, data_format="channels_first")
        
        # Feed-forward networks
        self.spatial_ffn = ParallelConvFFN(dim)
        self.freq_ffn = ParallelConvFFN(dim)
        
    def forward(self, spatial_feat, freq_feat):
        """
        Args:
            spatial_feat: (B, C, H, W)
            freq_feat: (B, C, H, W)
            
        Returns:
            enhanced_spatial: (B, C, H, W)
            enhanced_freq: (B, C, H, W)
        """
        # Process both cross-attention operations in parallel
        enhanced_spatial = self.spatial_enhancer((spatial_feat, freq_feat))
        enhanced_freq = self.frequency_enhancer((freq_feat, spatial_feat))
        
        # Apply residual connection
        spatial_out = spatial_feat + enhanced_spatial
        freq_out = freq_feat + enhanced_freq
        
        # Apply normalization
        spatial_out = self.spatial_norm1(spatial_out)
        freq_out = self.freq_norm1(freq_out)
        
        # Apply FFN in parallel
        spatial_ffn_out = self.spatial_ffn(spatial_out)
        freq_ffn_out = self.freq_ffn(freq_out)
        
        # Apply residual and final norm
        spatial_out = spatial_out + spatial_ffn_out
        freq_out = freq_out + freq_ffn_out
        
        spatial_out = self.spatial_norm2(spatial_out)
        freq_out = self.freq_norm2(freq_out)
        
        return spatial_out, freq_out

"""
Helper function to create optimized cross fusion block
"""
def create_optimized_cross_fusion(dim, use_deformable=True, heads=8, dropout=0.0):
    return OptimizedCrossFusionBlock(
        dim=dim,
        use_deformable=use_deformable,
        heads=heads,
        dropout=dropout
    )
