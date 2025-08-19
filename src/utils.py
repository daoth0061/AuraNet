"""
AuraNet: A Dual-Stream Forensic Network for Face Manipulation Detection
Core utility modules and helper functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LayerNorm(nn.Module):
    """LayerNorm that supports channels_first data format."""
    
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """Global Response Normalization layer."""
    
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class Block(nn.Module):
    """ConvNeXt V2 Block."""
    
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CBAMChannelAttention(nn.Module):
    """Channel attention module from CBAM."""
    
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        
        out = avg_out + max_out
        return torch.sigmoid(out).view(b, c, 1, 1)


class CBAMSpatialAttention(nn.Module):
    """Spatial attention module from CBAM."""
    
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                             padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return torch.sigmoid(x)


def create_grid_like(tensor):
    """Create a coordinate grid like the input tensor."""
    B, C, H, W = tensor.shape
    y, x = torch.meshgrid(torch.arange(H, dtype=tensor.dtype, device=tensor.device),
                         torch.arange(W, dtype=tensor.dtype, device=tensor.device),
                         indexing='ij')
    grid = torch.stack([x, y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
    return grid


def normalize_grid(grid):
    """Normalize grid coordinates to [-1, 1] range."""
    B, _, H, W = grid.shape
    grid = grid.clone()
    grid[:, 0] = 2.0 * grid[:, 0] / (W - 1) - 1.0  # x coordinate
    grid[:, 1] = 2.0 * grid[:, 1] / (H - 1) - 1.0  # y coordinate
    return grid


class ContinuousPositionalBias(nn.Module):
    """Continuous positional bias using an MLP."""
    
    def __init__(self, dim, heads, hidden_dim=128):
        super().__init__()
        self.heads = heads
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, heads)
        )

    def forward(self, query_coords, key_coords):
        """
        Args:
            query_coords: (B, H_q*W_q, 2)
            key_coords: (B, H_k*W_k, 2)
        Returns:
            bias: (B, heads, H_q*W_q, H_k*W_k)
        """
        B, N_q, _ = query_coords.shape
        N_k = key_coords.shape[1]
        
        # Compute relative distances
        query_coords = query_coords.unsqueeze(2)  # (B, N_q, 1, 2)
        key_coords = key_coords.unsqueeze(1)      # (B, 1, N_k, 2)
        
        rel_coords = query_coords - key_coords    # (B, N_q, N_k, 2)
        
        # Apply MLP to get bias values
        bias = self.mlp(rel_coords)  # (B, N_q, N_k, heads)
        bias = bias.permute(0, 3, 1, 2)  # (B, heads, N_q, N_k)
        
        return bias
