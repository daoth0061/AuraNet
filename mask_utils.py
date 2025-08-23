"""
Mask Utilities for FCMAE

This utility module provides functions for working with masks in various formats
for FCMAE pre-training.
"""

import torch
import numpy as np
from typing import Tuple, Union

def expand_mask_to_spatial(
    patch_mask: torch.Tensor, 
    height: int, 
    width: int, 
    patch_size: int
) -> torch.Tensor:
    """
    Convert a patch-level mask to a spatial (pixel-level) mask.
    
    Args:
        patch_mask: Patch-level mask tensor of shape (L,) where L is the number of patches
        height: Image height
        width: Image width
        patch_size: Size of each patch
        
    Returns:
        spatial_mask: Spatial mask tensor of shape (1, H, W)
    """
    # Calculate grid dimensions
    mask_h = height // patch_size
    mask_w = width // patch_size
    
    # Reshape patch mask to 2D grid
    grid_mask = patch_mask.reshape(mask_h, mask_w)
    
    # Expand each cell to patch_size x patch_size
    spatial_mask = torch.zeros(1, height, width, device=patch_mask.device)
    
    for i in range(mask_h):
        for j in range(mask_w):
            if grid_mask[i, j] > 0:  # If this patch is masked
                # Fill the corresponding area in spatial mask
                spatial_mask[:, 
                          i*patch_size:(i+1)*patch_size, 
                          j*patch_size:(j+1)*patch_size] = 1.0
    
    return spatial_mask

def create_random_mask(
    shape: Tuple[int, int, int], 
    mask_ratio: float, 
    patch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a random mask for FCMAE pre-training.
    
    Args:
        shape: Tuple of (B, H, W) - batch size, height, width
        mask_ratio: Ratio of patches to mask
        patch_size: Size of each patch
        
    Returns:
        patch_mask: Patch-level mask of shape (B, L)
        spatial_mask: Spatial mask of shape (B, 1, H, W)
    """
    B, H, W = shape
    
    # Calculate patch grid dimensions
    mask_h = H // patch_size
    mask_w = W // patch_size
    L = mask_h * mask_w
    
    # Create patch-level mask (B, L)
    num_masked = int(mask_ratio * L)
    patch_mask = torch.zeros(B, L)
    
    for b in range(B):
        # Randomly select patches to mask
        masked_indices = torch.randperm(L)[:num_masked]
        patch_mask[b, masked_indices] = 1.0
    
    # Create spatial mask (B, 1, H, W)
    spatial_mask = torch.zeros(B, 1, H, W)
    
    for b in range(B):
        # Reshape to 2D grid
        grid_mask = patch_mask[b].reshape(mask_h, mask_w)
        
        # Expand each cell
        for i in range(mask_h):
            for j in range(mask_w):
                if grid_mask[i, j] > 0:
                    spatial_mask[b, 0, 
                               i*patch_size:(i+1)*patch_size, 
                               j*patch_size:(j+1)*patch_size] = 1.0
    
    return patch_mask, spatial_mask

def apply_mask_to_image(
    image: torch.Tensor, 
    mask: torch.Tensor, 
    mask_type: str = 'spatial'
) -> torch.Tensor:
    """
    Apply a mask to an image tensor.
    
    Args:
        image: Image tensor of shape (B, C, H, W)
        mask: Mask tensor - either patch-level (B, L) or spatial (B, 1, H, W)
        mask_type: 'spatial' or 'patch' to indicate mask type
        
    Returns:
        masked_image: Masked image tensor of shape (B, C, H, W)
    """
    B, C, H, W = image.shape
    
    if mask_type == 'patch':
        # Convert patch-level mask to spatial
        patch_size = H // int(np.sqrt(mask.shape[1]))
        mask_h = H // patch_size
        mask_w = W // patch_size
        
        spatial_mask = torch.zeros(B, 1, H, W, device=image.device)
        
        for b in range(B):
            # Reshape to 2D grid
            grid_mask = mask[b].reshape(mask_h, mask_w)
            
            # Expand each cell
            for i in range(mask_h):
                for j in range(mask_w):
                    if grid_mask[i, j] > 0:
                        spatial_mask[b, 0, 
                                   i*patch_size:(i+1)*patch_size, 
                                   j*patch_size:(j+1)*patch_size] = 1.0
        
        mask = spatial_mask
    
    # Apply mask (1 = masked, 0 = visible)
    masked_image = image * (1 - mask)
    
    return masked_image

if __name__ == "__main__":
    # Example usage
    img_size = (128, 128)
    patch_size = 32
    mask_ratio = 0.75
    
    # Create a sample image
    img = torch.rand(1, 3, *img_size)
    
    # Create random mask
    patch_mask, spatial_mask = create_random_mask((1, *img_size), mask_ratio, patch_size)
    
    # Apply mask to image
    masked_img = apply_mask_to_image(img, spatial_mask, 'spatial')
    
    print(f"Image shape: {img.shape}")
    print(f"Patch mask shape: {patch_mask.shape}")
    print(f"Spatial mask shape: {spatial_mask.shape}")
    print(f"Masked image shape: {masked_img.shape}")
    print(f"Mask ratio: {patch_mask.sum().item() / patch_mask.numel():.2f}")
    
    print("✅ Mask utilities working correctly!")
