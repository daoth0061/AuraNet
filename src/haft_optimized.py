"""
HAFT: Hierarchical Adaptive Frequency Transform Block (Optimized)
Implements sophisticated frequency-domain processing with parallelization optimizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import yaml
import os
from functools import partial


class FrequencyContextEncoder(nn.Module):
    """Small CNN heads for extracting context vectors from frequency patches."""
    
    def __init__(self, context_vector_dim=64):
        super().__init__()
        self.context_vector_dim = context_vector_dim
        
        # Dictionary to store encoders for different levels and types
        self.encoders = nn.ModuleDict()
        
        # Create encoders for different hierarchy levels (0-3) and types (mag/phase)
        for level in range(4):  # max possible levels
            for enc_type in ['mag', 'phase']:
                encoder_name = f'level_{level}_{enc_type}'
                # Small CNN that processes single-channel 2D patches
                self.encoders[encoder_name] = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(4),  # Reduce to 4x4
                    nn.Conv2d(16, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),  # Global average pooling
                    nn.Flatten(),
                    nn.Linear(32, context_vector_dim)
                )
    
    def forward(self, patches_dict):
        """Optimized version that processes all patches in parallel
        
        Args:
            patches_dict: Dictionary with keys like 'level_0_mag', 'level_1_phase', etc.
                          Each value is a tensor of shape (B*num_patches, 1, patch_H, patch_W)
            
        Returns:
            context_vectors_dict: Dictionary with the same keys, but values as context vectors
                                  of shape (B*num_patches, context_vector_dim)
        """
        result = {}
        # Process each type of patch in parallel
        for key, patches in patches_dict.items():
            if key in self.encoders:
                result[key] = self.encoders[key](patches)
            
        return result


class FilterPredictor(nn.Module):
    """MLPs to predict 1D radial filter profiles."""
    
    def __init__(self, input_dim, num_radial_bins=16, hidden_dim=128):
        super().__init__()
        
        # Separate MLPs for magnitude and phase
        self.mag_mlp_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_radial_bins),
            nn.Softmax(dim=-1)  # Normalize the filter
        )
        
        self.phase_mlp_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_radial_bins),
            nn.Tanh()  # Phase adjustments in [-1, 1]
        )
    
    def forward(self, mag_vector, phase_vector):
        """
        Args:
            mag_vector: (B, input_dim)
            phase_vector: (B, input_dim)
            
        Returns:
            w_mag_profile: (B, num_radial_bins)
            w_phase_profile: (B, num_radial_bins)
        """
        w_mag_profile = self.mag_mlp_head(mag_vector)
        w_phase_profile = self.phase_mlp_head(phase_vector)
        
        return w_mag_profile, w_phase_profile


def reconstruct_radial_filter(profile_1d, patch_size):
    """
    Reconstruct 2D radial filter from 1D profile using Chebyshev distance.
    Optimized to process batches more efficiently.
    
    Args:
        profile_1d: (B, num_radial_bins) 1D filter profile
        patch_size: (H, W) size of the patch
        
    Returns:
        filter_2d: (B, 1, H, W) 2D radial filter
    """
    B, num_bins = profile_1d.shape
    H, W = patch_size
    
    # Create coordinate grid on the same device as profile_1d
    device = profile_1d.device
    u = torch.arange(H, dtype=torch.float32, device=device).view(-1, 1)
    v = torch.arange(W, dtype=torch.float32, device=device).view(1, -1)
    
    # Calculate Chebyshev distance from center
    center_u, center_v = H / 2.0, W / 2.0
    d = torch.maximum(torch.abs(u - center_u), torch.abs(v - center_v))
    
    # Normalize distance to [0, num_bins-1] range
    max_d = torch.maximum(torch.tensor(center_u, device=device), torch.tensor(center_v, device=device))
    d_normalized = d / max_d * (num_bins - 1)
    d_indices = torch.floor(d_normalized).long().clamp(0, num_bins - 1)
    
    # Create 2D filter by indexing the 1D profile - optimized batch indexing
    # Reshape d_indices to (H*W) for efficient indexing
    d_indices_flat = d_indices.reshape(-1)
    
    # Create indices for each batch element
    batch_indices = torch.arange(B, device=device).view(B, 1).expand(B, H*W)
    bin_indices = d_indices_flat.expand(B, H*W)
    
    # Create filter using advanced indexing
    filter_values = profile_1d[batch_indices, bin_indices]
    filter_2d = filter_values.reshape(B, 1, H, W)
    
    return filter_2d


class HAFT(nn.Module):
    """Hierarchical Adaptive Frequency Transform Block with optimization."""
    
    def __init__(self, in_channels, num_haft_levels=3, num_radial_bins=16, 
                 context_vector_dim=64):
        super().__init__()
        self.in_channels = in_channels
        self.num_haft_levels = num_haft_levels
        self.num_radial_bins = num_radial_bins
        self.context_vector_dim = context_vector_dim
        
        # Core components
        self.freq_context_encoder = FrequencyContextEncoder(context_vector_dim)
        
        # Filter predictor input dimension depends on the hierarchical context, as per the plan
        # In the original HAFT module, the input dimension is num_haft_levels * context_vector_dim
        # Here we're handling mag and phase separately, so each stream gets this dimension
        filter_input_dim = num_haft_levels * context_vector_dim
        self.filter_predictor = FilterPredictor(filter_input_dim, num_radial_bins)
        
        # Level embeddings for hierarchical processing
        # Matches original HAFT implementation which used 2*context_vector_dim for level embedding
        self.level_projection = nn.Linear(1, 2 * context_vector_dim) # Projects level index to embedding
        
        # Output projection to maintain channel dimensions
        self.output_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Precalculate channel group sizes for processing
        self.channel_group_size = 16  # Process more channels at once
        
    def forward(self, F_in):
        """
        Args:
            F_in: (B, C, H, W) input feature map
            
        Returns:
            F_out: (B, C, H, W) enhanced feature map
        """
        B, C, H, W = F_in.shape
        
        # Process multiple channels in parallel in groups
        channel_groups = []
        for start_c in range(0, C, self.channel_group_size):
            end_c = min(start_c + self.channel_group_size, C)
            channel_group = F_in[:, start_c:end_c, :, :]  # (B, group_size, H, W)
            
            # Restructure for parallel processing
            group_results = self._process_channel_group(channel_group)
            channel_groups.append(group_results)
            
        # Concatenate all processed channel groups
        F_out = torch.cat(channel_groups, dim=1)
        
        # Apply final output projection
        F_out = self.output_proj(F_out)
        
        return F_out
    
    def _process_channel_group(self, channel_group):
        """Process a group of channels in parallel."""
        B, group_C, H, W = channel_group.shape
        
        # Extract patches at all levels in parallel
        patches_by_level = {}
        contexts_by_level = {}
        
        # Prepare tensors to store information for reconstruction
        patch_positions = []
        
        # Multi-Level Context Extraction
        for level in range(self.num_haft_levels):
            num_patches_per_side = 2 ** level
            patch_h = H // num_patches_per_side
            patch_w = W // num_patches_per_side
            
            if patch_h < 4 or patch_w < 4:  # Skip if patches become too small
                continue
            
            # Generate all patch positions at this level
            level_positions = []
            for i in range(num_patches_per_side):
                for j in range(num_patches_per_side):
                    start_h, end_h = i * patch_h, (i + 1) * patch_h
                    start_w, end_w = j * patch_w, (j + 1) * patch_w
                    level_positions.append((start_h, end_h, start_w, end_w))
            
            patch_positions.append(level_positions)
            
            # Extract patches for all channels at this level
            # For each position, extract patches from all channels
            level_mag_patches = []
            level_phase_patches = []
            
            for c in range(group_C):
                channel_data = channel_group[:, c:c+1, :, :]
                
                # Extract and process all patches for this channel at this level
                channel_patches = []
                for pos in level_positions:
                    start_h, end_h, start_w, end_w = pos
                    patch = channel_data[:, :, start_h:end_h, start_w:end_w]
                    channel_patches.append(patch)
                
                # Stack patches for this channel
                if channel_patches:
                    stacked_patches = torch.cat(channel_patches, dim=0)  # (num_patches*B, 1, patch_h, patch_w)
                    
                    # Apply FFT to all patches at once
                    patches_float = stacked_patches.squeeze(1).to(torch.float32)
                    patches_fft = torch.fft.fft2(patches_float)
                    magnitude = torch.abs(patches_fft).unsqueeze(1)
                    phase = torch.atan2(patches_fft.imag, patches_fft.real).unsqueeze(1)
                    
                    # Add to level collections
                    level_mag_patches.append(magnitude)
                    level_phase_patches.append(phase)
            
            # Process all magnitude patches at this level in parallel
            if level_mag_patches and level_phase_patches:
                # Create batch dictionary for parallel context encoding
                patches_dict = {
                    f'level_{level}_mag': torch.cat(level_mag_patches, dim=0),
                    f'level_{level}_phase': torch.cat(level_phase_patches, dim=0)
                }
                
                # Store for later use
                patches_by_level[level] = patches_dict
                
                # Get context vectors for all patches at this level
                level_contexts = self.freq_context_encoder(patches_dict)
                contexts_by_level[level] = level_contexts
        
        # Process and enhance all patches
        if not patches_by_level:
            return channel_group  # No patches were processed
            
        # Prepare for hierarchical filtering - process the deepest level
        deepest_level = max(patches_by_level.keys())
        deepest_positions = patch_positions[deepest_level]
        num_patches = len(deepest_positions)
        
        # Prepare to store enhanced patches
        enhanced_channels = []
        
        # Process each channel separately but enhance all patches in parallel
        for c in range(group_C):
            # Create empty tensor for this channel's result
            enhanced_channel = torch.zeros_like(channel_group[:, c:c+1, :, :])
            
            # Get original channel data
            channel_data = channel_group[:, c:c+1, :, :]
            
            # Generate filter for each patch position
            for patch_idx, position in enumerate(deepest_positions):
                start_h, end_h, start_w, end_w = position
                
                # Gather ancestral contexts for this patch
                ancestral_contexts_mag = []
                ancestral_contexts_phase = []
                
                # Add level embeddings and gather contexts from all levels
                for level in sorted(contexts_by_level.keys()):
                    # Map patch index from deepest level to this level
                    if level == deepest_level:
                        mapped_idx = patch_idx
                    else:
                        level_positions_current = patch_positions[level]
                        # Find which patch at this level contains the current deepest patch
                        mapped_idx = self._find_containing_patch(position, level_positions_current)
                    
                    # Get context vectors for this level and mapped index
                    level_contexts = contexts_by_level[level]
                    num_patches_at_level = len(patch_positions[level])

                    # CORRECTED INDEXING
                    channel_offset = c * num_patches_at_level * B
                    patch_offset = mapped_idx * B
                    start_idx = channel_offset + patch_offset
                    end_idx = start_idx + B
                    
                    # Add level embedding - ensure it's properly added to the context vectors
                    level_input = torch.tensor([[float(level)]], device=channel_data.device, dtype=torch.float32)
                    level_input = level_input.expand(B, -1)  # (B, 1)
                    level_emb = self.level_projection(level_input)  # (B, 2*context_dim)
                    
                    # Split level embedding for magnitude and phase components
                    # This matches the original HAFT implementation
                    mid_dim = level_emb.shape[1] // 2
                    level_emb_mag = level_emb[:, :mid_dim]  # (B, context_dim)
                    level_emb_phase = level_emb[:, mid_dim:]  # (B, context_dim)
                    
                    # Get context vectors for this level and mapped index
                    level_contexts = contexts_by_level[level]
                    num_patches_at_level = len(patch_positions[level])

                    # CORRECTED INDEXING - ensure we get the right batch of context vectors
                    channel_offset = c * num_patches_at_level * B
                    patch_offset = mapped_idx * B
                    start_idx = channel_offset + patch_offset
                    end_idx = start_idx + B
                    
                    # Ensure the context vectors exist for this level
                    if f'level_{level}_mag' in level_contexts and f'level_{level}_phase' in level_contexts:
                        try:
                            # Get context vectors
                            mag_ctx = level_contexts[f'level_{level}_mag'][start_idx:end_idx]
                            phase_ctx = level_contexts[f'level_{level}_phase'][start_idx:end_idx]
                            
                            # Validate shapes - context vectors should be (B, context_vector_dim)
                            if mag_ctx.shape[0] == B and mag_ctx.shape[1] == self.context_vector_dim:
                                # Enrich with level embedding
                                enriched_mag = mag_ctx + level_emb_mag
                                enriched_phase = phase_ctx + level_emb_phase
                                
                                # Add to ancestral contexts lists
                                ancestral_contexts_mag.append(enriched_mag)
                                ancestral_contexts_phase.append(enriched_phase)
                        except Exception as e:
                            # Skip this level if there's an issue with the context vectors
                            print(f"Error processing level {level} context: {e}")
                            continue
                
                # Concatenate all ancestral contexts - check if we have contexts to concatenate
                if not ancestral_contexts_mag or not ancestral_contexts_phase:
                    # Skip this patch if no context vectors were gathered
                    continue
                    
                # Concatenate contexts for each stream
                fused_mag_vector = torch.cat(ancestral_contexts_mag, dim=1)  # (B, num_levels*context_vector_dim)
                fused_phase_vector = torch.cat(ancestral_contexts_phase, dim=1)  # (B, num_levels*context_vector_dim)
                
                # Validate tensor shapes match the filter predictor's expected input
                expected_dim = self.num_haft_levels * self.context_vector_dim
                if fused_mag_vector.shape[1] != expected_dim:
                    # If dimensions don't match, likely because some levels were skipped
                    # Pad with zeros to reach the expected dimension
                    padding_size = expected_dim - fused_mag_vector.shape[1]
                    if padding_size > 0:
                        padding = torch.zeros((B, padding_size), device=fused_mag_vector.device, dtype=fused_mag_vector.dtype)
                        fused_mag_vector = torch.cat([fused_mag_vector, padding], dim=1)
                        fused_phase_vector = torch.cat([fused_phase_vector, padding], dim=1)
                
                # Predict filter profiles
                w_mag_profile, w_phase_profile = self.filter_predictor(fused_mag_vector, fused_phase_vector)
                
                # Get patch size for this level
                num_patches_per_side = 2 ** deepest_level
                patch_h = H // num_patches_per_side
                patch_w = W // num_patches_per_side
                
                # Reconstruct 2D filters
                mag_filter_2d = reconstruct_radial_filter(w_mag_profile, (patch_h, patch_w))
                phase_filter_2d = reconstruct_radial_filter(w_phase_profile, (patch_h, patch_w))
                
                # Get original patch
                original_patch = channel_data[:, :, start_h:end_h, start_w:end_w]
                
                # Apply frequency domain filtering
                original_patch_float32 = original_patch.to(torch.float32)
                patch_fft = torch.fft.fft2(original_patch_float32.squeeze(1))
                magnitude = torch.abs(patch_fft)
                phase = torch.atan2(patch_fft.imag, patch_fft.real)
                
                # Apply learned filters
                enhanced_magnitude = magnitude * mag_filter_2d.squeeze(1)
                enhanced_phase = phase + phase_filter_2d.squeeze(1)
                
                # Reconstruct enhanced patch
                enhanced_fft = enhanced_magnitude * torch.exp(1j * enhanced_phase)
                enhanced_patch = torch.real(torch.fft.ifft2(enhanced_fft)).unsqueeze(1)
                
                # Convert back to original dtype
                enhanced_patch = enhanced_patch.to(original_patch.dtype)
                
                # Place enhanced patch in result tensor
                enhanced_channel[:, :, start_h:end_h, start_w:end_w] = enhanced_patch
            
            enhanced_channels.append(enhanced_channel)
        
        # Stack all enhanced channels
        return torch.cat(enhanced_channels, dim=1)
    
    def _find_containing_patch(self, target_position, level_positions):
        """Find which patch at a higher level contains the target patch."""
        target_start_h, target_end_h, target_start_w, target_end_w = target_position
        target_center_h = (target_start_h + target_end_h) // 2
        target_center_w = (target_start_w + target_end_w) // 2
        
        for idx, (start_h, end_h, start_w, end_w) in enumerate(level_positions):
            if (start_h <= target_center_h < end_h and 
                start_w <= target_center_w < end_w):
                return idx
        
        # If no containing patch found, return the closest one
        return 0  # Default to first patch

"""
Helper function to create optimized HAFT module
"""
def create_optimized_haft(in_channels, num_haft_levels=3, num_radial_bins=16, context_vector_dim=64):
    return HAFT(
        in_channels=in_channels,
        num_haft_levels=num_haft_levels,
        num_radial_bins=num_radial_bins,
        context_vector_dim=context_vector_dim
    )
