"""
HAFT: Hierarchical Adaptive Frequency Transform Block
Implements sophisticated frequency-domain processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
    
    def forward(self, patch, level, enc_type):
        """
        Args:
            patch: (B, 1, patch_H, patch_W) single-channel frequency patch
            level: int, hierarchy level
            enc_type: str, 'mag' or 'phase'
            
        Returns:
            context_vector: (B, context_vector_dim)
        """
        encoder_name = f'level_{level}_{enc_type}'
        if encoder_name not in self.encoders:
            raise ValueError(f"Encoder {encoder_name} not found")
        
        return self.encoders[encoder_name](patch)


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
    
    # Create 2D filter by indexing the 1D profile
    filter_2d = torch.zeros(B, 1, H, W, device=device)
    for b in range(B):
        filter_2d[b, 0] = profile_1d[b][d_indices]
    
    return filter_2d


class HAFT(nn.Module):
    """Hierarchical Adaptive Frequency Transform Block."""
    
    def __init__(self, in_channels, num_haft_levels=3, num_radial_bins=16, 
                 context_vector_dim=64):
        super().__init__()
        self.in_channels = in_channels
        self.num_haft_levels = num_haft_levels
        self.num_radial_bins = num_radial_bins
        self.context_vector_dim = context_vector_dim
        
        # Core components
        self.freq_context_encoder = FrequencyContextEncoder(context_vector_dim)
        
        # Filter predictor input dimension depends on the hierarchical context
        filter_input_dim = num_haft_levels * context_vector_dim
        self.filter_predictor = FilterPredictor(filter_input_dim, num_radial_bins)
        
        # Level embeddings for hierarchical processing - use Linear instead of Embedding
        self.level_projection = nn.Linear(1, 2 * context_vector_dim)
        
        # Output projection to maintain channel dimensions
        self.output_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, F_in):
        """
        Args:
            F_in: (B, C, H, W) input feature map
            
        Returns:
            F_out: (B, C, H, W) enhanced feature map
        """
        B, C, H, W = F_in.shape
        
        # MEMORY OPTIMIZATION: Sequential channel processing to reduce memory usage
        # Instead of processing all channels at once, process them sequentially
        enhanced_channels = []
        
        # Process channels in smaller groups to reduce peak memory usage
        channel_group_size = min(8, C)  # Process 8 channels at a time max
        
        for start_c in range(0, C, channel_group_size):
            end_c = min(start_c + channel_group_size, C)
            
            # Extract channel group
            channel_group = F_in[:, start_c:end_c, :, :]  # (B, group_size, H, W)
            
            # Process each channel in the group
            group_enhanced = []
            for c in range(channel_group.size(1)):
                channel_features = channel_group[:, c:c+1, :, :]  # (B, 1, H, W)
                enhanced_channel = self._process_channel(channel_features)
                group_enhanced.append(enhanced_channel)
                
                # Clear intermediate variables to save memory
                del channel_features
            
            # Concatenate group results
            group_result = torch.cat(group_enhanced, dim=1)
            enhanced_channels.append(group_result)
            
            # Clear group variables to save memory
            del channel_group, group_enhanced, group_result
            
            # Clear CUDA cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate all enhanced channels
        F_out = torch.cat(enhanced_channels, dim=1)  # (B, C, H, W)
        
        # Clear intermediate list
        del enhanced_channels
        
        # Output projection
        F_out = self.output_proj(F_out)
        
        return F_out
    
    def _process_channel(self, channel_features):
        """Process a single channel through the HAFT mechanism."""
        B, _, H, W = channel_features.shape
        
        # Multi-Level Context Extraction
        hierarchical_contexts = []
        
        for level in range(self.num_haft_levels):
            num_patches_per_side = 2 ** level
            patch_h = H // num_patches_per_side
            patch_w = W // num_patches_per_side
            
            if patch_h < 4 or patch_w < 4:  # Skip if patches become too small
                continue
                
            level_contexts = []
            
            # Extract patches at this level
            for i in range(num_patches_per_side):
                for j in range(num_patches_per_side):
                    # Extract patch
                    start_h, end_h = i * patch_h, (i + 1) * patch_h
                    start_w, end_w = j * patch_w, (j + 1) * patch_w
                    patch = channel_features[:, :, start_h:end_h, start_w:end_w]
                    
                    # Apply FFT
                    patch_fft = torch.fft.fft2(patch.squeeze(1))
                    magnitude = torch.abs(patch_fft).unsqueeze(1)
                    # Use atan2 instead of angle to avoid CUDA kernel compilation issues
                    phase = torch.atan2(patch_fft.imag, patch_fft.real).unsqueeze(1)
                    
                    # Get context vectors
                    cv_mag = self.freq_context_encoder(magnitude, level, 'mag')
                    cv_phase = self.freq_context_encoder(phase, level, 'phase')
                    
                    # Concatenate magnitude and phase contexts
                    cv_patch = torch.cat([cv_mag, cv_phase], dim=1)  # (B, 2*context_vector_dim)
                    level_contexts.append(cv_patch)
            
            if level_contexts:
                # Stack contexts for this level
                level_context_tensor = torch.stack(level_contexts, dim=1)  # (B, num_patches, 2*context_vector_dim)
                hierarchical_contexts.append(level_context_tensor)
        
        if not hierarchical_contexts:
            return channel_features
        
        # Hierarchical Filtering
        enhanced_patches = []
        deepest_level = len(hierarchical_contexts) - 1
        
        if deepest_level >= 0:
            deepest_contexts = hierarchical_contexts[deepest_level]  # (B, num_patches, 2*context_vector_dim)
            num_patches_per_side = 2 ** deepest_level
            patch_h = H // num_patches_per_side
            patch_w = W // num_patches_per_side
            
            for patch_idx in range(deepest_contexts.shape[1]):
                # Gather ancestral contexts
                ancestral_contexts = []
                
                for level in range(len(hierarchical_contexts)):
                    level_contexts = hierarchical_contexts[level]
                    
                    # Map patch index to the appropriate index at this level
                    level_patches_per_side = 2 ** level
                    if level_patches_per_side == 0:  # Level 0 (global context)
                        mapped_patch_idx = 0
                    else:
                        # Calculate which patch at this level corresponds to the deepest level patch
                        scale_factor = num_patches_per_side // level_patches_per_side
                        if scale_factor <= 1:
                            mapped_patch_idx = min(patch_idx, level_contexts.shape[1] - 1)
                        else:
                            row_idx = (patch_idx // num_patches_per_side) // scale_factor
                            col_idx = (patch_idx % num_patches_per_side) // scale_factor
                            mapped_patch_idx = row_idx * level_patches_per_side + col_idx
                            mapped_patch_idx = min(mapped_patch_idx, level_contexts.shape[1] - 1)
                    
                    # Add level embedding - use Linear layer instead of Embedding
                    level_input = torch.tensor([[float(level)]], device=channel_features.device, dtype=torch.float32)
                    level_input = level_input.expand(channel_features.shape[0], -1)  # (B, 1)
                    level_emb = self.level_projection(level_input)  # (B, 2*context_dim)
                    enriched_context = level_contexts[:, mapped_patch_idx] + level_emb
                    ancestral_contexts.append(enriched_context)
                
                # Concatenate all ancestral contexts
                fused_context_vector = torch.cat(ancestral_contexts, dim=1)  # (B, total_context_dim)
                
                # Split into magnitude and phase vectors
                mid_dim = fused_context_vector.shape[1] // 2
                fused_mag_vector = fused_context_vector[:, :mid_dim]
                fused_phase_vector = fused_context_vector[:, mid_dim:]
                
                # Predict filter profiles
                w_mag_profile, w_phase_profile = self.filter_predictor(fused_mag_vector, fused_phase_vector)
                
                # Reconstruct 2D filters
                mag_filter_2d = reconstruct_radial_filter(w_mag_profile, (patch_h, patch_w))
                phase_filter_2d = reconstruct_radial_filter(w_phase_profile, (patch_h, patch_w))
                
                # Apply filters to the corresponding patch
                i = patch_idx // num_patches_per_side
                j = patch_idx % num_patches_per_side
                start_h, end_h = i * patch_h, (i + 1) * patch_h
                start_w, end_w = j * patch_w, (j + 1) * patch_w
                
                original_patch = channel_features[:, :, start_h:end_h, start_w:end_w]
                
                # Apply frequency domain filtering
                patch_fft = torch.fft.fft2(original_patch.squeeze(1))
                magnitude = torch.abs(patch_fft)
                # Use atan2 instead of angle to avoid CUDA kernel compilation issues
                phase = torch.atan2(patch_fft.imag, patch_fft.real)
                
                # Apply learned filters
                enhanced_magnitude = magnitude * mag_filter_2d.squeeze(1)
                enhanced_phase = phase + phase_filter_2d.squeeze(1)
                
                # Reconstruct enhanced patch
                enhanced_fft = enhanced_magnitude * torch.exp(1j * enhanced_phase)
                enhanced_patch = torch.real(torch.fft.ifft2(enhanced_fft)).unsqueeze(1)
                
                # Store patch info for reconstruction
                enhanced_patches.append({
                    'patch': enhanced_patch,
                    'position': (start_h, end_h, start_w, end_w)
                })
        
        # Reconstruct the full feature map
        if enhanced_patches:
            enhanced_channel = torch.zeros_like(channel_features)
            for patch_info in enhanced_patches:
                start_h, end_h, start_w, end_w = patch_info['position']
                enhanced_channel[:, :, start_h:end_h, start_w:end_w] = patch_info['patch']
            return enhanced_channel
        else:
            return channel_features
