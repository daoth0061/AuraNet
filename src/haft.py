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
    device = profile_1d.device
    
    # Create coordinate grid
    u = torch.arange(H, dtype=torch.float32, device=device).view(-1, 1)
    v = torch.arange(W, dtype=torch.float32, device=device).view(1, -1)
    
    # Calculate Chebyshev distance from center
    center_u, center_v = H / 2.0, W / 2.0
    d = torch.maximum(torch.abs(u - center_u), torch.abs(v - center_v))
    
    # Normalize distance to [0, num_bins-1] range
    max_d = torch.maximum(torch.tensor(center_u), torch.tensor(center_v))
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
        
        # Level embeddings for hierarchical processing
        self.level_embedding = nn.Embedding(num_haft_levels, 2 * context_vector_dim)
        
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
        device = F_in.device
        
        # Process each channel separately in frequency domain
        enhanced_channels = []
        
        for c in range(C):
            channel_features = F_in[:, c:c+1, :, :]  # (B, 1, H, W)
            enhanced_channel = self._process_channel(channel_features)
            enhanced_channels.append(enhanced_channel)
        
        # Concatenate enhanced channels
        F_out = torch.cat(enhanced_channels, dim=1)  # (B, C, H, W)
        
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
                    phase = torch.angle(patch_fft).unsqueeze(1)
                    
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
                    
                    # Add level embedding
                    level_emb = self.level_embedding(torch.tensor(level, device=device))
                    enriched_context = level_contexts[:, patch_idx] + level_emb
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
                phase = torch.angle(patch_fft)
                
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
