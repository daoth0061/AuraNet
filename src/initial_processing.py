"""
Initial Processing Stage Components
Implements MSAF, MBConv, and Initial Spatial Stem
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
from utils import LayerNorm, SEBlock, CBAMChannelAttention, CBAMSpatialAttention
import yaml
import os


class SRMFilters(nn.Module):
    """Fixed SRM filter bank for steganalysis-inspired feature extraction."""
    
    def __init__(self):
        super().__init__()
        # Define the 10 SRM filters as specified in Appendix A.1
        filters = [
            # Sobel H
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            # Sobel V
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            # LoG 5x5
            [[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], 
             [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]],
            # EDGE 1 5x5
            [[0, 0, 0, 0, 0], [-1, -2, 0, 2, 1], [0, 0, 0, 0, 0], 
             [1, 2, 0, -2, -1], [0, 0, 0, 0, 0]],
            # EDGE 2 5x5
            [[-1, 0, 1, 0, -1], [-2, 0, 2, 0, -2], [0, 0, 0, 0, 0], 
             [2, 0, -2, 0, 2], [1, 0, -1, 0, 1]],
            # EDGE 3 5x5
            [[0, -1, 0, 1, 0], [0, -2, 0, 2, 0], [0, 0, 0, 0, 0], 
             [0, 2, 0, -2, 0], [0, 1, 0, -1, 0]],
            # SQUARE 3x3
            [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]],
            # SQUARE 5x5
            [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], 
             [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
            # D3,3 4x4
            [[1, -3, 3, -1], [-3, 9, -9, 3], [3, -9, 9, -3], [-1, 3, -3, 1]],
            # D4,4 5x5
            [[1, -4, 6, -4, 1], [-4, 16, -24, 16, -4], [6, -24, 36, -24, 6], 
             [-4, 16, -24, 16, -4], [1, -4, 6, -4, 1]]
        ]
        
        # Convert to tensors and pad to 5x5 where needed
        filter_tensors = []
        for f in filters:
            f_array = np.array(f, dtype=np.float32)
            h, w = f_array.shape
            
            # Pad to 5x5 if needed
            if h < 5 or w < 5:
                # Calculate padding for each side
                pad_h = 5 - h
                pad_w = 5 - w
                
                # Use asymmetric padding if needed
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                
                f_array = np.pad(f_array, 
                    ((pad_top, pad_bottom), (pad_left, pad_right)), 
                    'constant', constant_values=0)
            
            filter_tensors.append(torch.tensor(f_array).unsqueeze(0).unsqueeze(0))
        
        # Stack all filters
        srm_filters = torch.cat(filter_tensors, dim=0)  # (10, 1, 5, 5)
        
        # Register as non-trainable parameters
        self.register_buffer('filters', srm_filters)
        
    def forward(self, x):
        """Apply SRM filters to input image.
        
        Args:
            x: (B, 1, H, W) grayscale image
            
        Returns:
            (B, 10, H, W) SRM feature maps
        """
        # Apply all 10 SRM filters
        return F.conv2d(x, self.filters, padding=2)


class DWTExtractor(nn.Module):
    """2-level Discrete Wavelet Transform feature extractor."""
    
    def __init__(self):
        super().__init__()
        self.wavelets = ['haar', 'coif1']
        
    def forward(self, x):
        """Extract DWT features.
        
        Args:
            x: (B, 1, H, W) grayscale image
            
        Returns:
            (B, 6, H/2, W/2) DWT detail coefficients
        """
        B, _, H, W = x.shape
        
        dwt_features = []
        
        for wavelet in self.wavelets:
            # Convert to numpy for pywt processing
            x_np = x.squeeze(1).cpu().numpy()  # (B, H, W)
            
            batch_coeffs = []
            for i in range(B):
                # 2-level DWT
                coeffs = pywt.dwt2(x_np[i], wavelet)
                cA1, (cH1, cV1, cD1) = coeffs
                
                coeffs2 = pywt.dwt2(cA1, wavelet)
                cA2, (cH2, cV2, cD2) = coeffs2
                
                # Collect detail coefficients from both levels
                # Level 1: cH1, cV1, cD1 are at H/2 x W/2
                # Level 2: cH2, cV2, cD2 are at H/4 x W/4, need to upsample to H/2 x W/2
                
                # Convert level 1 coefficients to tensors first to get actual size
                cH1_tensor = torch.tensor(cH1).float()
                cV1_tensor = torch.tensor(cV1).float()
                cD1_tensor = torch.tensor(cD1).float()
                
                # Use actual size of level 1 coefficients as target
                target_size = cH1_tensor.shape  # Actual (H/2, W/2) size
                
                # Convert and upsample level 2 coefficients
                cH2_up = torch.tensor(cH2).float().unsqueeze(0).unsqueeze(0)
                cV2_up = torch.tensor(cV2).float().unsqueeze(0).unsqueeze(0)
                cD2_up = torch.tensor(cD2).float().unsqueeze(0).unsqueeze(0)
                
                cH2_up = F.interpolate(cH2_up, size=target_size, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                cV2_up = F.interpolate(cV2_up, size=target_size, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                cD2_up = F.interpolate(cD2_up, size=target_size, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                
                # Stack all detail coefficients
                detail_stack = torch.stack([
                    cH1_tensor, cV1_tensor, cD1_tensor,
                    cH2_up, cV2_up, cD2_up
                ], dim=0)  # (6, H/2, W/2)
                
                batch_coeffs.append(detail_stack)
            
            # Convert back to tensor and move to device
            batch_tensor = torch.stack(batch_coeffs, dim=0).to(x.device)  # (B, 6, H/2, W/2)
            dwt_features.append(batch_tensor)
        
        # Use features from the first wavelet for now
        # TODO: Could concatenate features from both wavelets if needed
        return dwt_features[0]


class MSAF(nn.Module):
    """Multi-Scale Artifact Fusion Module."""
    
    def __init__(self, config=None):
        super().__init__()
        
        # Load config if provided
        if config is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        # Extract parameters from config
        srm_channels = config['model']['msaf_srm_channels']
        dwt_channels = config['model']['msaf_dwt_channels']
        cbam_reduction = config['model']['cbam_reduction']
        
        self.srm_filters = SRMFilters()
        self.dwt_extractor = DWTExtractor()
        
        # Processing paths
        self.srm_path = nn.Sequential(
            nn.Conv2d(10, srm_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(srm_channels),
            nn.GELU()
        )
        
        self.dwt_path = nn.Sequential(
            nn.Conv2d(6, dwt_channels, kernel_size=1),
            nn.BatchNorm2d(dwt_channels),
            nn.GELU()
        )
        
        # CBAM attention
        total_channels = srm_channels + dwt_channels
        self.channel_attention = CBAMChannelAttention(total_channels, reduction=cbam_reduction)
        self.spatial_attention = CBAMSpatialAttention(kernel_size=7)
        
    def forward(self, rgb_image):
        """
        Args:
            rgb_image: (B, 3, H, W)
            
        Returns:
            initial_freq_features_H2: (B, 32, H/2, W/2)
        """
        B, _, H, W = rgb_image.shape
        
        # Convert to grayscale using standard luminance formula (ITU-R BT.709)
        # Y = 0.2126*R + 0.7152*G + 0.0722*B
        r, g, b = rgb_image[:, 0:1, :, :], rgb_image[:, 1:2, :, :], rgb_image[:, 2:3, :, :]
        grayscale = 0.2126 * r + 0.7152 * g + 0.0722 * b  # (B, 1, H, W)
        
        # Extract SRM features
        srm_features = self.srm_filters(grayscale)  # (B, 10, H, W)
        
        # Extract DWT features
        dwt_features = self.dwt_extractor(grayscale)  # (B, 6, H/2, W/2)
        
        # Process through respective paths
        srm_processed = self.srm_path(srm_features)  # (B, 20, H/2, W/2)
        dwt_processed = self.dwt_path(dwt_features)  # (B, 12, H/2, W/2)
        
        # Concatenate
        fused_artifacts = torch.cat([srm_processed, dwt_processed], dim=1)  # (B, 32, H/2, W/2)
        
        # Apply CBAM attention
        channel_att = self.channel_attention(fused_artifacts)
        output = fused_artifacts * channel_att
        
        spatial_att = self.spatial_attention(output)
        output = output * spatial_att
        
        return output


class MBConvDownsample(nn.Module):
    """MBConv Downsample Block."""
    
    def __init__(self, in_channels=None, out_channels=64, config=None):
        super().__init__()
        
        # Load config if provided
        if config is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        # Extract parameters from config
        expand_ratio = config['model']['mbconv_expand_ratio']
        se_reduction = config['model']['se_reduction']
        
        # Use config default if in_channels not specified
        if in_channels is None:
            in_channels = config['model']['msaf_fused_channels']
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        expanded_channels = in_channels * expand_ratio
        
        # Pointwise expand
        self.expand_conv = nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(expanded_channels)
        
        # Depthwise downsample
        self.depthwise_conv = nn.Conv2d(expanded_channels, expanded_channels, 
                                       kernel_size=3, stride=2, padding=1, 
                                       groups=expanded_channels, bias=False)
        self.depthwise_norm = LayerNorm(expanded_channels, data_format="channels_first")
        
        # SE block
        self.se_block = SEBlock(expanded_channels, reduction=se_reduction)
        
        # Pointwise project
        self.project_conv = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        
        # Residual path
        self.residual_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        # Final normalization
        self.final_norm = LayerNorm(out_channels, data_format="channels_first")
        
        self.gelu = nn.GELU()
        
    def forward(self, x):
        """
        Args:
            x: (B, 32, H/2, W/2)
            
        Returns:
            (B, 64, H/4, W/4)
        """
        # Residual path
        residual = self.residual_pool(x)
        residual = self.residual_conv(residual)
        
        # Main path
        # Expand
        out = self.expand_conv(x)
        out = self.expand_bn(out)
        out = self.gelu(out)
        
        # Depthwise
        out = self.depthwise_conv(out)
        out = self.depthwise_norm(out)
        out = self.gelu(out)
        
        # SE
        out = self.se_block(out)
        
        # Project
        out = self.project_conv(out)
        out = self.project_bn(out)
        
        # Add residual
        out = out + residual
        
        # Final activation
        out = self.final_norm(out)
        out = self.gelu(out)
        
        return out


class ArtifactModulatedStem(nn.Module):
    """
    Artifact-Modulated Stem (AMS) - Stage 1
    Follows the updated design from Project_AuraNet_Implementation_Plan.txt
    """
    
    def __init__(self, config=None):
        super().__init__()
        
        # Load config if provided
        if config is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        # Extract dimensions
        dims = config['dims']  # [64, 128, 256, 512]
        
        # 1.1.1 Parallel Feature Extraction
        
        # Spatial Path: Standard ConvNeXt-style stem
        self.spatial_stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),  # (B, 64, H/4, W/4)
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        
        # Artifact Path: Compact MSAF Variant
        self.msaf = MSAF(config=config)  # Outputs (B, 32, H/2, W/2)
        
        # 1.1.2 Parameter Prediction & Adaptive Modulation
        
        # Parameter Head: downsample from H/2 to H/4 and predict gamma, beta
        self.parameter_head = nn.Conv2d(32, 128, kernel_size=3, stride=2, padding=1)
        
    def forward(self, rgb_image):
        """
        Args:
            rgb_image: (B, 3, H, W)
            
        Returns:
            enhanced_features: (B, 64, H/4, W/4) - unified feature map for both streams
        """
        # 1.1.1 Parallel Feature Extraction
        
        # Spatial Path
        spatial_features = self.spatial_stem(rgb_image)  # (B, 64, H/4, W/4)
        
        # Artifact Path
        artifact_features = self.msaf(rgb_image)  # (B, 32, H/2, W/2)
        
        # 1.1.2 Parameter Prediction & Adaptive Modulation
        
        # Predict modulation parameters
        params = self.parameter_head(artifact_features)  # (B, 128, H/4, W/4)
        gamma, beta = torch.split(params, 64, dim=1)  # Each (B, 64, H/4, W/4)
        
        # Adaptive modulation (AdaIN-style)
        enhanced_features = gamma * spatial_features + beta
        
        return enhanced_features


class InitialSpatialStem(nn.Module):
    """Standard ConvNeXt-style stem following FCMAE pattern."""
    
    def __init__(self, config=None):
        super().__init__()
        
        # Load config if provided
        if config is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        # Extract dimensions - following FCMAE exactly
        dims = config['dims']  # [64, 128, 256, 512]
        
        # Standard ConvNeXt-style stem: single 4x4 conv with stride=4
        # This matches FCMAE's encoder stem exactly
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),  # 256x256 -> 64x64
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        
    def forward(self, rgb_image):
        """
        Args:
            rgb_image: (B, 3, H, W) - typically (B, 3, 256, 256)
            
        Returns:
            (B, 64, H/4, W/4) - typically (B, 64, 64, 64)
        """
        return self.stem(rgb_image)
