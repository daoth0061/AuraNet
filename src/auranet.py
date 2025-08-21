"""
AuraNet: Complete Model Implementation
Dual-Stream Forensic Network for Face Manipulation Detection
"""

import torch
import torch.nn as nn
import yaml
from initial_processing import MSAF, MBConvDownsample, InitialSpatialStem
from haft import HAFT
from cross_fusion import CrossFusionBlock
from output_heads import DSF, ClassificationHead, SegmentationHead
from output_heads import ImageDecoder, MaskDecoder, ContrastiveProjectionHead
from utils import Block, LayerNorm


class DownsampleLayer(nn.Module):
    """Downsample layer between stages."""
    
    def __init__(self, dim):
        super().__init__()
        self.downsample = nn.Sequential(
            LayerNorm(dim, data_format="channels_first"),
            nn.Conv2d(dim, dim * 2, kernel_size=2, stride=2)
        )
        
    def forward(self, x):
        return self.downsample(x)


class AuraNet(nn.Module):
    """
    AuraNet: A Dual-Stream Forensic Network for Face Manipulation Detection
    
    This model implements a sophisticated dual-stream architecture that processes
    images through both spatial and frequency domains for robust manipulation detection.
    """
    
    def __init__(self, config_path=None, config=None, **kwargs):
        super().__init__()
        
        # Load configuration
        self.config = self._load_config(config_path, config, **kwargs)
        
        # Extract configuration parameters
        self.img_size = self.config['img_size']
        self.num_stages = self.config['num_stages']
        self.depths = self.config['depths']
        self.dims = self.config['dims']
        self.num_haft_levels = self.config['num_haft_levels']
        self.num_radial_bins = self.config['num_radial_bins']
        self.context_vector_dim = self.config['context_vector_dim']
        self.dsf_dim = self.config['dsf_dim']
        self.decoder_embed_dim = self.config['decoder_embed_dim']
        self.decoder_depth = self.config['decoder_depth']
        self.num_classes = self.config['num_classes']
        
        # Build model components
        self._build_initial_processing()
        self._build_dual_stream_backbone()
        self._build_output_heads()
        
        # For pre-training
        self._build_pretraining_heads()
        
    def _load_config(self, config_path=None, config=None, **kwargs):
        """Load configuration from file or use defaults with overrides."""
        # Default configuration
        default_config = {
            'img_size': [256, 256],
            'num_stages': 4,
            'depths': [2, 2, 6, 2],
            'dims': [64, 128, 256, 512],
            'num_haft_levels': [3, 3, 2, 1],
            'num_radial_bins': 16,
            'context_vector_dim': 64,
            'dsf_dim': 512,
            'decoder_embed_dim': 342,
            'decoder_depth': 1,
            'num_classes': 2
        }
        
        # If config dict is passed directly, use it
        if config is not None:
            # Flatten the nested config structure
            flat_config = {}
            flat_config.update(config)  # Top-level keys
            
            # Add model-specific parameters if they exist
            if 'model' in config:
                flat_config.update(config['model'])
            
            default_config.update(flat_config)
        # Load from YAML file if provided
        elif config_path:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                
            # Flatten the nested structure
            flat_config = {}
            flat_config.update(file_config)  # Top-level keys
            
            # Add model-specific parameters if they exist
            if 'model' in file_config:
                flat_config.update(file_config['model'])
            
            default_config.update(flat_config)
        
        # Override with any provided kwargs
        default_config.update(kwargs)
        
        return default_config
    
    def _build_initial_processing(self):
        """Build Stage 1: Initial Processing components."""
        # MSAF for frequency stream initialization
        self.msaf = MSAF(config=self.config)
        
        # MBConv downsample block
        self.mbconv_downsample = MBConvDownsample(out_channels=self.dims[0], config=self.config)
        
        # Initial spatial stem
        self.spatial_stem = InitialSpatialStem(config=self.config)
        
    def _build_dual_stream_backbone(self):
        """Build Stages 2-5: Dual-Stream Backbone."""
        self.spatial_stages = nn.ModuleList()
        self.frequency_stages = nn.ModuleList() 
        self.cross_fusion_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        current_dim = self.dims[0]  # Start with 64 channels
        
        for stage_idx in range(self.num_stages):
            stage_dim = self.dims[stage_idx]
            stage_depth = self.depths[stage_idx]
            stage_haft_levels = self.num_haft_levels[stage_idx]
            
            # Spatial stream: Stack of ConvNeXt V2 blocks
            spatial_blocks = nn.Sequential(*[
                Block(current_dim) for _ in range(stage_depth)
            ])
            self.spatial_stages.append(spatial_blocks)
            
            # Frequency stream: HAFT block
            haft_block = HAFT(
                in_channels=current_dim,
                num_haft_levels=stage_haft_levels,
                num_radial_bins=self.num_radial_bins,
                context_vector_dim=self.context_vector_dim
            )
            self.frequency_stages.append(haft_block)
            
            # Cross-fusion block
            # Use deformable attention for high-resolution stages (0,1 -> stages 2,3)
            # Use standard attention for low-resolution stages (2,3 -> stages 4,5)
            use_deformable = stage_idx < 2
            cross_fusion = CrossFusionBlock(
                dim=current_dim,
                use_deformable=use_deformable,
                heads=8
            )
            self.cross_fusion_blocks.append(cross_fusion)
            
            # Downsample layer (except for the last stage)
            if stage_idx < self.num_stages - 1:
                downsample = DownsampleLayer(current_dim)
                self.downsample_layers.append(downsample)
                current_dim = stage_dim * 2 if stage_idx < len(self.dims) - 1 else stage_dim
            else:
                self.downsample_layers.append(nn.Identity())
    
    def _build_output_heads(self):
        """Build final fusion and output heads."""
        final_dim = self.dims[-1]  # 512
        
        # Dynamic Selection Fusion
        self.dsf = DSF(final_dim, config=self.config)
        
        # Classification head
        self.classification_head = ClassificationHead(
            input_dim=final_dim,
            config=self.config
        )
        
        # Segmentation head
        self.segmentation_head = SegmentationHead(
            input_dim=final_dim,
            decoder_embed_dim=self.decoder_embed_dim,
            decoder_depth=self.decoder_depth
        )
    
    def _build_pretraining_heads(self):
        """Build heads for self-supervised pre-training."""
        final_dim = self.dims[-1]  # 512
        
        # Image decoder for spatial stream
        self.image_decoder = ImageDecoder(
            input_dim=final_dim,
            output_channels=3,
            decoder_embed_dim=self.decoder_embed_dim,
            decoder_depth=self.decoder_depth
        )
        
        # Mask decoder for frequency stream
        self.mask_decoder = MaskDecoder(
            input_dim=final_dim,
            decoder_embed_dim=self.decoder_embed_dim,
            decoder_depth=self.decoder_depth
        )
        
        # Contrastive projection head
        self.contrastive_head = ContrastiveProjectionHead(
            input_dim=final_dim * 2,  # Concatenated spatial + frequency features
            projection_dim=128
        )
    
    def forward_encoder(self, x):
        """Forward pass through the dual-stream encoder."""
        B, _, H, W = x.shape
        
        # Stage 1: Initial Processing
        # Frequency stream initialization
        freq_features = self.msaf(x)  # (B, 32, H/2, W/2)
        freq_features = self.mbconv_downsample(freq_features)  # (B, 64, H/4, W/4)
        
        # Spatial stream initialization  
        spatial_features = self.spatial_stem(x)  # (B, 64, H/4, W/4)
        
        # Stages 2-5: Dual-Stream Backbone
        for stage_idx in range(self.num_stages):
            # Process through respective streams
            spatial_features = self.spatial_stages[stage_idx](spatial_features)
            freq_features = self.frequency_stages[stage_idx](freq_features)
            
            # Cross-fusion
            spatial_features, freq_features = self.cross_fusion_blocks[stage_idx](
                spatial_features, freq_features
            )
            
            # Downsample for next stage (except last)
            if stage_idx < self.num_stages - 1:
                spatial_features = self.downsample_layers[stage_idx](spatial_features)
                freq_features = self.downsample_layers[stage_idx](freq_features)
        
        return spatial_features, freq_features
    
    def forward(self, x, mode='finetune'):
        """
        Forward pass with different modes.
        
        Args:
            x: (B, 3, H, W) input images
            mode: str, 'finetune' or 'pretrain'
            
        Returns:
            Dictionary containing outputs based on mode
        """
        # Encode features
        spatial_feat, freq_feat = self.forward_encoder(x)
        
        if mode == 'pretrain':
            return self.forward_pretrain(x, spatial_feat, freq_feat)
        else:
            return self.forward_finetune(spatial_feat, freq_feat)
    
    def forward_finetune(self, spatial_feat, freq_feat):
        """Forward pass for supervised fine-tuning."""
        # Final fusion
        fused_output, pooled_output = self.dsf(spatial_feat, freq_feat)
        
        # Generate outputs
        classification_logits = self.classification_head(pooled_output)
        segmentation_mask = self.segmentation_head(fused_output)
        
        return {
            'classification_logits': classification_logits,
            'segmentation_mask': segmentation_mask,
            'fused_features': fused_output
        }
    
    def forward_pretrain(self, original_x, spatial_feat, freq_feat):
        """Forward pass for self-supervised pre-training."""
        # Reconstruction tasks
        reconstructed_image = self.image_decoder(spatial_feat)
        reconstructed_mask = self.mask_decoder(freq_feat)
        
        # Contrastive learning
        # Global average pooling for both streams
        spatial_pooled = torch.nn.functional.adaptive_avg_pool2d(spatial_feat, 1).flatten(1)
        freq_pooled = torch.nn.functional.adaptive_avg_pool2d(freq_feat, 1).flatten(1)
        
        # Concatenate and project
        combined_features = torch.cat([spatial_pooled, freq_pooled], dim=1)
        contrastive_embedding = self.contrastive_head(combined_features)
        
        return {
            'reconstructed_image': reconstructed_image,
            'reconstructed_mask': reconstructed_mask,
            'contrastive_embedding': contrastive_embedding,
            'spatial_features': spatial_feat,
            'freq_features': freq_feat
        }
    
    def get_config(self):
        """Return the model configuration."""
        return self.config.copy()


def create_auranet(config_path=None, config=None, **kwargs):
    """
    Factory function to create AuraNet model.
    
    Args:
        config_path: Path to YAML configuration file
        config: Configuration dictionary (alternative to config_path)
        **kwargs: Configuration overrides
        
    Returns:
        AuraNet model instance
    """
    return AuraNet(config_path=config_path, config=config, **kwargs)
