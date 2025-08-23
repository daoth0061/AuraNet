"""
AuraNet: Complete Model Implementation
Dual-Stream Forensic Network for Face Manipulation Detection
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import yaml
from initial_processing import ArtifactModulatedStem, MSAF, MBConvDownsample, InitialSpatialStem
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
        
        # MEMORY OPTIMIZATION: Enable gradient checkpointing
        self.use_gradient_checkpoint = self.config.get('memory_optimization', {}).get('gradient_checkpointing', False)
        
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
            'img_size': [64, 64],
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
        """Build Stage 1: Artifact-Modulated Stem (AMS)."""
        # Use the new AMS that combines both spatial and artifact processing
        self.ams = ArtifactModulatedStem(config=self.config)
        
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
        """Forward pass through the dual-stream encoder with optional gradient checkpointing."""
        # Ensure x has correct shape (B, C, H, W)
        if len(x.shape) != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (B, C, H, W), but got shape {x.shape}")
        
        B, _, H, W = x.shape
        
        # Stage 1: Artifact-Modulated Stem (AMS)
        # This produces a unified feature map for both streams
        unified_features = self.ams(x)  # (B, 64, H/4, W/4)
        
        # Initialize both streams with the same unified features
        spatial_features = unified_features
        freq_features = unified_features
        
        # Stages 2-5: Dual-Stream Backbone with gradient checkpointing
        for stage_idx in range(self.num_stages):
            if self.use_gradient_checkpoint and self.training:
                # Use gradient checkpointing for memory efficiency
                def stage_forward(spatial_feat, freq_feat):
                    # Process through respective streams
                    spatial_out = self.spatial_stages[stage_idx](spatial_feat)
                    freq_out = self.frequency_stages[stage_idx](freq_feat)
                    
                    # Cross-fusion
                    spatial_fused, freq_fused = self.cross_fusion_blocks[stage_idx](
                        spatial_out, freq_out
                    )
                    
                    return spatial_fused, freq_fused
                
                # Apply checkpointing
                spatial_features, freq_features = checkpoint.checkpoint(
                    stage_forward, spatial_features, freq_features, use_reentrant=False
                )
            else:
                # Standard forward pass
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
        # Ensure x has correct shape and type
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, but got {type(x)}")
        
        # Handle case where batch dimension is missing
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension
            print(f"WARNING: Input tensor missing batch dimension. Automatically adding it. New shape: {x.shape}")
        
        # Handle case where x is not 4D
        if len(x.shape) != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (B, C, H, W), but got shape {x.shape}")
            
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
    
    def load_convnextv2_pretrained_weights(self, pretrained_path: str, strict: bool = False):
        """
        Load ConvNeXtV2 pre-trained weights for spatial stream only.
        
        Args:
            pretrained_path: Path to ConvNeXtV2 pre-trained weights
            strict: Whether to strictly enforce weight matching
        """
        try:
            # Load pre-trained weights with mixed precision compatibility
            # Use weights_only=False for compatibility with older checkpoint formats
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            
            # Extract state dict (handle different checkpoint formats)
            if 'model' in checkpoint:
                pretrained_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                pretrained_dict = checkpoint['state_dict']
            else:
                pretrained_dict = checkpoint
            
            # Get current model state dict
            model_dict = self.state_dict()
            
            # Filter and map weights for spatial stream only
            mapped_weights = {}
            spatial_layers_loaded = 0
            
            # Map ConvNeXtV2 stages to AuraNet spatial stream
            for stage_idx in range(min(4, len(self.spatial_stages))):  # Up to 4 stages
                stage_prefix = f'stages.{stage_idx}'  # ConvNeXtV2 format
                auranet_prefix = f'spatial_stages.{stage_idx}'  # AuraNet format
                
                for key, value in pretrained_dict.items():
                    if key.startswith(stage_prefix):
                        # Map ConvNeXtV2 key to AuraNet key
                        new_key = key.replace(stage_prefix, auranet_prefix)
                        
                        # Only load if the key exists in our model and has matching shape
                        if new_key in model_dict:
                            if model_dict[new_key].shape == value.shape:
                                mapped_weights[new_key] = value
                                spatial_layers_loaded += 1
                            else:
                                # Handle GRN parameter shape mismatch
                                if 'grn.beta' in new_key or 'grn.gamma' in new_key:
                                    # ConvNeXtV2: [1, C] -> AuraNet: [1, 1, 1, C]
                                    if len(value.shape) == 2 and len(model_dict[new_key].shape) == 4:
                                        # Reshape [1, C] to [1, 1, 1, C]
                                        reshaped_value = value.view(1, 1, 1, -1)
                                        if model_dict[new_key].shape == reshaped_value.shape:
                                            mapped_weights[new_key] = reshaped_value
                                            spatial_layers_loaded += 1
                                        else:
                                            print(f"Shape mismatch after reshape for {new_key}: "
                                                  f"model {model_dict[new_key].shape} vs "
                                                  f"reshaped {reshaped_value.shape}")
                                    else:
                                        print(f"Shape mismatch for {new_key}: "
                                              f"model {model_dict[new_key].shape} vs "
                                              f"pretrained {value.shape}")
                                else:
                                    print(f"Shape mismatch for {new_key}: "
                                          f"model {model_dict[new_key].shape} vs "
                                          f"pretrained {value.shape}")
                        elif not strict:
                            # In non-strict mode, try to find similar layers
                            similar_keys = [k for k in model_dict.keys() if k.endswith(new_key.split('.')[-1])]
                            if similar_keys:
                                best_match = min(similar_keys, key=lambda x: abs(len(x) - len(new_key)))
                                if model_dict[best_match].shape == value.shape:
                                    mapped_weights[best_match] = value
                                    spatial_layers_loaded += 1
            
            # Also try to load the stem/downsample layers if they exist
            stem_mappings = {
                'downsample_layers.0.0.weight': 'ams.spatial_stem.0.weight',  # First conv
                'downsample_layers.0.0.bias': 'ams.spatial_stem.0.bias',
                'downsample_layers.0.1.weight': 'ams.spatial_stem.1.weight',  # LayerNorm
                'downsample_layers.0.1.bias': 'ams.spatial_stem.1.bias',
            }
            
            for pretrained_key, auranet_key in stem_mappings.items():
                if pretrained_key in pretrained_dict and auranet_key in model_dict:
                    if model_dict[auranet_key].shape == pretrained_dict[pretrained_key].shape:
                        mapped_weights[auranet_key] = pretrained_dict[pretrained_key]
                        spatial_layers_loaded += 1
            
            # Load the mapped weights
            if mapped_weights:
                # Handle mixed precision: ensure weights are in correct dtype
                current_dtype = next(self.parameters()).dtype
                for key, value in mapped_weights.items():
                    mapped_weights[key] = value.to(dtype=current_dtype)
                
                self.load_state_dict(mapped_weights, strict=False)
                print(f"✅ Successfully loaded {spatial_layers_loaded} layers from ConvNeXtV2 pretrained weights")
                print(f"   Loaded weights for spatial stream from: {pretrained_path}")
                print(f"   Frequency stream (HAFT) initialized randomly as expected")
            else:
                print(f"⚠️ No compatible weights found in {pretrained_path}")
                
        except Exception as e:
            print(f"❌ Failed to load pretrained weights from {pretrained_path}: {e}")
            print("Proceeding with random initialization...")


def create_auranet(config_path=None, config=None, use_pretrained=False, 
                  pretrained_path=None, **kwargs):
    """
    Factory function to create AuraNet model.
    
    Args:
        config_path: Path to YAML configuration file
        config: Configuration dictionary (alternative to config_path)
        use_pretrained: Whether to load ConvNeXtV2 pretrained weights
        pretrained_path: Path to pretrained weights file (default: convnextv2_pico_1k_224_fcmae.pt)
        **kwargs: Configuration overrides
        
    Returns:
        AuraNet model instance
    """
    model = AuraNet(config_path=config_path, config=config, **kwargs)
    
    # Load pretrained weights if requested
    if use_pretrained:
        if pretrained_path is None:
            # Default path for ConvNeXtV2 Pico FCMAE weights
            pretrained_path = "convnextv2_pico_1k_224_fcmae.pt"
        
        print(f"🔄 Loading ConvNeXtV2 pretrained weights for spatial stream...")
        model.load_convnextv2_pretrained_weights(pretrained_path, strict=False)
    
    return model
