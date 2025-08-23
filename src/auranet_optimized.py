"""
AuraNet: Optimized Implementation
Dual-Stream Forensic Network with performance optimizations
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import yaml
from initial_processing import ArtifactModulatedStem, MSAF, MBConvDownsample, InitialSpatialStem
from utils import Block, LayerNorm
import os
import importlib.util
from logging_utils import get_logger

# Set up logging
logger = get_logger(__name__)


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


class AuraNetOptimized(nn.Module):
    """
    AuraNet: A Dual-Stream Forensic Network for Face Manipulation Detection
    Optimized implementation with parallel processing and memory efficiency
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
        
        # Memory optimizations
        memory_optimization = self.config.get('memory_optimization', {})
        self.use_gradient_checkpoint = memory_optimization.get('gradient_checkpointing', False)
        self.use_optimized_modules = memory_optimization.get('use_optimized_modules', False)
        self.enable_channels_last = memory_optimization.get('enable_channels_last', False)
        self.enable_cudnn_autotuner = memory_optimization.get('enable_cudnn_autotuner', False)
        
        # Enable CuDNN autotuner
        if self.enable_cudnn_autotuner and torch.backends.cudnn.is_available():
            logger.info("Enabling CuDNN benchmark mode")
            torch.backends.cudnn.benchmark = True
        
        # Build model components
        self._build_initial_processing()
        self._build_dual_stream_backbone()
        self._build_output_heads()
        
        # For pre-training
        self._build_pretraining_heads()
        
        # Switch to channels_last memory format if enabled
        if self.enable_channels_last and torch.cuda.is_available():
            logger.info("Using channels_last memory format")
            self = self.to(memory_format=torch.channels_last)
        
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
                
            # Flatten the nested config structure
            flat_config = {}
            flat_config.update(file_config)  # Top-level keys
            
            # Add model-specific parameters if they exist
            if 'model' in file_config:
                flat_config.update(file_config['model'])
                
            default_config.update(flat_config)
            
        # Override with any directly passed parameters
        default_config.update(kwargs)
        
        return default_config
    
    def _build_initial_processing(self):
        """Build Stage 1: Artifact-Modulated Stem (AMS)."""
        # Use the new AMS that combines both spatial and artifact processing
        self.ams = ArtifactModulatedStem(config=self.config)
        
    def _get_optimized_module(self, module_type, *args, **kwargs):
        """Dynamically import and create optimized module variants if available"""
        if not self.use_optimized_modules:
            # Return the standard module based on module_type
            if module_type == 'HAFT':
                from haft import HAFT
                return HAFT(*args, **kwargs)
            elif module_type == 'CrossFusionBlock':
                from cross_fusion import CrossFusionBlock
                return CrossFusionBlock(*args, **kwargs)
            else:
                raise ValueError(f"Unknown module type: {module_type}")
        
        # Try to import the optimized version
        try:
            if module_type == 'HAFT':
                # Check if optimized version exists
                if importlib.util.find_spec("haft_optimized"):
                    from haft_optimized import create_optimized_haft
                    logger.info("Using optimized HAFT module")
                    return create_optimized_haft(*args, **kwargs)
                else:
                    logger.warning("Optimized HAFT module not found, using standard version")
                    from haft import HAFT
                    return HAFT(*args, **kwargs)
            elif module_type == 'CrossFusionBlock':
                # Check if optimized version exists
                if importlib.util.find_spec("cross_fusion_optimized"):
                    from cross_fusion_optimized import create_optimized_cross_fusion
                    logger.info("Using optimized CrossFusion module")
                    return create_optimized_cross_fusion(*args, **kwargs)
                else:
                    logger.warning("Optimized CrossFusion module not found, using standard version")
                    from cross_fusion import CrossFusionBlock
                    return CrossFusionBlock(*args, **kwargs)
            else:
                raise ValueError(f"Unknown module type: {module_type}")
        except Exception as e:
            logger.error(f"Error loading optimized module {module_type}: {e}")
            # Fallback to standard modules
            if module_type == 'HAFT':
                from haft import HAFT
                return HAFT(*args, **kwargs)
            elif module_type == 'CrossFusionBlock':
                from cross_fusion import CrossFusionBlock
                return CrossFusionBlock(*args, **kwargs)
    
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
            haft_block = self._get_optimized_module(
                'HAFT',
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
            cross_fusion = self._get_optimized_module(
                'CrossFusionBlock',
                dim=current_dim,
                use_deformable=use_deformable,
                heads=8
            )
            self.cross_fusion_blocks.append(cross_fusion)
            
            # Downsample layer (except for the last stage)
            if stage_idx < self.num_stages - 1:
                downsample = DownsampleLayer(current_dim)
                self.downsample_layers.append(downsample)
                current_dim = current_dim * 2  # Double channels after downsampling
                
    def _build_output_heads(self):
        """Build final fusion and output heads."""
        final_dim = self.dims[-1]  # 512
        
        # Import needed modules
        from output_heads import DSF, ClassificationHead, SegmentationHead
        
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
        # Import needed modules
        from output_heads import ImageDecoder, MaskDecoder, ContrastiveProjectionHead
        
        final_dim = self.dims[-1]  # 512
        
        # Image decoder for reconstructing masked patches
        self.image_decoder = ImageDecoder(
            in_dim=final_dim,
            decoder_embed_dim=self.decoder_embed_dim,
            decoder_depth=self.decoder_depth
        )
        
        # Mask decoder for predicting binary manipulation masks
        self.mask_decoder = MaskDecoder(
            in_dim=final_dim,
            decoder_embed_dim=self.decoder_embed_dim,
            decoder_depth=self.decoder_depth
        )
        
        # Contrastive learning projection head
        self.contrastive_head = ContrastiveProjectionHead(
            in_dim=final_dim,
            config=self.config
        )
        
    def _process_stage(self, stage_idx, spatial_features, freq_features):
        """Process a single stage of the dual-stream backbone with optimizations."""
        # Extract the stage modules
        spatial_blocks = self.spatial_stages[stage_idx]
        haft_block = self.frequency_stages[stage_idx]
        cross_fusion = self.cross_fusion_blocks[stage_idx]
        
        # Apply spatial branch blocks
        if self.use_gradient_checkpoint and self.training:
            spatial_features = checkpoint.checkpoint(spatial_blocks, spatial_features)
        else:
            spatial_features = spatial_blocks(spatial_features)
        
        # Apply frequency branch blocks
        if self.use_gradient_checkpoint and self.training:
            freq_features = checkpoint.checkpoint(haft_block, freq_features)
        else:
            freq_features = haft_block(freq_features)
        
        # Apply cross-fusion
        if self.use_gradient_checkpoint and self.training:
            spatial_features, freq_features = checkpoint.checkpoint(
                cross_fusion, spatial_features, freq_features
            )
        else:
            spatial_features, freq_features = cross_fusion(spatial_features, freq_features)
            
        return spatial_features, freq_features
        
    def forward_encoder(self, x):
        """Forward pass through the dual-stream encoder with optimization."""
        # Ensure x has correct shape (B, C, H, W)
        if len(x.shape) != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (B, C, H, W), but got shape {x.shape}")
        
        # Convert to channels_last memory format if enabled
        if self.enable_channels_last and torch.cuda.is_available():
            x = x.to(memory_format=torch.channels_last)
        
        # Stage 1: Artifact-Modulated Stem (AMS)
        # This produces a unified feature map for both streams
        unified_features = self.ams(x)  # (B, 64, H/4, W/4)
        
        # Initialize both streams with the same unified features
        spatial_features = unified_features
        freq_features = unified_features
        
        # Stages 2-5: Dual-Stream Backbone
        for stage_idx in range(self.num_stages):
            # Process this stage
            spatial_features, freq_features = self._process_stage(
                stage_idx, spatial_features, freq_features
            )
            
            # Apply downsampling if not the last stage
            if stage_idx < self.num_stages - 1:
                downsample = self.downsample_layers[stage_idx]
                spatial_features = downsample(spatial_features)
                freq_features = downsample(freq_features)
                
                # Clear some memory after downsampling
                if self.training and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return spatial_features, freq_features
    
    def forward_pretrain(self, x, spatial_feat, freq_feat):
        """Forward pass for self-supervised pre-training."""
        # Final fusion
        fused_output, pooled_output = self.dsf(spatial_feat, freq_feat)
        
        # Generate outputs for multiple pretraining tasks
        image_recon = self.image_decoder(fused_output)
        mask_pred = self.mask_decoder(freq_feat)
        
        # Contrastive feature
        contrastive_feat = self.contrastive_head(pooled_output)
        
        return {
            'reconstructed_image': image_recon,
            'predicted_mask': mask_pred,
            'contrastive_feature': contrastive_feat,
            'fused_features': fused_output
        }
    
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
    
    def forward(self, x, mode='finetune'):
        """
        Forward pass through the network.
        
        Args:
            x: (B, C, H, W) input image
            mode: 'pretrain' or 'finetune'
            
        Returns:
            output_dict: Dictionary of outputs depending on mode
        """
        # Encode through dual-stream backbone
        spatial_feat, freq_feat = self.forward_encoder(x)
        
        if mode == 'pretrain':
            return self.forward_pretrain(x, spatial_feat, freq_feat)
        else:
            return self.forward_finetune(spatial_feat, freq_feat)
        
    def load_pretrained_weights(self, pretrained_path, strict=False, **kwargs):
        """Load pretrained weights with support for partial loading."""
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pretrained weights file not found: {pretrained_path}")
        
        # Load the checkpoint
        logger.info(f"Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # If it's a full model checkpoint, extract just the model state
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Load weights
        try:
            missing, unexpected = self.load_state_dict(state_dict, strict=strict)
            if len(missing) > 0:
                logger.info(f"Missing keys: {missing}")
            if len(unexpected) > 0:
                logger.info(f"Unexpected keys: {unexpected}")
        except Exception as e:
            logger.error(f"Error loading pretrained weights: {e}")
            if not strict:
                # Try loading with a more flexible approach
                logger.info("Attempting to load weights with filter_by_prefix")
                self._load_with_filter(state_dict)
            else:
                raise
                
        logger.info("Pretrained weights loaded successfully")
        
    def _load_with_filter(self, state_dict):
        """Flexible loading of weights that match by prefix."""
        own_state = self.state_dict()
        loaded_keys = []
        
        for name, param in state_dict.items():
            # Find matching keys in the model
            for own_name in own_state:
                if own_name.endswith(name) or name.endswith(own_name):
                    # Check if shapes match
                    if param.shape == own_state[own_name].shape:
                        own_state[own_name].copy_(param)
                        loaded_keys.append(own_name)
                        break
        
        logger.info(f"Loaded {len(loaded_keys)} keys with flexible matching")


def create_optimized_auranet(config_path=None, config=None, use_pretrained=False, 
                             pretrained_path=None, **kwargs):
    """
    Create an optimized AuraNet model instance.
    
    Args:
        config_path: Path to YAML configuration file
        config: Configuration dictionary
        use_pretrained: Whether to load pretrained weights
        pretrained_path: Path to pretrained weights file
        **kwargs: Additional configuration parameters
        
    Returns:
        model: AuraNetOptimized instance
    """
    # Create model
    model = AuraNetOptimized(config_path=config_path, config=config, **kwargs)
    
    # Load pretrained weights if specified
    if use_pretrained and pretrained_path:
        model.load_pretrained_weights(pretrained_path, strict=False)
    
    # Compile model if enabled in config
    compile_model = model.config.get('compile_model', False)
    if compile_model and hasattr(torch, 'compile'):
        logger.info("Compiling model with torch.compile()")
        # Get device
        device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        try:
            model = torch.compile(model)
            logger.info("Model compiled successfully")
        except Exception as e:
            logger.error(f"Failed to compile model: {e}")
    
    return model
