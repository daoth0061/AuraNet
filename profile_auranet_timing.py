"""
AuraNet Module Timing Analysis
This script measures the running time of each module in the AuraNet model
during both inference and training modes.
"""

import torch
import torch.nn as nn
import time
import yaml
import os
import sys
import argparse
from pathlib import Path
import numpy as np

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from auranet import create_auranet
from initial_processing import ArtifactModulatedStem
from haft import HAFT
from cross_fusion import CrossFusionBlock
from output_heads import DSF, ClassificationHead, SegmentationHead
from output_heads import ImageDecoder, MaskDecoder, ContrastiveProjectionHead
from training import CombinedPretrainLoss, CombinedFinetuneLoss
from config_loader import load_resolution_config, create_resolution_aware_paths, validate_resolution_config


class AuraNetTimingProfiler:
    """Profiler for measuring AuraNet module execution times."""

    def __init__(self, config_path, img_size=128, batch_size=16):
        self.img_size = img_size
        self.batch_size = batch_size

        # Load resolution-specific configuration
        try:
            self.config = load_resolution_config(img_size)
            self.config = create_resolution_aware_paths(self.config, img_size)
            validate_resolution_config(self.config, img_size)
            print(f"✓ Loaded configuration for {img_size}×{img_size} images")
        except Exception as e:
            print(f"✗ Error loading configuration for {img_size}×{img_size}: {e}")
            print("Falling back to provided config file...")
            # Fallback to provided config if resolution-specific loading fails
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        # Update config for timing analysis
        self.config['img_size'] = [img_size, img_size]
        self.config['dims'] = [64, 128, 256, 512]  # Standard dims
        self.config['depths'] = [2, 2, 6, 2]  # Standard depths

        # Create model
        self.model = create_auranet(config=self.config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        # Create sample input
        self.sample_input = torch.randn(batch_size, 3, img_size, img_size).to(self.device)

        # Create sample labels for training mode
        self.sample_labels = torch.randint(0, 2, (batch_size,)).to(self.device)

        # Create sample masks for segmentation
        self.sample_masks = torch.randn(batch_size, 1, img_size, img_size).to(self.device)

        # Setup loss functions
        self.pretrain_criterion = CombinedPretrainLoss(config=self.config)
        self.finetune_criterion = CombinedFinetuneLoss(config=self.config)

        # Initialize timing results
        self.timing_results = {
            'inference': {},
            'training': {}
        }

    def time_module(self, module_name, module_func, *args, **kwargs):
        """Time a specific module execution."""
        # Warm up
        for _ in range(5):
            _ = module_func(*args, **kwargs)

        # Synchronize CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Measure time
        start_time = time.time()

        # Run multiple times for stable measurement
        num_runs = 10
        for _ in range(num_runs):
            result = module_func(*args, **kwargs)

        # Synchronize CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.time()

        # Calculate average time per run
        avg_time = (end_time - start_time) / num_runs * 1000  # Convert to milliseconds

        return avg_time, result

    def profile_inference(self):
        """Profile inference time for each module."""
        print("Profiling inference mode...")

        # AMS (Artifact-Modulated Stem)
        def ams_forward():
            return self.model.ams(self.sample_input)

        ams_time, ams_output = self.time_module("AMS", ams_forward)
        self.timing_results['inference']['AMS'] = ams_time
        print(".2f")

        # Backbone stages
        spatial_feat = ams_output
        freq_feat = ams_output

        for stage_idx in range(self.model.num_stages):
            stage_name = f"Stage_{stage_idx + 2}"

            # Spatial stream
            def spatial_forward():
                return self.model.spatial_stages[stage_idx](spatial_feat)

            spatial_time, spatial_output = self.time_module(f"{stage_name}_Spatial", spatial_forward)
            self.timing_results['inference'][f'{stage_name}_Spatial'] = spatial_time
            print(".2f")

            # Frequency stream (HAFT)
            def freq_forward():
                return self.model.frequency_stages[stage_idx](freq_feat)

            freq_time, freq_output = self.time_module(f"{stage_name}_Frequency", freq_forward)
            self.timing_results['inference'][f'{stage_name}_Frequency'] = freq_time
            print(".2f")

            # Cross-fusion
            def cross_fusion_forward():
                return self.model.cross_fusion_blocks[stage_idx](spatial_output, freq_output)

            cf_time, (spatial_feat, freq_feat) = self.time_module(f"{stage_name}_CrossFusion", cross_fusion_forward)
            self.timing_results['inference'][f'{stage_name}_CrossFusion'] = cf_time
            print(".2f")

            # Downsample (except last stage)
            if stage_idx < self.model.num_stages - 1:
                def downsample_forward():
                    return self.model.downsample_layers[stage_idx](spatial_feat), \
                           self.model.downsample_layers[stage_idx](freq_feat)

                ds_time, (spatial_feat, freq_feat) = self.time_module(f"{stage_name}_Downsample", downsample_forward)
                self.timing_results['inference'][f'{stage_name}_Downsample'] = ds_time
                print(".2f")

        # DSF (Dynamic Selection Fusion)
        def dsf_forward():
            return self.model.dsf(spatial_feat, freq_feat)

        dsf_time, (fused_output, pooled_output) = self.time_module("DSF", dsf_forward)
        self.timing_results['inference']['DSF'] = dsf_time
        print(".2f")

        # Classification Head
        def class_forward():
            return self.model.classification_head(pooled_output)

        class_time, class_output = self.time_module("ClassificationHead", class_forward)
        self.timing_results['inference']['ClassificationHead'] = class_time
        print(".2f")

        # Segmentation Head
        def seg_forward():
            return self.model.segmentation_head(fused_output)

        seg_time, seg_output = self.time_module("SegmentationHead", seg_forward)
        self.timing_results['inference']['SegmentationHead'] = seg_time
        print(".2f")

        # Total inference time
        total_inference = sum(self.timing_results['inference'].values())
        self.timing_results['inference']['Total'] = total_inference
        print(".2f")

    def profile_training(self):
        """Profile training time for each module."""
        print("\nProfiling training mode...")

        # Enable training mode
        self.model.train()

        # AMS (Artifact-Modulated Stem)
        def ams_forward():
            return self.model.ams(self.sample_input)

        ams_time, ams_output = self.time_module("AMS_train", ams_forward)
        self.timing_results['training']['AMS'] = ams_time
        print(".2f")

        # Backbone stages
        spatial_feat = ams_output
        freq_feat = ams_output

        for stage_idx in range(self.model.num_stages):
            stage_name = f"Stage_{stage_idx + 2}"

            # Spatial stream
            def spatial_forward():
                return self.model.spatial_stages[stage_idx](spatial_feat)

            spatial_time, spatial_output = self.time_module(f"{stage_name}_Spatial_train", spatial_forward)
            self.timing_results['training'][f'{stage_name}_Spatial'] = spatial_time
            print(".2f")

            # Frequency stream (HAFT)
            def freq_forward():
                return self.model.frequency_stages[stage_idx](freq_feat)

            freq_time, freq_output = self.time_module(f"{stage_name}_Frequency_train", freq_forward)
            self.timing_results['training'][f'{stage_name}_Frequency'] = freq_time
            print(".2f")

            # Cross-fusion
            def cross_fusion_forward():
                return self.model.cross_fusion_blocks[stage_idx](spatial_output, freq_output)

            cf_time, (spatial_feat, freq_feat) = self.time_module(f"{stage_name}_CrossFusion_train", cross_fusion_forward)
            self.timing_results['training'][f'{stage_name}_CrossFusion'] = cf_time
            print(".2f")

            # Downsample (except last stage)
            if stage_idx < self.model.num_stages - 1:
                def downsample_forward():
                    return self.model.downsample_layers[stage_idx](spatial_feat), \
                           self.model.downsample_layers[stage_idx](freq_feat)

                ds_time, (spatial_feat, freq_feat) = self.time_module(f"{stage_name}_Downsample_train", downsample_forward)
                self.timing_results['training'][f'{stage_name}_Downsample'] = ds_time
                print(".2f")

        # DSF (Dynamic Selection Fusion)
        def dsf_forward():
            return self.model.dsf(spatial_feat, freq_feat)

        dsf_time, (fused_output, pooled_output) = self.time_module("DSF_train", dsf_forward)
        self.timing_results['training']['DSF'] = dsf_time
        print(".2f")

        # Classification Head
        def class_forward():
            return self.model.classification_head(pooled_output)

        class_time, class_output = self.time_module("ClassificationHead_train", class_forward)
        self.timing_results['training']['ClassificationHead'] = class_time
        print(".2f")

        # Segmentation Head
        def seg_forward():
            return self.model.segmentation_head(fused_output)

        seg_time, seg_output = self.time_module("SegmentationHead_train", seg_forward)
        self.timing_results['training']['SegmentationHead'] = seg_time
        print(".2f")

        # Loss computation (for training)
        batch_data = {
            'image': self.sample_input,
            'label': self.sample_labels,
            'mask': self.sample_masks
        }

        def loss_forward():
            outputs = self.model(self.sample_input, mode='finetune')
            return self.finetune_criterion(outputs, batch_data)

        loss_time, loss_output = self.time_module("Loss_Computation", loss_forward)
        self.timing_results['training']['Loss_Computation'] = loss_time
        print(".2f")

        # Total training time
        total_training = sum(self.timing_results['training'].values())
        self.timing_results['training']['Total'] = total_training
        print(".2f")

    def profile_pretraining(self):
        """Profile pre-training mode with additional heads."""
        print("\nProfiling pre-training mode...")

        self.model.train()

        # Get encoder features
        def encoder_forward():
            return self.model.forward_encoder(self.sample_input)

        enc_time, (spatial_feat, freq_feat) = self.time_module("Encoder", encoder_forward)
        self.timing_results['pretraining'] = {'Encoder': enc_time}
        print(".2f")

        # Image Decoder
        def image_decoder_forward():
            return self.model.image_decoder(spatial_feat)

        img_dec_time, _ = self.time_module("ImageDecoder", image_decoder_forward)
        self.timing_results['pretraining']['ImageDecoder'] = img_dec_time
        print(".2f")

        # Mask Decoder
        def mask_decoder_forward():
            return self.model.mask_decoder(freq_feat)

        mask_dec_time, _ = self.time_module("MaskDecoder", mask_decoder_forward)
        self.timing_results['pretraining']['MaskDecoder'] = mask_dec_time
        print(".2f")

        # Contrastive Projection Head
        def contrastive_forward():
            combined = torch.cat([spatial_feat.mean(dim=[2, 3]), freq_feat.mean(dim=[2, 3])], dim=1)
            return self.model.contrastive_head(combined)

        cont_time, _ = self.time_module("ContrastiveHead", contrastive_forward)
        self.timing_results['pretraining']['ContrastiveHead'] = cont_time
        print(".2f")

        # Pre-training loss
        batch_data = {
            'image': self.sample_input,
            'mask': self.sample_masks
        }

        def pretrain_loss_forward():
            outputs = self.model(self.sample_input, mode='pretrain')
            return self.pretrain_criterion(outputs, batch_data)

        loss_time, _ = self.time_module("PretrainLoss", pretrain_loss_forward)
        self.timing_results['pretraining']['PretrainLoss'] = loss_time
        print(".2f")

        # Total pre-training time
        total_pretrain = sum(self.timing_results['pretraining'].values())
        self.timing_results['pretraining']['Total'] = total_pretrain
        print(".2f")

    def print_summary(self):
        """Print timing summary."""
        print("\n" + "="*80)
        print("AURANET TIMING ANALYSIS SUMMARY")
        print("="*80)
        print(f"Image size: {self.img_size}x{self.img_size}")
        print(f"Batch size: {self.batch_size}")
        print(f"Device: {self.device}")
        print()

        for mode, timings in self.timing_results.items():
            print(f"{mode.upper()} MODE:")
            print("-" * 40)

            # Sort by time (descending)
            sorted_timings = sorted(timings.items(), key=lambda x: x[1], reverse=True)

            for module, time_ms in sorted_timings:
                if module != 'Total':
                    percentage = (time_ms / timings['Total']) * 100
                    print("6.2f")
                else:
                    print("-" * 40)
                    print("6.2f")
            print()

    def save_results(self, output_path="timing_results.yaml"):
        """Save timing results to YAML file."""
        with open(output_path, 'w') as f:
            yaml.dump(self.timing_results, f, default_flow_style=False)
        print(f"Timing results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Profile AuraNet module timing')
    parser.add_argument('--config', type=str, default='config_celeb_df.yaml',
                       help='Path to configuration file')
    parser.add_argument('--img_size', type=int, default=128,
                       help='Image size for profiling')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for profiling')
    parser.add_argument('--output', type=str, default='timing_results.yaml',
                       help='Output file for timing results')

    args = parser.parse_args()

    # Create profiler
    profiler = AuraNetTimingProfiler(args.config, args.img_size, args.batch_size)

    # Profile inference
    profiler.profile_inference()

    # Profile training
    profiler.profile_training()

    # Profile pre-training
    profiler.profile_pretraining()

    # Print summary
    profiler.print_summary()

    # Save results
    profiler.save_results(args.output)


if __name__ == "__main__":
    main()
