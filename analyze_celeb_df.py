"""
Celeb-DF Dataset Analysis and Preparation Script
Analyzes dataset structure and creates training/test splits according to the specified strategy
"""

import os
import sys
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import re
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from celeb_df_dataset import analyze_celeb_df_dataset


class CelebDFAnalyzer:
    """Analyzer for Celeb-DF dataset structure and statistics."""
    
    def __init__(self, data_root: str, config_path: str = None):
        self.data_root = Path(data_root)
        
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent / 'config_celeb_df.yaml'
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dataset_config = self.config['dataset']
        
        # Dataset directories
        self.real_dir = self.data_root / self.dataset_config['subfolders']['real']
        self.fake_dir = self.data_root / self.dataset_config['subfolders']['fake']
        self.mask_dir = self.data_root / self.dataset_config['subfolders']['mask']
    
    def analyze_dataset_structure(self) -> Dict:
        """Analyze the complete dataset structure."""
        print("Analyzing Celeb-DF dataset structure...")
        
        analysis = {
            'basic_stats': self._get_basic_statistics(),
            'video_analysis': self._analyze_videos(),
            'frame_analysis': self._analyze_frames(),
            'split_analysis': self._analyze_train_test_splits(),
            'matching_analysis': self._analyze_fake_mask_matching()
        }
        
        return analysis
    
    def _get_basic_statistics(self) -> Dict:
        """Get basic dataset statistics."""
        stats = analyze_celeb_df_dataset(str(self.data_root))
        return stats
    
    def _analyze_videos(self) -> Dict:
        """Analyze video-level statistics."""
        video_stats = {
            'real_videos': {},
            'fake_videos': {},
            'mask_videos': {}
        }
        
        # Analyze real videos
        if self.real_dir.exists():
            for video_dir in self.real_dir.iterdir():
                if video_dir.is_dir():
                    images = list(video_dir.glob('*.jpg'))
                    video_stats['real_videos'][video_dir.name] = len(images)
        
        # Analyze fake videos
        if self.fake_dir.exists():
            for video_dir in self.fake_dir.iterdir():
                if video_dir.is_dir():
                    images = list(video_dir.glob('*.jpg'))
                    video_stats['fake_videos'][video_dir.name] = len(images)
        
        # Analyze mask videos
        if self.mask_dir.exists():
            for video_dir in self.mask_dir.iterdir():
                if video_dir.is_dir():
                    images = list(video_dir.glob('*.jpg'))
                    video_stats['mask_videos'][video_dir.name] = len(images)
        
        # Summary statistics
        summary = {
            'total_real_videos': len(video_stats['real_videos']),
            'total_fake_videos': len(video_stats['fake_videos']),
            'total_mask_videos': len(video_stats['mask_videos']),
            'avg_frames_per_real_video': np.mean(list(video_stats['real_videos'].values())) if video_stats['real_videos'] else 0,
            'avg_frames_per_fake_video': np.mean(list(video_stats['fake_videos'].values())) if video_stats['fake_videos'] else 0,
            'avg_frames_per_mask_video': np.mean(list(video_stats['mask_videos'].values())) if video_stats['mask_videos'] else 0
        }
        
        return {**video_stats, 'summary': summary}
    
    def _analyze_frames(self) -> Dict:
        """Analyze frame-level statistics."""
        frame_stats = {
            'real_frames': defaultdict(list),
            'fake_frames': defaultdict(list),
            'mask_frames': defaultdict(list)
        }
        
        def extract_frame_info(filename: str) -> Tuple[int, int]:
            """Extract face_id and frame_id from filename."""
            pattern = r'cropped_face_(\d+)_frame_(\d+)\.jpg'
            match = re.match(pattern, filename)
            if match:
                return int(match.group(1)), int(match.group(2))
            return 0, 0
        
        # Analyze real frames
        if self.real_dir.exists():
            for video_dir in self.real_dir.iterdir():
                if video_dir.is_dir():
                    for img_file in video_dir.glob('*.jpg'):
                        face_id, frame_id = extract_frame_info(img_file.name)
                        frame_stats['real_frames'][video_dir.name].append({
                            'filename': img_file.name,
                            'face_id': face_id,
                            'frame_id': frame_id
                        })
        
        # Analyze fake frames
        if self.fake_dir.exists():
            for video_dir in self.fake_dir.iterdir():
                if video_dir.is_dir():
                    for img_file in video_dir.glob('*.jpg'):
                        face_id, frame_id = extract_frame_info(img_file.name)
                        frame_stats['fake_frames'][video_dir.name].append({
                            'filename': img_file.name,
                            'face_id': face_id,
                            'frame_id': frame_id
                        })
        
        # Analyze mask frames
        if self.mask_dir.exists():
            for video_dir in self.mask_dir.iterdir():
                if video_dir.is_dir():
                    for img_file in video_dir.glob('*.jpg'):
                        face_id, frame_id = extract_frame_info(img_file.name)
                        frame_stats['mask_frames'][video_dir.name].append({
                            'filename': img_file.name,
                            'face_id': face_id,
                            'frame_id': frame_id
                        })
        
        return frame_stats
    
    def _analyze_train_test_splits(self) -> Dict:
        """Analyze train/test splits according to the sampling strategy."""
        split_stats = {
            'real': {'train': 0, 'test': 0},
            'fake': {'train': 0, 'test': 0},
            'mask': {'train': 0, 'test': 0}
        }
        
        frame_modulo = self.dataset_config['frame_modulo']
        test_remainder = self.dataset_config['test_remainder']
        
        def is_test_sample(frame_id: int) -> bool:
            """Determine if frame should be in test set."""
            remainder = (frame_id // frame_modulo) % 10
            return remainder == test_remainder
        
        # Analyze real images
        if self.real_dir.exists():
            for video_dir in self.real_dir.iterdir():
                if video_dir.is_dir():
                    for img_file in video_dir.glob('*.jpg'):
                        pattern = r'cropped_face_(\d+)_frame_(\d+)\.jpg'
                        match = re.match(pattern, img_file.name)
                        if match:
                            frame_id = int(match.group(2))
                            if is_test_sample(frame_id):
                                split_stats['real']['test'] += 1
                            else:
                                split_stats['real']['train'] += 1
        
        # Analyze fake images
        if self.fake_dir.exists():
            for video_dir in self.fake_dir.iterdir():
                if video_dir.is_dir():
                    for img_file in video_dir.glob('*.jpg'):
                        pattern = r'cropped_face_(\d+)_frame_(\d+)\.jpg'
                        match = re.match(pattern, img_file.name)
                        if match:
                            frame_id = int(match.group(2))
                            if is_test_sample(frame_id):
                                split_stats['fake']['test'] += 1
                            else:
                                split_stats['fake']['train'] += 1
        
        # Analyze mask images
        if self.mask_dir.exists():
            for video_dir in self.mask_dir.iterdir():
                if video_dir.is_dir():
                    for img_file in video_dir.glob('*.jpg'):
                        pattern = r'cropped_face_(\d+)_frame_(\d+)\.jpg'
                        match = re.match(pattern, img_file.name)
                        if match:
                            frame_id = int(match.group(2))
                            if is_test_sample(frame_id):
                                split_stats['mask']['test'] += 1
                            else:
                                split_stats['mask']['train'] += 1
        
        # Calculate ratios
        for category in split_stats:
            total = split_stats[category]['train'] + split_stats[category]['test']
            if total > 0:
                split_stats[category]['train_ratio'] = split_stats[category]['train'] / total
                split_stats[category]['test_ratio'] = split_stats[category]['test'] / total
            else:
                split_stats[category]['train_ratio'] = 0
                split_stats[category]['test_ratio'] = 0
        
        return split_stats
    
    def _analyze_fake_mask_matching(self) -> Dict:
        """Analyze how fake images match with mask images."""
        matching_stats = {
            'total_fake_images': 0,
            'total_mask_images': 0,
            'matched_pairs': 0,
            'unmatched_fake': 0,
            'unmatched_mask': 0,
            'matching_ratio': 0.0,
            'video_level_matching': {}
        }
        
        # Get all mask files
        mask_files = set()
        if self.mask_dir.exists():
            for video_dir in self.mask_dir.iterdir():
                if video_dir.is_dir():
                    for mask_file in video_dir.glob('*.jpg'):
                        rel_path = mask_file.relative_to(self.mask_dir)
                        mask_files.add(str(rel_path))
                        matching_stats['total_mask_images'] += 1
        
        # Check fake images for matches
        video_matching = defaultdict(lambda: {'fake_count': 0, 'matched_count': 0})
        
        if self.fake_dir.exists():
            for video_dir in self.fake_dir.iterdir():
                if video_dir.is_dir():
                    video_name = video_dir.name
                    
                    for fake_file in video_dir.glob('*.jpg'):
                        matching_stats['total_fake_images'] += 1
                        video_matching[video_name]['fake_count'] += 1
                        
                        rel_path = fake_file.relative_to(self.fake_dir)
                        if str(rel_path) in mask_files:
                            matching_stats['matched_pairs'] += 1
                            video_matching[video_name]['matched_count'] += 1
        
        # Calculate statistics
        matching_stats['unmatched_fake'] = matching_stats['total_fake_images'] - matching_stats['matched_pairs']
        matching_stats['unmatched_mask'] = matching_stats['total_mask_images'] - matching_stats['matched_pairs']
        
        if matching_stats['total_fake_images'] > 0:
            matching_stats['matching_ratio'] = matching_stats['matched_pairs'] / matching_stats['total_fake_images']
        
        # Video-level matching ratios
        for video_name, stats in video_matching.items():
            if stats['fake_count'] > 0:
                stats['matching_ratio'] = stats['matched_count'] / stats['fake_count']
        
        matching_stats['video_level_matching'] = dict(video_matching)
        
        return matching_stats
    
    def generate_report(self, output_dir: str = "analysis_reports"):
        """Generate comprehensive analysis report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print("Generating comprehensive analysis report...")
        
        # Perform analysis
        analysis = self.analyze_dataset_structure()
        
        # Save analysis results
        report_file = output_dir / "celeb_df_analysis.json"
        with open(report_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Generate summary report
        self._generate_summary_report(analysis, output_dir)
        
        # Generate visualizations
        self._generate_visualizations(analysis, output_dir)
        
        print(f"Analysis complete! Reports saved to {output_dir}")
    
    def _generate_summary_report(self, analysis: Dict, output_dir: Path):
        """Generate human-readable summary report."""
        summary_file = output_dir / "summary_report.txt"
        
        with open(summary_file, 'w') as f:
            f.write("CELEB-DF DATASET ANALYSIS REPORT\n")
            f.write("=" * 40 + "\n\n")
            
            # Basic statistics
            basic_stats = analysis['basic_stats']
            f.write("BASIC STATISTICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Real videos: {basic_stats['real_videos']}\n")
            f.write(f"Fake videos: {basic_stats['fake_videos']}\n")
            f.write(f"Mask videos: {basic_stats['mask_videos']}\n")
            f.write(f"Real images: {basic_stats['real_images']}\n")
            f.write(f"Fake images: {basic_stats['fake_images']}\n")
            f.write(f"Mask images: {basic_stats['mask_images']}\n")
            f.write(f"Matched fake-mask pairs: {basic_stats['matched_fake_mask_pairs']}\n\n")
            
            # Video analysis
            video_analysis = analysis['video_analysis']['summary']
            f.write("VIDEO ANALYSIS:\n")
            f.write("-" * 15 + "\n")
            f.write(f"Average frames per real video: {video_analysis['avg_frames_per_real_video']:.1f}\n")
            f.write(f"Average frames per fake video: {video_analysis['avg_frames_per_fake_video']:.1f}\n")
            f.write(f"Average frames per mask video: {video_analysis['avg_frames_per_mask_video']:.1f}\n\n")
            
            # Split analysis
            split_analysis = analysis['split_analysis']
            f.write("TRAIN/TEST SPLIT ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            for category in ['real', 'fake', 'mask']:
                stats = split_analysis[category]
                f.write(f"{category.upper()}:\n")
                f.write(f"  Train: {stats['train']} ({stats['train_ratio']:.1%})\n")
                f.write(f"  Test:  {stats['test']} ({stats['test_ratio']:.1%})\n")
            f.write("\n")
            
            # Matching analysis
            matching = analysis['matching_analysis']
            f.write("FAKE-MASK MATCHING ANALYSIS:\n")
            f.write("-" * 35 + "\n")
            f.write(f"Total fake images: {matching['total_fake_images']}\n")
            f.write(f"Total mask images: {matching['total_mask_images']}\n")
            f.write(f"Matched pairs: {matching['matched_pairs']}\n")
            f.write(f"Matching ratio: {matching['matching_ratio']:.1%}\n")
            f.write(f"Unmatched fake images: {matching['unmatched_fake']}\n")
            f.write(f"Unmatched mask images: {matching['unmatched_mask']}\n\n")
            
            f.write("TRAINING RECOMMENDATIONS:\n")
            f.write("-" * 30 + "\n")
            total_train_real = split_analysis['real']['train']
            total_train_fake_matched = matching['matched_pairs'] * split_analysis['fake']['train_ratio']
            f.write(f"Estimated training samples:\n")
            f.write(f"  Real: {total_train_real}\n")
            f.write(f"  Fake (with masks): {total_train_fake_matched:.0f}\n")
            f.write(f"  Total: {total_train_real + total_train_fake_matched:.0f}\n")
            
            # Balance recommendations
            if total_train_real > total_train_fake_matched * 1.5:
                f.write(f"\nRECOMMendation: Dataset is imbalanced towards real images.\n")
                f.write(f"Consider applying class weights or data augmentation for fake images.\n")
            elif total_train_fake_matched > total_train_real * 1.5:
                f.write(f"\nRECOMMendation: Dataset is imbalanced towards fake images.\n")
                f.write(f"Consider applying class weights or data augmentation for real images.\n")
            else:
                f.write(f"\nDataset appears reasonably balanced.\n")
    
    def _generate_visualizations(self, analysis: Dict, output_dir: Path):
        """Generate visualization plots."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            
            plt.style.use('default')
            
            # Set up the plotting environment
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Celeb-DF Dataset Analysis', fontsize=16)
            
            # Plot 1: Basic statistics
            ax1 = axes[0, 0]
            basic_stats = analysis['basic_stats']
            categories = ['Real Videos', 'Fake Videos', 'Mask Videos']
            values = [basic_stats['real_videos'], basic_stats['fake_videos'], basic_stats['mask_videos']]
            ax1.bar(categories, values, color=['green', 'red', 'blue'])
            ax1.set_title('Number of Videos by Category')
            ax1.set_ylabel('Count')
            
            # Plot 2: Image counts
            ax2 = axes[0, 1]
            image_categories = ['Real Images', 'Fake Images', 'Mask Images']
            image_values = [basic_stats['real_images'], basic_stats['fake_images'], basic_stats['mask_images']]
            ax2.bar(image_categories, image_values, color=['green', 'red', 'blue'])
            ax2.set_title('Number of Images by Category')
            ax2.set_ylabel('Count')
            ax2.tick_params(axis='x', rotation=45)
            
            # Plot 3: Train/Test split
            ax3 = axes[1, 0]
            split_analysis = analysis['split_analysis']
            categories = list(split_analysis.keys())
            train_counts = [split_analysis[cat]['train'] for cat in categories]
            test_counts = [split_analysis[cat]['test'] for cat in categories]
            
            x = np.arange(len(categories))
            width = 0.35
            ax3.bar(x - width/2, train_counts, width, label='Train', color='lightblue')
            ax3.bar(x + width/2, test_counts, width, label='Test', color='lightcoral')
            ax3.set_xlabel('Category')
            ax3.set_ylabel('Count')
            ax3.set_title('Train/Test Split by Category')
            ax3.set_xticks(x)
            ax3.set_xticklabels(categories)
            ax3.legend()
            
            # Plot 4: Matching analysis
            ax4 = axes[1, 1]
            matching = analysis['matching_analysis']
            matching_data = [
                matching['matched_pairs'],
                matching['unmatched_fake'],
                matching['unmatched_mask']
            ]
            labels = ['Matched Pairs', 'Unmatched Fake', 'Unmatched Mask']
            colors = ['lightgreen', 'orange', 'lightblue']
            ax4.pie(matching_data, labels=labels, colors=colors, autopct='%1.1f%%')
            ax4.set_title('Fake-Mask Matching Distribution')
            
            plt.tight_layout()
            
            # Save the plot
            plot_file = output_dir / "analysis_plots.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Visualizations saved to {plot_file}")
            
        except ImportError:
            print("Warning: matplotlib/seaborn not available. Skipping visualizations.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Analyze Celeb-DF Dataset')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of Celeb-DF dataset')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='analysis_reports',
                       help='Output directory for reports')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = CelebDFAnalyzer(args.data_root, args.config)
    
    # Generate comprehensive report
    analyzer.generate_report(args.output_dir)


if __name__ == '__main__':
    main()
