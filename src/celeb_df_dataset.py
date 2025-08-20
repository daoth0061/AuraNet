"""
Celeb-DF Dataset Loader for AuraNet
Handles the specific structure and sampling strategy for Celeb-DF dataset
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import json
import numpy as np
import cv2
import yaml
import glob
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import re


class CelebDFDataset(Dataset):
    """Dataset class for Celeb-DF with specific sampling strategy."""
    
    def __init__(self, data_root: str, mode: str = 'finetune', split: str = 'train', 
                 config: Optional[Dict] = None, transform=None):
        """
        Args:
            data_root: Root directory containing Celeb_DF_Real, Celeb_DF_Fake, Celeb_DF_Mask
            mode: 'pretrain' or 'finetune'
            split: 'train' or 'test'
            config: Configuration dictionary
            transform: Optional transform to apply
        """
        self.data_root = Path(data_root)
        self.mode = mode
        self.split = split
        
        # Load config
        if config is None:
            config_path = Path(__file__).parent.parent / 'config_celeb_df.yaml'
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        self.config = config
        
        # Dataset parameters
        self.dataset_config = config['dataset']
        self.img_size = tuple(config['img_size'])
        self.split_ratio = self.dataset_config['split_ratio']
        self.frame_modulo = self.dataset_config['frame_modulo']
        self.split_modulo = self.dataset_config['split_modulo']
        self.test_remainder = self.dataset_config['test_remainder']
        
        # Build dataset
        self.samples = self._build_dataset()
        
        # Setup transforms
        if transform is None:
            self.transform = self._get_default_transforms()
        else:
            self.transform = transform
        
        logging.info(f"Created CelebDF {split} dataset with {len(self.samples)} samples")
    
    def _build_dataset(self) -> List[Dict]:
        """Build dataset with specific sampling strategy."""
        samples = []
        
        # Get subdirectories
        real_dir = self.data_root / self.dataset_config['subfolders']['real']
        fake_dir = self.data_root / self.dataset_config['subfolders']['fake']
        mask_dir = self.data_root / self.dataset_config['subfolders']['mask']
        
        # Process real images
        if real_dir.exists():
            real_samples = self._process_real_images(real_dir)
            samples.extend(real_samples)
        
        # Process fake images (matched with masks)
        if fake_dir.exists() and mask_dir.exists():
            fake_samples = self._process_fake_images(fake_dir, mask_dir)
            samples.extend(fake_samples)
        
        # Apply train/test split based on frame ID
        filtered_samples = self._apply_split_strategy(samples)
        
        return filtered_samples
    
    def _process_real_images(self, real_dir: Path) -> List[Dict]:
        """Process real images."""
        samples = []
        
        # Get all subdirectories (video IDs)
        for video_dir in real_dir.iterdir():
            if not video_dir.is_dir():
                continue
                
            # Get all images in this video directory
            image_files = list(video_dir.glob('*.jpg')) + list(video_dir.glob('*.png'))
            
            for image_path in image_files:
                samples.append({
                    'image_path': str(image_path),
                    'mask_path': None,  # No mask for real images
                    'label': 0,  # Real
                    'video_id': video_dir.name,
                    'frame_info': self._extract_frame_info(image_path.name)
                })
        
        return samples
    
    def _process_fake_images(self, fake_dir: Path, mask_dir: Path) -> List[Dict]:
        """Process fake images, only including those with corresponding masks."""
        samples = []
        
        # Get all mask files first to determine which fake images to include
        mask_files = {}
        for video_dir in mask_dir.iterdir():
            if not video_dir.is_dir():
                continue
            
            for mask_path in video_dir.glob('*.jpg'):
                # Create relative path as key
                rel_path = mask_path.relative_to(mask_dir)
                mask_files[str(rel_path)] = mask_path
        
        # Process fake images that have corresponding masks
        for video_dir in fake_dir.iterdir():
            if not video_dir.is_dir():
                continue
            
            for image_path in video_dir.glob('*.jpg'):
                # Check if corresponding mask exists
                rel_path = image_path.relative_to(fake_dir)
                
                if str(rel_path) in mask_files:
                    samples.append({
                        'image_path': str(image_path),
                        'mask_path': str(mask_files[str(rel_path)]),
                        'label': 1,  # Fake
                        'video_id': video_dir.name,
                        'frame_info': self._extract_frame_info(image_path.name)
                    })
        
        return samples
    
    def _extract_frame_info(self, filename: str) -> Dict:
        """Extract frame information from filename."""
        # Pattern: cropped_face_X_frame_Y.jpg
        pattern = r'cropped_face_(\d+)_frame_(\d+)\.jpg'
        match = re.match(pattern, filename)
        
        if match:
            face_id = int(match.group(1))
            frame_id = int(match.group(2))
            return {
                'face_id': face_id,
                'frame_id': frame_id,
                'filename': filename
            }
        else:
            return {
                'face_id': 0,
                'frame_id': 0,
                'filename': filename
            }
    
    def _apply_split_strategy(self, samples: List[Dict]) -> List[Dict]:
        """Apply train/test split based on frame ID modulo strategy."""
        filtered_samples = []
        
        for sample in samples:
            frame_id = sample['frame_info']['frame_id']
            
            # Apply sampling strategy: (frame_id // 10) % 4, if remainder == 0 -> test, else -> train
            remainder = (frame_id // self.frame_modulo) % self.split_modulo
            is_test = (remainder == self.test_remainder)
            
            # Include sample based on requested split
            if (self.split == 'test' and is_test) or (self.split == 'train' and not is_test):
                filtered_samples.append(sample)
        
        return filtered_samples
    
    def _get_default_transforms(self):
        """Get default transforms based on mode and split."""
        aug_config = self.config['data_augmentation']
        
        if self.split == 'train':
            # Basic transforms for all samples
            basic_transform_list = [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=aug_config['mean'], std=aug_config['std'])
            ]
            
            # Augmentation transforms for real samples only
            aug_transform_list = [
                transforms.Resize(self.img_size),
                transforms.RandomHorizontalFlip(p=aug_config['horizontal_flip_prob']),
                transforms.RandomRotation(degrees=aug_config['rotation_degrees']),
            ]
            
            # Add color jitter only for pre-training on real samples
            if self.mode == 'pretrain':
                aug_transform_list.append(
                    transforms.ColorJitter(
                        brightness=aug_config['brightness'], 
                        contrast=aug_config['contrast'], 
                        saturation=aug_config['saturation'], 
                        hue=aug_config['hue']
                    )
                )
            
            aug_transform_list.extend([
                transforms.ToTensor(),
                transforms.Normalize(mean=aug_config['mean'], std=aug_config['std'])
            ])
            
            # Store both transforms
            self.basic_transform = transforms.Compose(basic_transform_list)
            self.aug_transform = transforms.Compose(aug_transform_list)
            
            return None  # We'll handle transforms in __getitem__
        else:
            # Test transforms without augmentation
            transform_list = [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=aug_config['mean'], std=aug_config['std'])
            ]
            
            return transforms.Compose(transform_list)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a single sample."""
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Apply appropriate transform based on sample type and split
        if self.split == 'train' and hasattr(self, 'aug_transform'):
            # For training: apply augmentation only to real samples (label = 0)
            if sample['label'] == 0:  # Real sample
                image_tensor = self.aug_transform(image)
            else:  # Fake sample
                image_tensor = self.basic_transform(image)
        else:
            # For test or when no augmentation transforms are defined
            image_tensor = self.transform(image)
        
        if self.mode == 'pretrain':
            return self._get_pretrain_sample(image_tensor, sample)
        else:
            return self._get_finetune_sample(image_tensor, sample)
    
    def _get_pretrain_sample(self, image_tensor: torch.Tensor, sample: Dict):
        """Prepare sample for pre-training."""
        B, H, W = 1, self.img_size[0], self.img_size[1]
        
        # Create random mask
        mask_ratio = self.config['training']['pretrain']['mask_ratio']
        block_size = self.config['model']['block_size']
        mask = self._create_random_mask((B, H, W), mask_ratio, block_size)
        mask = mask.squeeze(0)  # Remove batch dimension
        
        # Create masked image
        masked_image = image_tensor * (1 - mask)
        
        # Load ground truth mask if available
        if sample['mask_path'] is not None:
            gt_mask = Image.open(sample['mask_path']).convert('L')
            gt_mask = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor()
            ])(gt_mask)
        else:
            # Create dummy mask for real images
            gt_mask = torch.zeros(1, H, W)
        
        return {
            'image': masked_image,
            'original_image': image_tensor,
            'mask': mask,
            'ground_truth_mask': gt_mask,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'video_id': sample['video_id'],
            'frame_info': sample['frame_info']
        }
    
    def _get_finetune_sample(self, image_tensor: torch.Tensor, sample: Dict):
        """Prepare sample for fine-tuning."""
        H, W = self.img_size
        
        # Load mask if available
        if sample['mask_path'] is not None:
            mask = Image.open(sample['mask_path']).convert('L')
            mask = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor()
            ])(mask)
        else:
            # Create dummy mask for real images
            mask = torch.zeros(1, H, W)
        
        return {
            'image': image_tensor,
            'mask': mask,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'video_id': sample['video_id'],
            'frame_info': sample['frame_info']
        }
    
    def _create_random_mask(self, shape: Tuple, mask_ratio: float, block_size: int):
        """Create random mask for pre-training."""
        B, H, W = shape
        
        # Create block-wise mask
        mask_h = H // block_size
        mask_w = W // block_size
        
        # Create mask at block level
        block_mask = torch.rand(B, mask_h, mask_w) < mask_ratio
        
        # Upsample to full resolution
        mask = torch.nn.functional.interpolate(
            block_mask.float().unsqueeze(1), 
            size=(H, W), 
            mode='nearest'
        )
        
        return mask


def create_celeb_df_dataloaders(data_root: str, mode: str = 'finetune', 
                               config: Optional[Dict] = None) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for Celeb-DF dataset.
    
    Args:
        data_root: Root directory of Celeb-DF dataset
        mode: 'pretrain' or 'finetune'
        config: Configuration dictionary
        
    Returns:
        train_loader, test_loader: DataLoader objects
    """
    # Load config if not provided
    if config is None:
        config_path = Path(__file__).parent.parent / 'config_celeb_df.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Get batch size based on mode
    if mode == 'pretrain':
        batch_size = config['training']['pretrain']['batch_size']
    else:
        batch_size = config['training']['finetune']['batch_size']
    
    # Data loading parameters
    data_config = config['data_loading']
    
    # Create datasets
    train_dataset = CelebDFDataset(
        data_root=data_root,
        mode=mode,
        split='train',
        config=config
    )
    
    test_dataset = CelebDFDataset(
        data_root=data_root,
        mode=mode,
        split='test',
        config=config
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        drop_last=data_config['drop_last'],
        prefetch_factor=data_config.get('prefetch_factor', 2),
        persistent_workers=data_config.get('persistent_workers', True)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        prefetch_factor=data_config.get('prefetch_factor', 2),
        persistent_workers=data_config.get('persistent_workers', True)
    )
    
    return train_loader, test_loader


def analyze_celeb_df_dataset(data_root: str) -> Dict:
    """
    Analyze the Celeb-DF dataset structure and provide statistics.
    
    Args:
        data_root: Root directory of Celeb-DF dataset
        
    Returns:
        Dictionary with dataset statistics
    """
    data_root = Path(data_root)
    stats = {
        'real_videos': 0,
        'fake_videos': 0,
        'mask_videos': 0,
        'real_images': 0,
        'fake_images': 0,
        'mask_images': 0,
        'matched_fake_mask_pairs': 0
    }
    
    # Analyze real images
    real_dir = data_root / 'Celeb_DF_Real'
    if real_dir.exists():
        for video_dir in real_dir.iterdir():
            if video_dir.is_dir():
                stats['real_videos'] += 1
                stats['real_images'] += len(list(video_dir.glob('*.jpg')))
    
    # Analyze fake images
    fake_dir = data_root / 'Celeb_DF_Fake'
    if fake_dir.exists():
        for video_dir in fake_dir.iterdir():
            if video_dir.is_dir():
                stats['fake_videos'] += 1
                stats['fake_images'] += len(list(video_dir.glob('*.jpg')))
    
    # Analyze mask images
    mask_dir = data_root / 'Celeb_DF_Mask'
    mask_files = set()
    if mask_dir.exists():
        for video_dir in mask_dir.iterdir():
            if video_dir.is_dir():
                stats['mask_videos'] += 1
                for mask_file in video_dir.glob('*.jpg'):
                    stats['mask_images'] += 1
                    # Store relative path for matching
                    rel_path = mask_file.relative_to(mask_dir)
                    mask_files.add(str(rel_path))
    
    # Count matched fake-mask pairs
    if fake_dir.exists():
        for video_dir in fake_dir.iterdir():
            if video_dir.is_dir():
                for fake_file in video_dir.glob('*.jpg'):
                    rel_path = fake_file.relative_to(fake_dir)
                    if str(rel_path) in mask_files:
                        stats['matched_fake_mask_pairs'] += 1
    
    return stats


if __name__ == "__main__":
    # Example usage and testing
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Celeb-DF Dataset Loader')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of Celeb-DF dataset')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze dataset statistics')
    
    args = parser.parse_args()
    
    if args.analyze:
        print("Analyzing Celeb-DF dataset...")
        stats = analyze_celeb_df_dataset(args.data_root)
        print(f"Dataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print()
    
    # Test data loaders
    print("Testing data loaders...")
    train_loader, test_loader = create_celeb_df_dataloaders(
        data_root=args.data_root,
        mode='finetune'
    )
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")
    
    # Test one batch
    for batch in train_loader:
        print(f"Sample batch shapes:")
        for key, value in batch.items():
            if torch.is_tensor(value):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)} (length: {len(value)})")
        break
