"""
Data loading and preprocessing utilities for AuraNet
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import numpy as np
import cv2
import yaml
import sys
from pathlib import Path

# Import mask utilities
sys.path.append(str(Path(__file__).parent.parent))
from mask_utils import create_random_mask, apply_mask_to_image


class AuraNetDataset(Dataset):
    """Dataset for AuraNet training and evaluation."""
    
    def __init__(self, data_root, annotation_file, mode='finetune', transform=None, 
                 config=None):
        """
        Args:
            data_root: str, root directory of the dataset
            annotation_file: str, path to annotation file (JSON format)
            mode: str, 'pretrain' or 'finetune'
            transform: torchvision transforms
            config: dict or str, configuration parameters or path to config file
        """
        self.data_root = data_root
        self.mode = mode
        
        # Load config if provided
        if config is not None:
            if isinstance(config, str):
                with open(config, 'r') as f:
                    self.config = yaml.safe_load(f)
            else:
                self.config = config
        else:
            # Load default config
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        # Extract parameters from config
        self.img_size = tuple(self.config['img_size'])
        self.mask_ratio = self.config['training']['pretrain']['mask_ratio']
        self.patch_size = self.config['model'].get('patch_size', 32)  # Use patch_size consistently
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Set up transforms
        if transform is None:
            self.transform = self._get_default_transforms()
        else:
            self.transform = transform
    
    def _get_default_transforms(self):
        """Get default transforms based on mode."""
        aug_config = self.config['data_augmentation']
        
        if self.mode == 'pretrain':
            # More aggressive augmentation for pre-training
            return transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.RandomHorizontalFlip(p=aug_config['horizontal_flip_prob']),
                transforms.RandomRotation(degrees=aug_config['rotation_degrees']),
                transforms.ColorJitter(
                    brightness=aug_config['brightness'], 
                    contrast=aug_config['contrast'], 
                    saturation=aug_config['saturation'], 
                    hue=aug_config['hue']
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=aug_config['mean'], std=aug_config['std'])
            ])
        else:
            # Standard augmentation for fine-tuning
            return transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.RandomHorizontalFlip(p=aug_config['horizontal_flip_prob']),
                transforms.ToTensor(),
                transforms.Normalize(mean=aug_config['mean'], std=aug_config['std'])
            ])
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """Get a single sample."""
        annotation = self.annotations[idx]
        
        # Load image
        img_path = os.path.join(self.data_root, annotation['image'])
        image = Image.open(img_path).convert('RGB')
        
        if self.mode == 'pretrain':
            return self._get_pretrain_sample(image, annotation)
        else:
            return self._get_finetune_sample(image, annotation)
    
    def _get_pretrain_sample(self, image, annotation):
        """Prepare sample for pre-training."""
        # Apply transforms
        original_image = self.transform(image)
        
        # Create random mask
        B, H, W = 1, self.img_size[0], self.img_size[1]
        mask = self._create_random_mask((B, H, W), self.mask_ratio)
        
        # Create masked image using apply_mask_to_image
        masked_image = apply_mask_to_image(original_image.unsqueeze(0), mask, 'spatial').squeeze(0)
        
        # Create ground truth mask (for simplicity, we'll use a random binary mask)
        # In practice, this might be derived from manipulation annotations
        gt_mask = torch.rand(1, H, W) > 0.5
        gt_mask = gt_mask.float()
        
        # Get pseudo-label for contrastive learning
        label = annotation.get('label', 0)  # Default to 0 if no label
        
        sample = {
            'image': masked_image,
            'original_image': original_image,
            'mask': mask.squeeze(0),  # Remove batch dimension
            'ground_truth_mask': gt_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }
        
        return sample
    
    def _get_finetune_sample(self, image, annotation):
        """Prepare sample for fine-tuning."""
        # Apply transforms
        image_tensor = self.transform(image)
        
        # Load mask if available
        mask_path = annotation.get('mask', None)
        if mask_path and os.path.exists(os.path.join(self.data_root, mask_path)):
            mask = Image.open(os.path.join(self.data_root, mask_path)).convert('L')
            mask = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor()
            ])(mask)
        else:
            # Create dummy mask if not available
            mask = torch.zeros(1, *self.img_size)
        
        # Get label
        label = annotation.get('label', 0)
        
        sample = {
            'image': image_tensor,
            'mask': mask,
            'label': torch.tensor(label, dtype=torch.long)
        }
        
        return sample
    
    def _create_random_mask(self, shape, mask_ratio):
        """Create random mask for pre-training."""
        B, H, W = shape
        patch_size = self.patch_size
        
        # Use the imported function from mask_utils
        patch_mask, spatial_mask = create_random_mask(shape, mask_ratio, patch_size)
        
        # Return the spatial mask directly (already has the right shape)
        return spatial_mask


def create_data_loaders(train_data_root, val_data_root, train_annotations, val_annotations,
                       mode='finetune', config=None):
    """
    Create data loaders for training and validation.
    
    Args:
        train_data_root: str, training data root directory
        val_data_root: str, validation data root directory
        train_annotations: str, path to training annotations
        val_annotations: str, path to validation annotations
        mode: str, 'pretrain' or 'finetune'
        config: dict or str, configuration parameters or path to config file
        
    Returns:
        train_loader, val_loader: DataLoader objects
    """
    # Load config if provided
    if config is not None:
        if isinstance(config, str):
            with open(config, 'r') as f:
                config = yaml.safe_load(f)
    else:
        # Load default config
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Extract parameters from config
    if mode == 'pretrain':
        batch_size = config['training']['pretrain']['batch_size']
    else:
        batch_size = config['training']['finetune']['batch_size']
    
    num_workers = config['data_loading']['num_workers']
    pin_memory = config['data_loading']['pin_memory']
    drop_last = config['data_loading']['drop_last']
    img_size = tuple(config['img_size'])
    
    # Create datasets
    train_dataset = AuraNetDataset(
        data_root=train_data_root,
        annotation_file=train_annotations,
        mode=mode,
        config=config
    )
    
    val_dataset = AuraNetDataset(
        data_root=val_data_root,
        annotation_file=val_annotations,
        mode=mode,
        config=config
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


def collate_fn(batch):
    """Custom collate function for batching."""
    # Default collate for now, can be customized if needed
    return torch.utils.data.dataloader.default_collate(batch)


# Example annotation file format:
ANNOTATION_FORMAT_EXAMPLE = """
[
    {
        "image": "path/to/image.jpg",
        "mask": "path/to/mask.png",  # Optional, for fine-tuning
        "label": 0,  # 0 for real, 1 for fake
        "metadata": {
            "manipulation_type": "deepfake",
            "source": "dataset_name"
        }
    },
    ...
]
"""


def create_sample_annotations(data_root, output_file):
    """
    Create sample annotation file from a directory structure.
    Assumes structure: data_root/{real,fake}/*.jpg
    
    Args:
        data_root: str, root directory with real/fake subdirectories
        output_file: str, path to output annotation file
    """
    annotations = []
    
    # Process real images
    real_dir = os.path.join(data_root, 'real')
    if os.path.exists(real_dir):
        for img_file in os.listdir(real_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                annotations.append({
                    'image': f'real/{img_file}',
                    'label': 0,
                    'metadata': {'manipulation_type': 'none', 'source': 'real'}
                })
    
    # Process fake images
    fake_dir = os.path.join(data_root, 'fake')
    if os.path.exists(fake_dir):
        for img_file in os.listdir(fake_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                annotations.append({
                    'image': f'fake/{img_file}',
                    'label': 1,
                    'metadata': {'manipulation_type': 'unknown', 'source': 'fake'}
                })
    
    # Save annotations
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Created {len(annotations)} annotations in {output_file}")
    
    return annotations
