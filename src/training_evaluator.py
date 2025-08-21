"""
Training-compatible evaluator for AuraNet
Provides detailed evaluation during training with comprehensive metrics
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from pytorch_msssim import ssim
import cv2
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging


class TrainingEvaluator:
    """Evaluator designed for use during training with comprehensive metrics."""
    
    def __init__(self, config: Dict, mask_gt_dir: Optional[str] = None, device: str = 'cuda'):
        """
        Args:
            config: Training configuration dictionary
            mask_gt_dir: Directory containing ground truth masks for mask evaluation (relative to data_root)
            device: Device to use for evaluation
        """
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Construct full path to mask GT directory
        if mask_gt_dir:
            data_root = Path(config['dataset']['data_root'])
            self.mask_gt_dir = data_root / mask_gt_dir
            
            # Check if ground truth masks are available
            if not self.mask_gt_dir.exists():
                self.logger.warning(f"Mask GT directory not found: {self.mask_gt_dir}")
                self.mask_gt_dir = None
        else:
            self.mask_gt_dir = None
    
    def evaluate_classification(self, predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate classification performance with comprehensive metrics.
        
        Args:
            predictions: (N, num_classes) prediction probabilities
            labels: (N,) ground truth labels
            
        Returns:
            Dictionary containing classification metrics
        """
        # Convert to numpy
        probs = predictions.cpu().numpy()
        labels_np = labels.cpu().numpy()
        preds = np.argmax(probs, axis=1)
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(labels_np, preds)
        
        # Precision, Recall, F1 for each class
        precision, recall, f1, support = precision_recall_fscore_support(
            labels_np, preds, average=None, labels=[0, 1], zero_division=0
        )
        
        metrics['precision_real'] = precision[0] if len(precision) > 0 else 0.0
        metrics['precision_fake'] = precision[1] if len(precision) > 1 else 0.0
        metrics['recall_real'] = recall[0] if len(recall) > 0 else 0.0
        metrics['recall_fake'] = recall[1] if len(recall) > 1 else 0.0
        metrics['f1_real'] = f1[0] if len(f1) > 0 else 0.0
        metrics['f1_fake'] = f1[1] if len(f1) > 1 else 0.0
        
        # Average metrics
        metrics['precision_avg'] = np.mean(precision) if len(precision) > 0 else 0.0
        metrics['recall_avg'] = np.mean(recall) if len(recall) > 0 else 0.0
        metrics['f1_avg'] = np.mean(f1) if len(f1) > 0 else 0.0
        
        # AUC and AOC (Area Over Curve)
        if probs.shape[1] > 1 and len(np.unique(labels_np)) > 1:
            try:
                # AUC (Area Under ROC Curve)
                auc = roc_auc_score(labels_np, probs[:, 1])
                metrics['auc'] = auc
                metrics['aoc'] = 1.0 - auc  # Area Over Curve
                
                # Calculate EER (Equal Error Rate)
                fpr, tpr, thresholds = roc_curve(labels_np, probs[:, 1])
                fnr = 1 - tpr
                eer_idx = np.nanargmin(np.absolute(fpr - fnr))
                metrics['eer'] = fpr[eer_idx]
                
            except ValueError as e:
                self.logger.warning(f"Could not calculate AUC/EER: {e}")
                metrics['auc'] = 0.0
                metrics['aoc'] = 1.0
                metrics['eer'] = 1.0
        else:
            metrics['auc'] = 0.0
            metrics['aoc'] = 1.0
            metrics['eer'] = 1.0
        
        # Confusion matrix
        cm = confusion_matrix(labels_np, preds, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negative'] = int(tn)
            metrics['false_positive'] = int(fp)
            metrics['false_negative'] = int(fn)
            metrics['true_positive'] = int(tp)
            
            # Specificity and Sensitivity
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return metrics
    
    def evaluate_mask_reconstruction(self, pred_masks: torch.Tensor, 
                                   gt_masks: Optional[torch.Tensor] = None,
                                   image_paths: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate mask reconstruction performance.
        
        Args:
            pred_masks: (N, 1, H, W) predicted masks
            gt_masks: (N, 1, H, W) ground truth masks (if available)
            image_paths: List of image paths for loading GT masks from disk
            
        Returns:
            Dictionary containing mask reconstruction metrics
        """
        if gt_masks is None and (image_paths is None or self.mask_gt_dir is None):
            return {'mask_eval_available': False}
        
        metrics = {'mask_eval_available': True}
        
        # If GT masks not provided, try to load from disk
        if gt_masks is None and image_paths is not None:
            gt_masks = self._load_gt_masks(image_paths)
            if gt_masks is None:
                return {'mask_eval_available': False}
        
        # Convert to same device
        if isinstance(gt_masks, torch.Tensor):
            gt_masks = gt_masks.to(pred_masks.device)
        
        # Calculate PSNR
        psnr_values = []
        ssim_values = []
        
        for i in range(pred_masks.shape[0]):
            pred_mask = pred_masks[i]  # (1, H, W)
            gt_mask = gt_masks[i] if isinstance(gt_masks, torch.Tensor) else torch.tensor(gt_masks[i]).to(pred_masks.device)
            
            # Ensure same shape
            if gt_mask.dim() == 2:
                gt_mask = gt_mask.unsqueeze(0)  # Add channel dimension
            
            # Calculate PSNR
            mse = F.mse_loss(pred_mask, gt_mask)
            if mse > 0:
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                psnr_values.append(psnr.item())
            else:
                psnr_values.append(float('inf'))
            
            # Calculate SSIM
            try:
                ssim_val = ssim(pred_mask.unsqueeze(0), gt_mask.unsqueeze(0), 
                               data_range=1.0, size_average=True)
                ssim_values.append(ssim_val.item())
            except Exception as e:
                self.logger.warning(f"SSIM calculation failed: {e}")
                ssim_values.append(0.0)
        
        # Average metrics
        metrics['psnr'] = np.mean(psnr_values) if psnr_values else 0.0
        metrics['ssim'] = np.mean(ssim_values) if ssim_values else 0.0
        metrics['psnr_std'] = np.std(psnr_values) if len(psnr_values) > 1 else 0.0
        metrics['ssim_std'] = np.std(ssim_values) if len(ssim_values) > 1 else 0.0
        
        # Additional mask metrics
        if len(psnr_values) > 0:
            metrics['psnr_min'] = np.min([p for p in psnr_values if p != float('inf')])
            metrics['psnr_max'] = np.max([p for p in psnr_values if p != float('inf')])
        
        return metrics
    
    def _load_gt_masks(self, image_paths: List[str]) -> Optional[np.ndarray]:
        """Load ground truth masks from disk using proper Celeb-DF structure."""
        if not self.mask_gt_dir:
            return None
        
        gt_masks = []
        for img_path in image_paths:
            img_path_obj = Path(img_path)
            
            # Extract video_id and filename from fake image path
            # Expected structure: .../Celeb_DF_Fake/video_id/filename.jpg
            # Corresponding mask: .../Celeb_DF_Mask/video_id/filename.jpg
            if 'Celeb_DF_Fake' in str(img_path_obj):
                # Extract relative path from Celeb_DF_Fake
                parts = img_path_obj.parts
                fake_idx = None
                for i, part in enumerate(parts):
                    if part == 'Celeb_DF_Fake':
                        fake_idx = i
                        break
                
                if fake_idx is not None and len(parts) > fake_idx + 2:
                    video_id = parts[fake_idx + 1]
                    filename = parts[fake_idx + 2]
                    
                    # Construct mask path
                    mask_path = self.mask_gt_dir / video_id / filename
                    
                    if mask_path.exists():
                        try:
                            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                            if mask is not None:
                                # Normalize to [0, 1]
                                mask = mask.astype(np.float32) / 255.0
                                # Resize to match prediction size 
                                target_size = tuple(self.config['img_size'])
                                mask = cv2.resize(mask, target_size)
                                gt_masks.append(mask)
                            else:
                                self.logger.warning(f"Failed to load mask: {mask_path}")
                                return None
                        except Exception as e:
                            self.logger.warning(f"Failed to load mask {mask_path}: {e}")
                            return None
                    else:
                        self.logger.warning(f"Mask not found: {mask_path}")
                        return None
                else:
                    self.logger.warning(f"Cannot parse fake image path structure: {img_path}")
                    return None
            else:
                # For real images, create dummy mask (all zeros)
                target_size = tuple(self.config['img_size'])
                dummy_mask = np.zeros(target_size, dtype=np.float32)
                gt_masks.append(dummy_mask)
        
        return np.array(gt_masks) if gt_masks else None
    
    def evaluate_epoch(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
                      mode: str = 'finetune') -> Dict[str, float]:
        """
        Comprehensive evaluation for one epoch.
        
        Args:
            model: The model to evaluate
            dataloader: DataLoader for evaluation data
            mode: Model mode ('pretrain' or 'finetune')
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        model.eval()
        
        all_predictions = []
        all_labels = []
        all_pred_masks = []
        all_gt_masks = []
        all_image_paths = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Forward pass with correct mode
                outputs = model(batch['image'], mode=mode)
                
                # Extract outputs based on mode
                if mode == 'pretrain':
                    # In pretrain mode, we have reconstructed_mask from frequency features
                    if 'reconstructed_mask' in outputs:
                        all_pred_masks.append(outputs['reconstructed_mask'].cpu())
                    # Use ground_truth_mask from batch for pretrain evaluation
                    if 'ground_truth_mask' in batch:
                        all_gt_masks.append(batch['ground_truth_mask'].cpu())
                else:
                    # In finetune mode, we have classification and segmentation
                    if 'classification_logits' in outputs:
                        predictions = F.softmax(outputs['classification_logits'], dim=1)
                        all_predictions.append(predictions.cpu())
                        all_labels.append(batch['label'].cpu())
                    
                    if 'segmentation_mask' in outputs:
                        all_pred_masks.append(outputs['segmentation_mask'].cpu())
                    
                    # Use mask from batch for finetune evaluation
                    if 'mask' in batch:
                        all_gt_masks.append(batch['mask'].cpu())
                
                # Collect image paths if available
                if 'image_path' in batch:
                    all_image_paths.extend(batch['image_path'])
        
        # Combine all predictions
        metrics = {}
        
        # Classification evaluation
        if all_predictions and all_labels:
            all_predictions = torch.cat(all_predictions, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            class_metrics = self.evaluate_classification(all_predictions, all_labels)
            metrics.update({f'class_{k}': v for k, v in class_metrics.items()})
        
        # Mask evaluation using GT masks from batch (preferred) or disk fallback
        if all_pred_masks and all_gt_masks:
            # Use GT masks from batch - this is the preferred method
            try:
                all_pred_masks = torch.cat(all_pred_masks, dim=0)
                all_gt_masks = torch.cat(all_gt_masks, dim=0)
                
                mask_metrics = self.evaluate_mask_reconstruction(
                    all_pred_masks, 
                    all_gt_masks
                )
                metrics.update({f'mask_{k}': v for k, v in mask_metrics.items()})
                
            except Exception as e:
                self.logger.warning(f"Batch-based mask evaluation failed: {e}")
                metrics['mask_eval_available'] = False
        elif all_pred_masks:
            # Fallback to disk-based GT loading
            try:
                all_pred_masks = torch.cat(all_pred_masks, dim=0)
                
                mask_metrics = self.evaluate_mask_reconstruction(
                    all_pred_masks, 
                    image_paths=all_image_paths if all_image_paths else None
                )
                metrics.update({f'mask_{k}': v for k, v in mask_metrics.items()})
                
            except Exception as e:
                self.logger.warning(f"Disk-based mask evaluation failed: {e}")
                metrics['mask_eval_available'] = False
        
        return metrics
    
    def log_metrics(self, metrics: Dict[str, float], epoch: int, phase: str):
        """Log metrics in a structured format."""
        self.logger.info(f"Epoch {epoch} - {phase} Metrics:")
        
        # Classification metrics
        class_metrics = {k.replace('class_', ''): v for k, v in metrics.items() if k.startswith('class_')}
        if class_metrics:
            self.logger.info(f"  Classification:")
            for key, value in class_metrics.items():
                if isinstance(value, float):
                    self.logger.info(f"    {key}: {value:.4f}")
                else:
                    self.logger.info(f"    {key}: {value}")
        
        # Mask metrics
        mask_metrics = {k.replace('mask_', ''): v for k, v in metrics.items() if k.startswith('mask_')}
        if mask_metrics and mask_metrics.get('eval_available', True):
            self.logger.info(f"  Mask Reconstruction:")
            for key, value in mask_metrics.items():
                if key != 'eval_available' and isinstance(value, float):
                    self.logger.info(f"    {key}: {value:.4f}")