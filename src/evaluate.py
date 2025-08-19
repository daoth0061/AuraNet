"""
Evaluation and inference utilities for AuraNet
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import json

from .auranet import create_auranet


class AuraNetEvaluator:
    """Evaluation and inference utilities for AuraNet."""
    
    def __init__(self, config_path, checkpoint_path, device='cuda'):
        """
        Args:
            config_path: str, path to model configuration file
            checkpoint_path: str, path to trained model checkpoint
            device: str, device to use for inference
        """
        self.device = device
        
        # Load model
        self.model = create_auranet(config_path=config_path)
        self._load_checkpoint(checkpoint_path)
        self.model.to(device)
        self.model.eval()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}")
    
    def predict_single_image(self, image_path):
        """
        Predict on a single image.
        
        Args:
            image_path: str, path to the image file
            
        Returns:
            dict containing predictions
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model(input_tensor, mode='finetune')
            
            # Process outputs
            logits = outputs['classification_logits']
            probs = F.softmax(logits, dim=1)
            pred_label = torch.argmax(logits, dim=1).item()
            confidence = probs.max().item()
            
            # Get segmentation mask
            seg_mask = outputs['segmentation_mask'].squeeze(0).squeeze(0).cpu().numpy()
            
        results = {
            'predicted_label': pred_label,
            'confidence': confidence,
            'probabilities': {
                'real': probs[0, 0].item(),
                'fake': probs[0, 1].item()
            },
            'segmentation_mask': seg_mask,
            'prediction': 'fake' if pred_label == 1 else 'real'
        }
        
        return results
    
    def evaluate_dataset(self, data_loader):
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader: DataLoader for the dataset
            
        Returns:
            dict containing evaluation metrics
        """
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0
        
        print("Evaluating model...")
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluation"):
                # Move batch to device
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(self.device)
                
                # Forward pass
                outputs = self.model(batch['image'], mode='finetune')
                
                # Process predictions
                logits = outputs['classification_logits']
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                # Collect results
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Compute metrics
        metrics = self._compute_detailed_metrics(all_preds, all_labels, all_probs)
        
        return metrics
    
    def _compute_detailed_metrics(self, preds, labels, probs):
        """Compute detailed evaluation metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(labels, preds)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, average=None, labels=[0, 1]
        )
        
        metrics['precision_real'] = precision[0]
        metrics['precision_fake'] = precision[1]
        metrics['recall_real'] = recall[0]
        metrics['recall_fake'] = recall[1]
        metrics['f1_real'] = f1[0]
        metrics['f1_fake'] = f1[1]
        
        # Average metrics
        metrics['precision_avg'] = np.mean(precision)
        metrics['recall_avg'] = np.mean(recall)
        metrics['f1_avg'] = np.mean(f1)
        
        # AUC
        if probs.shape[1] > 1:
            metrics['auc'] = roc_auc_score(labels, probs[:, 1])
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        metrics['classification_report'] = classification_report(
            labels, preds, target_names=['Real', 'Fake'], output_dict=True
        )
        
        return metrics
    
    def visualize_prediction(self, image_path, save_path=None):
        """
        Visualize prediction results.
        
        Args:
            image_path: str, path to the image file
            save_path: str, optional path to save the visualization
        """
        # Get prediction
        results = self.predict_single_image(image_path)
        
        # Load original image
        original_image = Image.open(image_path).convert('RGB')
        original_array = np.array(original_image)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_array)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Segmentation mask
        seg_mask = results['segmentation_mask']
        axes[1].imshow(seg_mask, cmap='hot', alpha=0.7)
        axes[1].imshow(original_array, alpha=0.3)
        axes[1].set_title('Manipulation Mask')
        axes[1].axis('off')
        
        # Prediction info
        pred_text = f"Prediction: {results['prediction'].upper()}\n"
        pred_text += f"Confidence: {results['confidence']:.3f}\n"
        pred_text += f"Real Prob: {results['probabilities']['real']:.3f}\n"
        pred_text += f"Fake Prob: {results['probabilities']['fake']:.3f}"
        
        axes[2].text(0.1, 0.5, pred_text, transform=axes[2].transAxes, 
                    fontsize=14, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[2].axis('off')
        axes[2].set_title('Prediction Results')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
    
    def plot_confusion_matrix(self, metrics, save_path=None):
        """Plot confusion matrix."""
        cm = np.array(metrics['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Real', 'Fake'],
                   yticklabels=['Real', 'Fake'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def batch_inference(self, image_dir, output_file):
        """
        Run inference on a batch of images.
        
        Args:
            image_dir: str, directory containing images
            output_file: str, path to save results JSON
        """
        results = {}
        
        # Get all image files
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Processing {len(image_files)} images...")
        
        for img_file in tqdm(image_files):
            img_path = os.path.join(image_dir, img_file)
            try:
                pred_results = self.predict_single_image(img_path)
                # Remove numpy arrays for JSON serialization
                results[img_file] = {
                    'predicted_label': pred_results['predicted_label'],
                    'confidence': pred_results['confidence'],
                    'probabilities': pred_results['probabilities'],
                    'prediction': pred_results['prediction']
                }
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                results[img_file] = {'error': str(e)}
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_file}")
        
        return results


def main():
    """Example usage of the evaluator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate AuraNet')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, choices=['single', 'dataset', 'batch'], 
                       default='single', help='Evaluation mode')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--image_dir', type=str, help='Directory of images for batch processing')
    parser.add_argument('--data_root', type=str, help='Data root for dataset evaluation')
    parser.add_argument('--annotations', type=str, help='Annotations file for dataset evaluation')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = AuraNetEvaluator(args.config, args.checkpoint, args.device)
    
    if args.mode == 'single':
        if not args.image:
            print("Please provide --image for single image evaluation")
            return
        
        results = evaluator.predict_single_image(args.image)
        print(f"Prediction: {results['prediction']}")
        print(f"Confidence: {results['confidence']:.3f}")
        print(f"Probabilities: {results['probabilities']}")
        
        # Visualize if needed
        if args.output:
            evaluator.visualize_prediction(args.image, args.output)
    
    elif args.mode == 'batch':
        if not args.image_dir or not args.output:
            print("Please provide --image_dir and --output for batch processing")
            return
        
        results = evaluator.batch_inference(args.image_dir, args.output)
        print(f"Processed {len(results)} images")
    
    elif args.mode == 'dataset':
        if not args.data_root or not args.annotations:
            print("Please provide --data_root and --annotations for dataset evaluation")
            return
        
        from .data_loader import create_data_loaders
        
        # Create data loader
        _, val_loader = create_data_loaders(
            args.data_root, args.data_root, args.annotations, args.annotations,
            batch_size=32, num_workers=4, mode='finetune'
        )
        
        # Evaluate
        metrics = evaluator.evaluate_dataset(val_loader)
        
        print("Evaluation Results:")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"F1 Score: {metrics['f1_avg']:.3f}")
        print(f"AUC: {metrics['auc']:.3f}")
        
        # Save detailed results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Detailed results saved to {args.output}")


if __name__ == '__main__':
    main()
