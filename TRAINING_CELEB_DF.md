# AuraNet Training on Celeb-DF Dataset

This document provides comprehensive instructions for training AuraNet on the Celeb-DF dataset with multi-GPU support and the specified sampling strategy.

## Dataset Structure

The Celeb-DF dataset should have the following structure:
```
celeb-df-dataset/
├── celeb-df-real/
│   ├── video_id_1/
│   │   ├── cropped_face_1_frame_100.jpg
│   │   ├── cropped_face_1_frame_110.jpg
│   │   └── ...
│   ├── video_id_2/
│   └── ...
├── celeb-df-fake/
│   ├── video_id_1/
│   │   ├── cropped_face_1_frame_100.jpg
│   │   ├── cropped_face_1_frame_110.jpg
│   │   └── ...
│   ├── video_id_2/
│   └── ...
└── celeb-df-mask/
    ├── video_id_1/
    │   ├── cropped_face_1_frame_100.jpg
    │   ├── cropped_face_1_frame_110.jpg
    │   └── ...
    ├── video_id_2/
    └── ...
```

## Train/Test Split Strategy

The implemented sampling strategy follows your specifications:
- **Split Ratio**: 75% train, 25% test
- **Sampling Method**: Based on frame ID from filename
- **Logic**: `frame_id // 10 % 10`
  - If remainder == 4: Test set
  - Otherwise: Train set
- **Fake Image Matching**: Only fake images with corresponding masks are used

## Quick Start

### 1. Analyze Dataset
First, analyze your dataset structure and statistics:
```bash
python analyze_celeb_df.py --data_root /path/to/celeb-df-dataset
```

### 2. Single GPU Training (Fine-tuning only)
```bash
python launch_training.py --data_root /path/to/celeb-df-dataset --mode finetune --gpus 1
```

### 3. Multi-GPU Training (Fine-tuning only)
```bash
python launch_training.py --data_root /path/to/celeb-df-dataset --mode finetune --gpus 4
```

### 4. Complete Training Pipeline (Pre-training + Fine-tuning)
```bash
python launch_training.py --data_root /path/to/celeb-df-dataset --mode both --gpus 4
```

## Detailed Training Options

### Configuration Files
- `config_celeb_df.yaml`: Main configuration for Celeb-DF training
- Automatically handles dataset-specific parameters
- Optimized hyperparameters for face manipulation detection

### Training Modes

#### Fine-tuning Only (Recommended)
```bash
python launch_training.py \
    --data_root /path/to/celeb-df-dataset \
    --mode finetune \
    --gpus 4 \
    --batch_size 8 \
    --epochs 30
```

#### Pre-training + Fine-tuning
```bash
python launch_training.py \
    --data_root /path/to/celeb-df-dataset \
    --mode both \
    --gpus 4 \
    --batch_size 16
```

#### Pre-training Only
```bash
python launch_training.py \
    --data_root /path/to/celeb-df-dataset \
    --mode pretrain \
    --gpus 4 \
    --batch_size 16 \
    --epochs 50
```

### Multi-GPU Configuration

#### Automatic GPU Detection
```bash
python launch_training.py --data_root /path/to/data --mode finetune
# Uses all available GPUs
```

#### Specify Number of GPUs
```bash
python launch_training.py --data_root /path/to/data --mode finetune --gpus 2
```

#### Specify Specific GPU IDs
```bash
python launch_training.py --data_root /path/to/data --mode finetune --gpu_ids "0,2,3"
```

### Advanced Options

#### Custom Hyperparameters
```bash
python launch_training.py \
    --data_root /path/to/celeb-df-dataset \
    --mode finetune \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --epochs 50
```

#### Resume Training
```bash
python launch_training.py \
    --data_root /path/to/celeb-df-dataset \
    --mode finetune \
    --resume_from checkpoints/celeb_df/best_model.pth
```

#### Dry Run (Test Configuration)
```bash
python launch_training.py \
    --data_root /path/to/celeb-df-dataset \
    --mode finetune \
    --dry_run
```

## Configuration Details

### Key Configuration Parameters

```yaml
# Dataset Configuration
dataset:
  data_root: "path/to/celeb-df-dataset"
  split_ratio: 0.75  # 75% train, 25% test
  frame_modulo: 10   # Frame ID sampling parameter
  test_remainder: 4  # Test set condition

# Training Parameters
training:
  finetune:
    batch_size: 8      # Per GPU batch size
    encoder_lr: 1e-5   # Pre-trained encoder learning rate
    head_lr: 1e-3      # New heads learning rate
    epochs: 30         # Number of epochs
    patience: 8        # Early stopping patience

# Multi-GPU Settings
distributed:
  enabled: true
  backend: "nccl"
```

### Data Augmentation (Face-Optimized)
```yaml
data_augmentation:
  horizontal_flip_prob: 0.5
  rotation_degrees: 10      # Reduced for faces
  brightness: 0.1           # Reduced for faces
  contrast: 0.1
  saturation: 0.1
  hue: 0.05
```

## Monitoring and Logging

### TensorBoard
Logs are automatically saved to `logs/celeb_df/`. View with:
```bash
tensorboard --logdir logs/celeb_df/
```

### Weights & Biases (Optional)
Enable in configuration:
```yaml
logging:
  wandb:
    enabled: true
    project: "auranet-celeb-df"
    entity: "your_username"
```

### Checkpoints
- Saved to `checkpoints/celeb_df/`
- Best model saved as `best_model.pth`
- Regular checkpoints every 5 epochs
- Automatic cleanup of old checkpoints

## Performance Optimization

### Memory Optimization
```yaml
# Enable mixed precision training
mixed_precision: true

# Optimize data loading
data_loading:
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2
  persistent_workers: true
```

### Model Compilation (PyTorch 2.0+)
```yaml
compile_model: true  # Enable for faster training
```

## Troubleshooting

### Common Issues

#### GPU Memory Issues
- Reduce batch size: `--batch_size 4`
- Enable mixed precision in config: `mixed_precision: true`
- Reduce image size in config: `img_size: [224, 224]`

#### Dataset Loading Issues
```bash
# Check dataset structure
python analyze_celeb_df.py --data_root /path/to/data

# Test data loading
python -c "
from src.celeb-df-dataset import create_celeb-df-dataloaders
train_loader, test_loader = create_celeb-df-dataloaders('/path/to/data')
print(f'Train batches: {len(train_loader)}, Test batches: {len(test_loader)}')
"
```

#### Multi-GPU Issues
- Ensure NCCL is properly installed
- Check GPU visibility: `nvidia-smi`
- Use single GPU for debugging: `--gpus 1`

### Environment Setup
```bash
# Install dependencies
pip install torch torchvision torchaudio
pip install pytorch-metric-learning pytorch-msssim PyWavelets
pip install tensorboardX wandb
pip install matplotlib seaborn pandas

# For distributed training
pip install ninja  # For faster compilation
```

## Expected Performance

### Dataset Statistics (Typical Celeb-DF)
- ~13k real videos, ~6k fake videos
- ~500k+ total images
- ~75% for training, 25% for testing
- Only fake images with corresponding masks used

### Training Time Estimates (4 GPUs)
- **Fine-tuning**: ~6-8 hours (30 epochs)
- **Pre-training**: ~12-15 hours (50 epochs)
- **Full pipeline**: ~18-23 hours

### Memory Requirements
- **Single GPU**: 8GB+ VRAM recommended
- **Multi-GPU**: 6GB+ per GPU recommended
- **RAM**: 32GB+ system RAM recommended

## Results and Evaluation

### Metrics Tracked
- Accuracy, Precision, Recall, F1-score
- AUC-ROC, Average Precision
- Segmentation IoU (if using masks)

### Model Outputs
- Binary classification (real/fake)
- Segmentation mask (manipulation localization)
- Confidence scores

### Best Practices
1. **Start with fine-tuning**: Faster and often sufficient
2. **Monitor validation metrics**: Use early stopping
3. **Analyze dataset first**: Understand data distribution
4. **Use mixed precision**: Saves memory and speeds training
5. **Regular checkpointing**: Don't lose progress

## Example Training Script

```bash
#!/bin/bash

# Set dataset path
DATA_ROOT="/path/to/celeb-df-dataset"

# Analyze dataset first
echo "Analyzing dataset..."
python analyze_celeb_df.py --data_root $DATA_ROOT

# Run fine-tuning on 4 GPUs
echo "Starting training..."
python launch_training.py \
    --data_root $DATA_ROOT \
    --mode finetune \
    --gpus 4 \
    --batch_size 8 \
    --epochs 30 \
    --learning_rate 1e-4

echo "Training complete!"
```

For more advanced configurations and custom training loops, refer to the source code in `train_celeb_df.py` and modify as needed.
