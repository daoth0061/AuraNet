# AuraNet: A Dual-Stream Forensic Network for Face Manipulation Detection

AuraNet is a state-of-the-art deep learning model designed for detecting face manipulation in images. It employs a sophisticated dual-stream architecture that processes images through both spatial and frequency domains, enabling robust detection of various types of face manipulations including deepfakes, face swaps, and other synthetic media.

## 🌟 Key Features

- **Dual-Stream Architecture**: Parallel processing through spatial and frequency domains
- **MSAF Module**: Multi-Scale Artifact Fusion for initial feature extraction
- **HAFT Block**: Hierarchical Adaptive Frequency Transform for sophisticated frequency analysis
- **Cross-Fusion**: Bidirectional attention mechanism between streams
- **Dynamic Selection Fusion**: Adaptive combination of features from both streams
- **Self-Supervised Pre-training**: Robust feature learning from unlabeled data
- **Configurable Architecture**: Easy customization through YAML configuration

## 🏗️ Architecture Overview

```
Input Image (RGB)
├── Spatial Stream
│   ├── Initial Spatial Stem
│   ├── ConvNeXt V2 Blocks (4 stages)
│   └── Cross-Fusion with Frequency Stream
└── Frequency Stream
    ├── MSAF (Multi-Scale Artifact Fusion)
    │   ├── SRM Filters (10 fixed filters)
    │   └── DWT Features (2-level wavelet)
    ├── MBConv Downsample
    ├── HAFT Blocks (4 stages)
    └── Cross-Fusion with Spatial Stream

Final Outputs:
├── Classification: Real/Fake binary prediction
└── Segmentation: Grayscale manipulation mask
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/AuraNet.git
cd AuraNet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the demo to verify installation:
```bash
python -m src.demo
```

### Basic Usage

```python
from src import create_auranet, AuraNetTrainer

# Create model
model = create_auranet(config_path='config.yaml')

# For inference
from src.evaluate import AuraNetEvaluator

evaluator = AuraNetEvaluator(
    config_path='config.yaml',
    checkpoint_path='path/to/trained/model.pth'
)

# Predict on single image
results = evaluator.predict_single_image('path/to/image.jpg')
print(f"Prediction: {results['prediction']}")
print(f"Confidence: {results['confidence']}")
```

## 📊 Training

AuraNet uses a two-stage training approach:

### Stage 1: Self-Supervised Pre-training
```bash
python -m src.train \
    --config config.yaml \
    --mode pretrain \
    --data_root /path/to/data \
    --train_annotations train_annotations.json \
    --val_annotations val_annotations.json \
    --save_dir ./checkpoints
```

### Stage 2: Supervised Fine-tuning
```bash
python -m src.train \
    --config config.yaml \
    --mode finetune \
    --data_root /path/to/data \
    --train_annotations train_annotations.json \
    --val_annotations val_annotations.json \
    --save_dir ./checkpoints \
    --pretrained_checkpoint ./checkpoints/pretrain_best.pth
```

## 📁 Data Format

### Directory Structure
```
data_root/
├── real/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── fake/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### Annotation Format (JSON)
```json
[
  {
    "image": "real/image1.jpg",
    "mask": "masks/image1_mask.png",  // Optional
    "label": 0,  // 0 for real, 1 for fake
    "metadata": {
      "manipulation_type": "none",
      "source": "original"
    }
  }
]
```

## ⚙️ Configuration

The model is highly configurable through YAML files. Key parameters include:

```yaml
# Model architecture
img_size: [256, 256]
depths: [2, 2, 6, 2]  # Blocks per stage
dims: [64, 128, 256, 512]  # Channels per stage

# HAFT parameters
num_haft_levels: [3, 3, 2, 1]
num_radial_bins: 16
context_vector_dim: 64

# Training parameters
training:
  pretrain:
    batch_size: 32
    learning_rate: 1e-4
    epochs: 100
  finetune:
    batch_size: 16
    encoder_lr: 1e-5
    head_lr: 1e-3
    epochs: 50
```

## 🔬 Model Components

### 1. MSAF (Multi-Scale Artifact Fusion)
- **SRM Filters**: 10 fixed steganalysis-inspired filters for artifact detection
- **DWT Features**: 2-level discrete wavelet transform for frequency analysis
- **CBAM Attention**: Channel and spatial attention for feature enhancement

### 2. HAFT (Hierarchical Adaptive Frequency Transform)
- **Multi-Level Context**: Hierarchical patch analysis at different scales
- **Adaptive Filtering**: Learned frequency-domain filters based on context
- **Radial Reconstruction**: 2D filter reconstruction using Chebyshev distance

### 3. Cross-Fusion Block
- **Deformable Attention**: For high-resolution stages (adaptive sampling)
- **Standard Attention**: For low-resolution stages (global attention)
- **Bidirectional Dialogue**: Information exchange between streams

### 4. Output Heads
- **Classification Head**: Binary real/fake prediction with AM-Softmax loss
- **Segmentation Head**: Pixel-level manipulation mask prediction
- **Dynamic Selection Fusion**: Adaptive feature combination

## 📈 Evaluation

### Single Image Prediction
```python
from src.evaluate import AuraNetEvaluator

evaluator = AuraNetEvaluator('config.yaml', 'model.pth')
results = evaluator.predict_single_image('test_image.jpg')

# Visualize results
evaluator.visualize_prediction('test_image.jpg', 'result.png')
```

### Dataset Evaluation
```python
# Create data loader
from src.data_loader import create_data_loaders
_, val_loader = create_data_loaders(
    data_root, data_root, 'train.json', 'val.json', mode='finetune'
)

# Evaluate
metrics = evaluator.evaluate_dataset(val_loader)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"AUC: {metrics['auc']:.3f}")
```

### Batch Processing
```bash
python -m src.evaluate \
    --config config.yaml \
    --checkpoint model.pth \
    --mode batch \
    --image_dir /path/to/images \
    --output results.json
```

## 🧪 Experiments and Results

The model has been designed based on state-of-the-art research in:
- Face manipulation detection
- Frequency-domain analysis
- Multi-modal fusion
- Self-supervised learning

Key innovations:
1. **Dual-stream processing** with sophisticated cross-attention
2. **Hierarchical frequency analysis** with adaptive filtering
3. **Multi-scale artifact fusion** combining SRM and wavelet features
4. **Dynamic feature selection** for optimal stream combination

## 🔧 Advanced Usage

### Custom Loss Functions
```python
from src.training import CombinedFinetuneLoss

# Custom loss weights
criterion = CombinedFinetuneLoss(seg_weight=0.8, class_weight=1.2)
```

### Model Customization
```python
# Create model with custom architecture
model = create_auranet(
    dims=[32, 64, 128, 256],  # Smaller model
    depths=[1, 1, 2, 1],      # Fewer blocks
    num_haft_levels=[2, 2, 1, 1]  # Simpler HAFT
)
```

### Transfer Learning
```python
# Load pre-trained weights for different dataset
model = create_auranet()
checkpoint = torch.load('pretrained_model.pth')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
```

## 📚 Technical Details

### SRM Filters
The model uses 10 fixed SRM (Spatial Rich Model) filters:
- Sobel horizontal and vertical
- Laplacian of Gaussian (5×5)
- Edge detection filters (3 variants)
- Square filters (3×3 and 5×5)
- Directional filters (D3,3 and D4,4)

### HAFT Architecture
The Hierarchical Adaptive Frequency Transform processes patches at multiple scales:
- **Level 0**: 1×1 patch (global context)
- **Level 1**: 2×2 patches
- **Level 2**: 4×4 patches
- **Level 3**: 8×8 patches

Each level contributes to adaptive filter learning through hierarchical context aggregation.

### Cross-Attention Mechanisms
- **Deformable Cross-Attention**: For stages 2-3 (high resolution)
  - Adaptive sampling based on learned offsets
  - Continuous positional bias
- **Standard Cross-Attention**: For stages 4-5 (low resolution)
  - Global attention mechanism
  - Reduced computational complexity

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Memory Error**
   - Reduce batch size in config
   - Use gradient checkpointing
   - Enable mixed precision training

2. **Slow Training**
   - Ensure CUDA is available
   - Use multiple GPUs with DataParallel
   - Optimize data loading with more workers

3. **Poor Performance**
   - Check data quality and labels
   - Verify pre-processing pipeline
   - Consider longer pre-training

### Dependencies
- Python >= 3.8
- PyTorch >= 1.12.0
- CUDA >= 11.0 (for GPU training)

## 📄 Citation

If you use AuraNet in your research, please cite:

```bibtex
@article{auranet2024,
  title={AuraNet: A Dual-Stream Forensic Network for Face Manipulation Detection},
  author={Your Name},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ConvNeXt V2 architecture inspiration
- SRM filters from steganalysis research  
- Wavelet transform implementations
- PyTorch and torchvision teams

---

For more detailed documentation, examples, and tutorials, please visit our [documentation site](https://your-username.github.io/AuraNet/).
