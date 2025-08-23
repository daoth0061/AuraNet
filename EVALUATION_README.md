# AuraNet Comprehensive Evaluation System

## Overview

Hệ thống evaluation mới của AuraNet cung cấp đánh giá chi tiết cho 2 tasks chính:

1. **Classification Task**: Phân loại ảnh real/fake với các metrics: accuracy, AUC, AOC, EER, precision, recall, F1, specificity, sensitivity
2. **Mask Reconstruction Task**: Đánh giá chất lượng mask reconstruction với PSNR, SSIM và các metrics khác

## Key Features

### 1. TrainingEvaluator Class
- Tích hợp sẵn vào quá trình training
- Chạy evaluation chi tiết sau mỗi epoch
- Hỗ trợ cả pretrain và finetune modes
- Tự động load ground truth masks từ disk

### 2. Comprehensive Classification Metrics
- **Accuracy**: Độ chính xác tổng thể
- **AUC**: Area Under ROC Curve
- **AOC**: Area Over Curve (1 - AUC)
- **EER**: Equal Error Rate
- **Precision/Recall/F1**: Cho từng class và average
- **Specificity/Sensitivity**: Độ nhạy và độ đặc hiệu
- **Confusion Matrix**: Ma trận nhầm lẫn

### 3. Mask Reconstruction Metrics
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **Statistics**: Mean, Std, Min, Max cho mỗi metric

### 4. Logging System Integration
- Tất cả kết quả evaluation được log vào file 
- Log details với timestamp và module source
- Dễ dàng phân tích metrics sau khi training
- Export kết quả evaluation dưới dạng JSON

## Usage

### 1. Training với Evaluation tích hợp

```bash
python launch_training.py \
    --data_root celeb-df-dataset \
    --mask_gt_dir /kaggle/input/ff-mask/ \
    --mode both \
    --gpus 1 \
    --batch_size 8
```

### 2. Configuration trong YAML

```yaml
evaluation:
  batch_size: 16
  mask_gt_dir: "/kaggle/input/ff-mask/"  # Path to ground truth masks
  detailed_eval_freq: 1  # Run detailed evaluation every N epochs
  log_dir: "logs/evaluation"  # Directory to save evaluation logs
```

### 3. Test Evaluation System

```bash
# Test cơ bản
python test_evaluation.py

# Test với dataset thực
python test_evaluation.py dataset --data_root celeb-df-dataset --mask_gt_dir /path/to/masks
```

## File Structure

```
src/
├── training_evaluator.py     # Main evaluation class
├── celeb-df-dataset.py      # Updated dataset với image_path
└── ...

config_celeb_df.yaml         # Updated config với evaluation settings
launch_training.py           # Updated launcher với mask_gt_dir support
test_evaluation.py           # Test script cho evaluation system
logs/
├── auranet_20240821_163505.log  # Main log file
└── evaluation/                 # Evaluation log files
    ├── eval_20240821_163505.log
    └── eval_metrics_20240821_163505.json
```

## Evaluation Output Example

```
VALIDATION Metrics - Epoch 5:
==================================================
Classification Metrics:
  accuracy: 0.8750
  auc: 0.9234
  aoc: 0.0766
  eer: 0.1250
  precision_real: 0.8571
  precision_fake: 0.9000
  recall_real: 0.9000
  recall_fake: 0.8571
  f1_real: 0.8780
  f1_fake: 0.8780
  specificity: 0.9000
  sensitivity: 0.8571

Mask Reconstruction Metrics:
  psnr: 28.4521
  ssim: 0.8945
  psnr_std: 3.2145
  ssim_std: 0.0876
==================================================
```

## Key Benefits

1. **Automated Integration**: Không cần code thêm, evaluation tự động chạy
2. **Comprehensive Metrics**: Tất cả metrics quan trọng cho cả 2 tasks
3. **Flexible GT Loading**: Tự động tìm và load ground truth masks
4. **Memory Efficient**: Batch-wise evaluation để tiết kiệm memory
5. **Detailed Logging**: Logs chi tiết với formatting đẹp và lưu vào file
6. **Error Handling**: Robust error handling cho missing data

## Ground Truth Mask Directory Structure

```
/kaggle/input/ff-mask/
├── image1.png
├── image2.png
├── image1_mask.png
├── image2_mask.png
└── ...
```

Hệ thống sẽ tự động tìm masks với các patterns:
- `{image_name}.png`
- `{image_name}.jpg` 
- `{image_name}_mask.png`
- `{image_name}_mask.jpg`

## Integration với Training Pipeline

TrainingEvaluator được tích hợp sẵn vào `DistributedTrainer`:

1. Khởi tạo trong `__init__`
2. Chạy evaluation trong `validate_epoch`
3. Log metrics sau mỗi epoch với logging system
4. Lưu metrics vào file và tensorboard/wandb

## Performance Considerations

- Evaluation chỉ chạy trên rank 0 (distributed training)
- Batch-wise processing để tiết kiệm memory
- Lazy loading của ground truth masks
- Efficient tensor operations với PyTorch

## Logging Integration

Tất cả kết quả evaluation được ghi vào file log:

```python
from src.logging_utils import get_logger

class TrainingEvaluator:
    def __init__(self, config, device):
        self.logger = get_logger(__name__)
        self.config = config
        self.device = device
        
    def evaluate(self, model, dataloader):
        self.logger.info("Starting evaluation...")
        # Evaluation logic
        metrics = self._compute_metrics(predictions, targets)
        
        # Log detailed metrics
        self.logger.info(f"VALIDATION Metrics - Epoch {epoch}:")
        self.logger.info("="*50)
        
        for metric_name, value in metrics.items():
            self.logger.info(f"  {metric_name}: {value:.4f}")
            
        self.logger.info("="*50)
        
        # Export metrics to JSON
        self._export_metrics_to_json(metrics, epoch)
        
        return metrics
```

## Troubleshooting

1. **Missing GT masks**: Hệ thống sẽ skip mask evaluation nếu không tìm thấy GT
2. **Memory issues**: Giảm batch_size trong config
3. **Slow evaluation**: Giảm `detailed_eval_freq` trong config
4. **Missing logs**: Kiểm tra thư mục logs và quyền ghi file
