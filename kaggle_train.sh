#!/bin/bash
# Kaggle Training Script for AuraNet

# Directory paths
WORKING_DIR="/kaggle/working/AuraNet"
DATA_ROOT="/kaggle/input"
CONFIG_FILE="/kaggle/working/AuraNet/config_celeb_df_memory_optimized.yaml"
PRETRAINED_PATH="/kaggle/input/convnextv2-pico/pytorch/default/1/convnextv2_pico_1k_224_fcmae.pt"

# Training parameters
MODE="pretrain"  # Options: pretrain, finetune, both
NUM_GPUS=2
USE_PRETRAINED="yes"

# Print configuration
echo "========================================"
echo "AuraNet Training on Kaggle"
echo "========================================"
echo "Working directory: $WORKING_DIR"
echo "Data root: $DATA_ROOT"
echo "Config file: $CONFIG_FILE"
echo "Mode: $MODE"
echo "GPUs: $NUM_GPUS"
echo "Use pretrained weights: $USE_PRETRAINED"
echo "Pretrained path: $PRETRAINED_PATH"
echo "========================================"

# Create working directory
mkdir -p $WORKING_DIR

# Copy configuration file
cp "/kaggle/input/auranet-config/config_celeb_df_memory_optimized.yaml" $CONFIG_FILE

# Run training
echo "Starting training..."
python train_celeb_df.py \
    --config $CONFIG_FILE \
    --mode $MODE \
    --data_root $DATA_ROOT \
    --gpus $NUM_GPUS \
    --use_pretrained $USE_PRETRAINED \
    --pretrained_path $PRETRAINED_PATH \
    --kaggle \
    --kaggle_working_dir $WORKING_DIR

# Check exit status
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed!"
fi
