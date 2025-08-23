# CUDA 11.4 Installation Guide for AuraNet

This guide helps you set up AuraNet on systems with CUDA 11.4.

## Quick Start

### Option 1: Using Conda (Recommended)
```bash
# Create and activate environment
conda env create -f environment_cuda114.yml
conda activate auranet-cuda114

# Install PyTorch and remaining dependencies
python setup_cuda114.py

# Or manually:
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_cuda114.txt
```

### Option 2: Using pip with virtual environment
```bash
# Create virtual environment
python -m venv auranet_cuda114
# Windows:
auranet_cuda114\Scripts\activate
# Linux/Mac:
source auranet_cuda114/bin/activate

# Run installation script
python install_cuda114.py

# Or install manually:
pip install -r requirements_cuda114.txt
```

### Option 3: Manual PyTorch Installation
```bash
# Install PyTorch first
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Then install other requirements
pip install -r requirements_cuda114.txt
```

## Key Differences for CUDA 11.4

1. **PyTorch Version**: Uses PyTorch 2.0.1 with CUDA 11.8 binaries (compatible with CUDA 11.4)
2. **Version Pinning**: More restrictive version ranges for better compatibility
3. **Optional Packages**: Some packages like `xformers` and `apex` may need special handling

## Verification

After installation, run:
```bash
python count_params.py  # Should work without errors
python -c "import torch; print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'CUDA not available')"
```

## Troubleshooting

### Common Issues:

1. **CUDA Version Mismatch**:
   ```bash
   nvcc --version  # Check CUDA compiler version
   nvidia-smi      # Check driver version
   ```

2. **PyTorch CUDA Not Available**:
   - Ensure CUDA 11.4+ is installed
   - Check that `nvidia-smi` shows compatible driver version
   - Try reinstalling PyTorch: `pip install torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 --force-reinstall`

3. **Package Conflicts**:
   - Use fresh virtual environment
   - Install packages in the order specified in the installation script

4. **xformers Installation Fails**:
   ```bash
   # Skip xformers if it fails (it's optional)
   pip install -r requirements_cuda114.txt --ignore-installed xformers
   ```

5. **Memory Issues During Training**:
   - Reduce batch size in `config_celeb_df.yaml`
   - Enable gradient checkpointing
   - Use mixed precision training (already enabled by default)

## System Requirements

- **CUDA**: 11.4 or higher
- **Python**: 3.8 - 3.11 (3.11 recommended)
- **GPU Memory**: At least 8GB recommended for training
- **RAM**: At least 16GB recommended

## Next Steps

1. Update dataset path in `config_celeb_df.yaml`:
   ```yaml
   dataset:
     data_root: "path/to/your/celeb_df_dataset"  # Update this
   ```

2. Test the setup:
   ```bash
   python count_params.py
   python launch_training.py --help
   ```

3. Start training:
   ```bash
   python launch_training.py --mode pretrain
   ```
