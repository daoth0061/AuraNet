#!/usr/bin/env python3
"""
Post-installation script for AuraNet CUDA 11.4 environment
Run this after creating the conda environment
"""

import subprocess
import sys
import os

def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"\n{'='*50}")
    print(f"{description}")
    print(f"{'='*50}")
    print(f"Running: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print("Output:", result.stdout)
    if result.stderr:
        print("Error:", result.stderr)
    
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        return False
    else:
        print(f"✅ Success: {description}")
        return True

def main():
    print("AuraNet CUDA 11.4 Post-Installation Script")
    print("="*50)
    
    # Check if we're in the right conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'Unknown')
    print(f"Current conda environment: {conda_env}")
    
    if 'auranet-cuda114' not in conda_env.lower():
        print("⚠️  Warning: You should run this in the auranet-cuda114 environment")
        print("Run: conda activate auranet-cuda114")
        response = input("Continue anyway? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            sys.exit(0)
    
    # Step 1: Install PyTorch with CUDA 11.4 support
    pytorch_cmd = (
        f"{sys.executable} -m pip install "
        "torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 "
        "--extra-index-url https://download.pytorch.org/whl/cu118"
    )
    
    if not run_command(pytorch_cmd, "Installing PyTorch with CUDA 11.4 support"):
        print("Failed to install PyTorch. Please check your internet connection and try again.")
        sys.exit(1)
    
    # Step 2: Install remaining requirements
    requirements_cmd = f"{sys.executable} -m pip install -r requirements_cuda114.txt"
    
    if not run_command(requirements_cmd, "Installing remaining requirements"):
        print("Some packages failed to install. This might be okay for optional packages.")
    
    # Step 3: Verify installation
    print("\n" + "="*50)
    print("Verifying Installation")
    print("="*50)
    
    # Test PyTorch
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"✅ CUDA version: {torch.version.cuda}")
        else:
            print("❌ CUDA not available in PyTorch")
    except ImportError:
        print("❌ Failed to import PyTorch")
        sys.exit(1)
    
    # Test other key imports
    test_imports = [
        ("torchvision", "TorchVision"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("yaml", "PyYAML"),
        ("tqdm", "tqdm"),
    ]
    
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✅ {name} imported successfully")
        except ImportError:
            print(f"❌ Failed to import {name}")
    
    print("\n" + "="*50)
    print("Installation Complete!")
    print("="*50)
    print("\nNext steps:")
    print("1. Update dataset path in config_celeb_df.yaml")
    print("2. Test the model: python count_params.py")
    print("3. Start training: python launch_training.py --mode pretrain")
    
if __name__ == "__main__":
    main()
