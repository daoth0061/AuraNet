#!/usr/bin/env python3
"""
AuraNet Installation Script for CUDA 11.4
Handles the installation of all dependencies optimized for CUDA 11.4 systems.
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def run_command(cmd, check=True, shell=False):
    """Run a command and handle errors gracefully."""
    print(f"Running: {cmd}")
    if isinstance(cmd, str) and not shell:
        cmd = cmd.split()
    
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True, shell=shell)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def check_cuda_version():
    """Check CUDA version."""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            output = result.stdout
            print("CUDA Compiler version:")
            print(output)
            return True
        else:
            print("CUDA not found or nvcc not in PATH")
            return False
    except FileNotFoundError:
        print("CUDA not found or nvcc not in PATH")
        return False

def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"Python version: {version.major}.{version.minor}.{version.micro} ✓")
        return True
    else:
        print(f"Python version: {version.major}.{version.minor}.{version.micro} ✗")
        print("Python 3.8+ is required")
        return False

def install_pytorch_cuda114():
    """Install PyTorch with CUDA 11.4 support."""
    print("\n" + "="*50)
    print("Installing PyTorch for CUDA 11.4...")
    print("="*50)
    
    # Install PyTorch with CUDA 11.8 (compatible with CUDA 11.4)
    pytorch_cmd = [
        sys.executable, "-m", "pip", "install", 
        "torch==2.0.1+cu118", 
        "torchvision==0.15.2+cu118", 
        "torchaudio==2.0.2+cu118",
        "--extra-index-url", "https://download.pytorch.org/whl/cu118"
    ]
    
    success = run_command(pytorch_cmd)
    if not success:
        print("Failed to install PyTorch. Trying alternative approach...")
        # Try with pip install directly
        alt_cmd = f"{sys.executable} -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118"
        return run_command(alt_cmd, shell=True)
    
    return success

def install_requirements():
    """Install other requirements."""
    print("\n" + "="*50)
    print("Installing other dependencies...")
    print("="*50)
    
    requirements_file = "requirements_cuda114.txt"
    if not os.path.exists(requirements_file):
        print(f"Requirements file {requirements_file} not found!")
        return False
    
    # Create a temporary requirements file without the PyTorch lines
    temp_req = "temp_requirements.txt"
    with open(requirements_file, 'r') as f:
        lines = f.readlines()
    
    # Filter out PyTorch-related lines and --extra-index-url lines
    filtered_lines = []
    skip_next = False
    for line in lines:
        if skip_next:
            skip_next = False
            continue
        if any(pkg in line.lower() for pkg in ['torch==', 'torchvision==', 'torchaudio==', '--extra-index-url']):
            continue
        if line.strip().endswith('--extra-index-url https://download.pytorch.org/whl/cu118'):
            skip_next = True
            continue
        filtered_lines.append(line)
    
    with open(temp_req, 'w') as f:
        f.writelines(filtered_lines)
    
    # Install remaining requirements
    cmd = [sys.executable, "-m", "pip", "install", "-r", temp_req]
    success = run_command(cmd)
    
    # Clean up temporary file
    if os.path.exists(temp_req):
        os.remove(temp_req)
    
    return success

def install_optional_packages():
    """Install optional packages that might fail."""
    print("\n" + "="*50)
    print("Installing optional packages...")
    print("="*50)
    
    optional_packages = [
        "xformers==0.0.16",  # For CUDA 11.4
        "cupy-cuda114",      # CuPy for CUDA 11.4
        "lightly",           # May fail, but not critical
    ]
    
    for package in optional_packages:
        print(f"\nTrying to install {package}...")
        cmd = [sys.executable, "-m", "pip", "install", package]
        success = run_command(cmd, check=False)
        if success:
            print(f"✓ {package} installed successfully")
        else:
            print(f"⚠ {package} installation failed (this is optional)")

def verify_installation():
    """Verify that key packages are installed correctly."""
    print("\n" + "="*50)
    print("Verifying installation...")
    print("="*50)
    
    # Test imports
    test_imports = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("yaml", "PyYAML"),
        ("tqdm", "tqdm"),
        ("sklearn", "scikit-learn"),
        ("PIL", "Pillow"),
    ]
    
    failed_imports = []
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name}")
            failed_imports.append(name)
    
    # Test CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ PyTorch version: {torch.__version__}")
        else:
            print("⚠ CUDA not available in PyTorch")
    except ImportError:
        print("✗ Could not import PyTorch")
        failed_imports.append("PyTorch")
    
    return len(failed_imports) == 0

def main():
    """Main installation routine."""
    print("AuraNet Installation Script for CUDA 11.4")
    print("="*50)
    
    # Check system requirements
    if not check_python_version():
        sys.exit(1)
    
    check_cuda_version()
    
    # Create virtual environment recommendation
    print("\n" + "="*50)
    print("RECOMMENDATION: Use a conda environment")
    print("="*50)
    print("conda create -n auranet python=3.11")
    print("conda activate auranet")
    print("Then run this script again.")
    print("")
    
    response = input("Continue with installation? (y/N): ").lower()
    if response not in ['y', 'yes']:
        print("Installation cancelled.")
        sys.exit(0)
    
    # Upgrade pip first
    print("\n" + "="*50)
    print("Upgrading pip...")
    print("="*50)
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    
    # Install PyTorch
    if not install_pytorch_cuda114():
        print("Failed to install PyTorch. Please check your CUDA installation.")
        sys.exit(1)
    
    # Install other requirements
    if not install_requirements():
        print("Some packages failed to install. Check the output above.")
        # Don't exit, continue with verification
    
    # Install optional packages
    install_optional_packages()
    
    # Verify installation
    if verify_installation():
        print("\n" + "="*50)
        print("✓ Installation completed successfully!")
        print("="*50)
        print("\nNext steps:")
        print("1. Update the dataset path in config_celeb_df.yaml")
        print("2. Run: python count_params.py  # to check model parameters")
        print("3. Run: python launch_training.py --help  # to see training options")
    else:
        print("\n" + "="*50)
        print("⚠ Installation completed with some issues.")
        print("="*50)
        print("Please check the failed imports above and install them manually.")

if __name__ == "__main__":
    main()
