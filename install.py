#!/usr/bin/env python3
"""
Installation script for AuraNet project
Handles proper package installation and environment setup
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command with error handling."""
    print(f"\n{description}...")
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed")
        print(f"Error: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error Output: {e.stderr}")
        return False

def main():
    """Main installation process."""
    print("=== AuraNet Installation Script ===")
    
    # Check if we're in the right directory
    if not Path("setup.py").exists():
        print("Error: Please run this script from the AuraNet root directory")
        sys.exit(1)
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        print("Failed to install requirements. Please check the error messages above.")
        return False
    
    # Install the package in development mode
    if not run_command("pip install -e .", "Installing AuraNet package"):
        print("Failed to install AuraNet package. Please check the error messages above.")
        return False
    
    print("\n=== Installation Complete ===")
    print("AuraNet has been installed successfully!")
    print("\nYou can now run:")
    print("  python train_celeb_df.py --help")
    print("  python launch_training.py --help")
    print("  python analyze_celeb_df.py --help")
    
    return True

if __name__ == "__main__":
    main()
