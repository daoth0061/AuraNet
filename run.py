#!/usr/bin/env python3
"""
Run script for AuraNet training with proper path setup
Alternative to installing the package
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python run.py <script_name> [args...]")
        print("Examples:")
        print("  python run.py train_celeb_df.py --help")
        print("  python run.py analyze_celeb_df.py --data_root /path/to/data")
        print("  python run.py launch_training.py --config config_celeb_df.yaml")
        sys.exit(1)
    
    script_name = sys.argv[1]
    script_args = sys.argv[2:]
    
    # Check if script exists
    script_path = project_root / script_name
    if not script_path.exists():
        print(f"Error: Script '{script_name}' not found in {project_root}")
        sys.exit(1)
    
    # Update sys.argv for the target script
    sys.argv = [str(script_path)] + script_args
    
    # Execute the script
    try:
        with open(script_path, 'r') as f:
            code = compile(f.read(), str(script_path), 'exec')
            exec(code, {'__file__': str(script_path), '__name__': '__main__'})
    except Exception as e:
        print(f"Error executing {script_name}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
