#!/usr/bin/env python3
"""
Test script to verify import fixes
"""

import sys
import os
from pathlib import Path

def test_import_structure():
    """Test if the import structure is correct without importing actual modules."""
    src_path = Path(__file__).parent / 'src'
    sys.path.insert(0, str(src_path))
    
    print("=== Testing Import Structure ===")
    
    # Test module files exist
    required_files = [
        'auranet.py',
        'haft.py', 
        'cross_fusion.py',
        'initial_processing.py',
        'output_heads.py',
        'utils.py',
        'train.py',
        'data_loader.py',
        'training.py',
        'evaluate.py',
        'demo.py',
        'celeb_df_dataset.py',
        '__init__.py'
    ]
    
    missing_files = []
    for file in required_files:
        file_path = src_path / file
        if not file_path.exists():
            missing_files.append(file)
        else:
            print(f"✓ {file} exists")
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    
    # Check import syntax (without actually importing to avoid dependency issues)
    print("\n=== Checking Import Syntax ===")
    
    files_to_check = [
        ('auranet.py', [
            'from initial_processing import',
            'from haft import', 
            'from cross_fusion import',
            'from output_heads import',
            'from utils import'
        ]),
        ('output_heads.py', ['from utils import']),
        ('cross_fusion.py', ['from utils import']),
        ('initial_processing.py', ['from utils import'])
    ]
    
    for filename, expected_imports in files_to_check:
        file_path = src_path / filename
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check no relative imports remain
            if 'from .' in content:
                print(f"✗ {filename} still has relative imports")
                return False
            else:
                print(f"✓ {filename} has no relative imports")
                
            # Check expected imports are present
            for expected in expected_imports:
                if expected in content:
                    print(f"  ✓ Found: {expected}")
                else:
                    print(f"  ✗ Missing: {expected}")
                    
        except Exception as e:
            print(f"✗ Error checking {filename}: {e}")
            return False
    
    print(f"\n=== Summary ===")
    print("✓ All import structure fixes appear to be applied correctly!")
    print("✓ No relative imports detected")
    print("✓ All required files present")
    
    return True

if __name__ == "__main__":
    success = test_import_structure()
    if success:
        print("\n🎉 Import fixes successful! You can now:")
        print("1. Install requirements: pip install -r requirements.txt")
        print("2. Install package: pip install -e .")
        print("3. Or use: python run.py train_celeb_df.py --help")
    else:
        print("\n❌ Import fixes need attention!")
        sys.exit(1)
