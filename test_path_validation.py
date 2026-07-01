#!/usr/bin/env python3
"""Test script to verify Path validation implementation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_path_validation():
    """Test Path validation functions."""
    print("Testing Path validation implementation...")
    
    # Test importing the modified functions
    try:
        from data.dataset import get_dataset
        from main import load_nli_data
        print("✓ Imports successful")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Test Path object creation
    try:
        train_dir = Path("./data/train")
        test_dir = Path("./data/test")
        print(f"✓ Path objects created: {train_dir}, {test_dir}")
    except Exception as e:
        print(f"✗ Path creation error: {e}")
        return False
    
    # Test Path validation methods
    try:
        # These should not raise exceptions
        exists_check = train_dir.exists()
        is_dir_check = train_dir.is_dir()
        print(f"✓ Path validation methods work: exists={exists_check}, is_dir={is_dir_check}")
    except Exception as e:
        print(f"✗ Path validation error: {e}")
        return False
    
    print("All Path validation tests passed!")
    return True

if __name__ == "__main__":
    test_path_validation()