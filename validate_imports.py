#!/usr/bin/env python3
"""
Validation script to check that all modules can be imported correctly.
"""

import sys
import traceback

def validate_imports():
    """Validate that all modules can be imported without errors."""
    modules_to_test = [
        "src.models.config",
        "src.utils.memory",
        "src.data.dataset",
        "src.main"
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            failed_imports.append((module, str(e)))
            print(f"✗ {module} - {str(e)}")
    
    if failed_imports:
        print("\nFailed imports:")
        for module, error in failed_imports:
            print(f"  {module}: {error}")
        return False
    else:
        print("\nAll modules imported successfully!")
        return True

if __name__ == "__main__":
    success = validate_imports()
    sys.exit(0 if success else 1)