#!/usr/bin/env python3
"""
MeshAI CLI Entry Point

This script provides the main entry point for the MeshAI CLI tool.
It can be installed as a console script or run directly.
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path to import meshai modules
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

try:
    from meshai.cli.main import main
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"Error importing MeshAI CLI: {e}")
    print("Please ensure MeshAI SDK is properly installed:")
    print("  pip install -e .")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(1)