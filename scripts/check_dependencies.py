#!/usr/bin/env python3
"""
Console script to check UniCeption dependencies.
"""

import sys
from pathlib import Path

# Add the parent directory to the path to import uniception
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_dependencies():
    """Check if optional dependencies are available."""
    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.version.cuda}")
        else:
            print("CUDA not available")
    except ImportError:
        print("PyTorch not installed")

    try:
        import xformers

        print(f"XFormers version: {xformers.__version__}")
    except ImportError:
        print("XFormers not installed")

    try:
        from uniception.models.libs.croco.curope import cuRoPE2D

        print("CroCo RoPE extension available")
    except ImportError:
        print("CroCo RoPE extension not available")


def main():
    """Main entry point for the check dependencies script."""
    print("Checking UniCeption Dependencies...")
    print("=" * 40)
    check_dependencies()


if __name__ == "__main__":
    main()
