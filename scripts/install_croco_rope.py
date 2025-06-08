#!/usr/bin/env python3
"""
Console script to install CroCo RoPE extension.
"""

import os
import subprocess
import sys
from pathlib import Path


def install_croco_rope():
    """Install CroCo RoPE extension."""
    try:
        # Find the project root (where setup.py is located)
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        curope_path = project_root / "uniception" / "models" / "libs" / "croco" / "curope"

        if curope_path.exists():
            print("Installing CroCo RoPE extension...")
            original_cwd = os.getcwd()
            try:
                os.chdir(curope_path)
                subprocess.check_call([sys.executable, "setup.py", "build_ext", "--inplace"])
                print("CroCo RoPE extension installed successfully!")
                return True
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to install CroCo RoPE extension: {e}")
                print("You can install it later by running:")
                print(f"cd {curope_path} && python setup.py build_ext --inplace")
                return False
            finally:
                os.chdir(original_cwd)
        else:
            print("Warning: CroCo RoPE source code not found.")
            print(f"Expected location: {curope_path}")
            return False
    except Exception as e:
        print(f"Warning: Error during CroCo RoPE installation: {e}")
        return False


def main():
    """Main entry point for the CroCo RoPE installation script."""
    print("UniCeption CroCo RoPE Extension Installer")
    print("=" * 45)

    success = install_croco_rope()

    if success:
        print("\n✓ CroCo RoPE extension installation completed successfully!")
        sys.exit(0)
    else:
        print("\n⚠ CroCo RoPE extension installation failed or skipped.")
        print("This is typically due to missing CUDA development tools.")
        print("The extension is optional and UniCeption will work without it.")
        sys.exit(1)


if __name__ == "__main__":
    main()
