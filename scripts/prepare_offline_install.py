#!/usr/bin/env python3
"""
Script to prepare dependencies for offline installation.

This script downloads all necessary wheel files for offline installation
of UniCeption in environments without internet access.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def download_wheels(output_dir: Path, extras: list = None):
    """Download wheel files for offline installation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create temporary requirements files
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)

    try:
        # Create requirements files
        create_requirements_files(temp_dir, extras)

        # Download base dependencies
        base_cmd = [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--dest",
            str(output_dir),
            "-r",
            str(temp_dir / "requirements-base.txt"),
        ]

        print(f"Downloading base dependencies to {output_dir}...")
        subprocess.check_call(base_cmd)

        # Download optional dependencies if requested
        if extras:
            for extra in extras:
                if extra == "all":
                    # Download all extras
                    for req_file in ["requirements-xformers.txt", "requirements-dev.txt"]:
                        if (temp_dir / req_file).exists():
                            cmd = [
                                sys.executable,
                                "-m",
                                "pip",
                                "download",
                                "--dest",
                                str(output_dir),
                                "-r",
                                str(temp_dir / req_file),
                            ]
                            print(
                                f"Downloading {req_file.replace('requirements-', '').replace('.txt', '')} dependencies..."
                            )
                            try:
                                subprocess.check_call(cmd)
                            except subprocess.CalledProcessError as e:
                                print(f"Warning: Failed to download {extra} dependencies: {e}")
                else:
                    req_file = temp_dir / f"requirements-{extra}.txt"
                    if req_file.exists():
                        cmd = [sys.executable, "-m", "pip", "download", "--dest", str(output_dir), "-r", str(req_file)]
                        print(f"Downloading {extra} dependencies...")
                        try:
                            subprocess.check_call(cmd)
                        except subprocess.CalledProcessError as e:
                            print(f"Warning: Failed to download {extra} dependencies: {e}")

        # Create final offline installation files
        create_offline_installation_files(output_dir)

        print("Download completed successfully!")

    except subprocess.CalledProcessError as e:
        print(f"Error downloading wheels: {e}")
        sys.exit(1)
    finally:
        # Clean up temporary files
        import shutil

        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def create_requirements_files(temp_dir: Path, extras: list = None):
    """Create temporary requirements files for downloading."""

    # Base requirements (including PyTorch)
    base_reqs = [
        "numpy",
        "torch",
        "torchvision",
        "torchaudio",
        "timm",
        "black",
        "jaxtyping",
        "matplotlib",
        "Pillow",
        "scikit-learn",
        "einops",
        "rerun-sdk",
        "pre-commit",
        "minio",
        "pytest",
        "isort",
    ]

    # Write base requirements
    with open(temp_dir / "requirements-base.txt", "w") as f:
        for req in base_reqs:
            f.write(f"{req}\n")

    # XFormers requirements
    with open(temp_dir / "requirements-xformers.txt", "w") as f:
        f.write("xformers\n")

    # Dev requirements
    dev_reqs = [
        "black",
        "isort",
        "pre-commit",
        "pytest",
    ]

    with open(temp_dir / "requirements-dev.txt", "w") as f:
        for req in dev_reqs:
            f.write(f"{req}\n")


def create_offline_installation_files(output_dir: Path):
    """Create requirements files and installation script for offline use."""

    # Base requirements (including PyTorch)
    base_reqs = [
        "numpy",
        "torch",
        "torchvision",
        "torchaudio",
        "timm",
        "black",
        "jaxtyping",
        "matplotlib",
        "Pillow",
        "scikit-learn",
        "einops",
        "rerun-sdk",
        "pre-commit",
        "minio",
        "pytest",
        "isort",
    ]

    # Write base requirements
    with open(output_dir / "requirements-base.txt", "w") as f:
        for req in base_reqs:
            f.write(f"{req}\n")

    # XFormers requirements
    with open(output_dir / "requirements-xformers.txt", "w") as f:
        f.write("xformers\n")

    # Dev requirements
    dev_reqs = [
        "black",
        "isort",
        "pre-commit",
        "pytest",
    ]

    with open(output_dir / "requirements-dev.txt", "w") as f:
        for req in dev_reqs:
            f.write(f"{req}\n")

    # Create installation script
    install_script = output_dir / "install_offline.sh"
    with open(install_script, "w") as f:
        f.write(
            """#!/bin/bash
# Offline installation script for UniCeption

set -e

echo "Installing UniCeption dependencies offline..."

# Check if we're in the right directory
if [ ! -f "requirements-base.txt" ]; then
    echo "Error: requirements-base.txt not found. Please run this script from the offline_wheels directory."
    exit 1
fi

# Install base dependencies (includes PyTorch)
echo "Installing base dependencies (including PyTorch)..."
pip install --no-index --find-links . -r requirements-base.txt

# Install XFormers if requested
if [ "$INSTALL_XFORMERS" = "true" ]; then
    echo "Installing XFormers..."
    pip install --no-index --find-links . -r requirements-xformers.txt
fi

# Install dev dependencies if requested
if [ "$INSTALL_DEV" = "true" ]; then
    echo "Installing development dependencies..."
    pip install --no-index --find-links . -r requirements-dev.txt
fi

# Navigate back to UniCeption directory and install the package
echo "Installing UniCeption package..."
cd ..
pip install --no-deps -e .

# Install CroCo RoPE extension if requested
if [ "$INSTALL_CROCO_ROPE" = "true" ]; then
    echo "Installing CroCo RoPE extension..."
    cd uniception/models/libs/croco/curope
    python setup.py build_ext --inplace
    cd -
fi

echo "Offline installation completed successfully!"
echo ""
echo "To verify installation, run:"
echo "python setup.py check_deps"
"""
        )

    # Make script executable
    install_script.chmod(0o755)

    # Create Windows batch script as well
    install_bat = output_dir / "install_offline.bat"
    with open(install_bat, "w") as f:
        f.write(
            """@echo off
REM Offline installation script for UniCeption (Windows)

echo Installing UniCeption dependencies offline...

REM Check if we're in the right directory
if not exist "requirements-base.txt" (
    echo Error: requirements-base.txt not found. Please run this script from the offline_wheels directory.
    exit /b 1
)

REM Install base dependencies (includes PyTorch)
echo Installing base dependencies (including PyTorch)...
pip install --no-index --find-links . -r requirements-base.txt

REM Install XFormers if requested
if "%INSTALL_XFORMERS%"=="true" (
    echo Installing XFormers...
    pip install --no-index --find-links . -r requirements-xformers.txt
)

REM Install dev dependencies if requested
if "%INSTALL_DEV%"=="true" (
    echo Installing development dependencies...
    pip install --no-index --find-links . -r requirements-dev.txt
)

REM Navigate back to UniCeption directory and install the package
echo Installing UniCeption package...
cd ..
pip install --no-deps -e .

REM Install CroCo RoPE extension if requested
if "%INSTALL_CROCO_ROPE%"=="true" (
    echo Installing CroCo RoPE extension...
    cd uniception\\models\\libs\\croco\\curope
    python setup.py build_ext --inplace
    cd ..\\..\\..\\..\\..
)

echo Offline installation completed successfully!
echo.
echo To verify installation, run:
echo python setup.py check_deps
"""
        )

    # Create a README for offline installation
    readme_file = output_dir / "README_OFFLINE.md"
    with open(readme_file, "w") as f:
        f.write(
            """# UniCeption Offline Installation

This directory contains all the necessary files for installing UniCeption without internet access.

## Files Included

- `requirements-base.txt` - Core dependencies (including PyTorch)
- `requirements-xformers.txt` - XFormers dependency
- `requirements-dev.txt` - Development dependencies
- `install_offline.sh` - Installation script for Unix/Linux/macOS
- `install_offline.bat` - Installation script for Windows
- `*.whl` files - Downloaded wheel packages

## Installation Instructions

### Unix/Linux/macOS

```bash
# Set environment variables for optional components
export INSTALL_XFORMERS=true        # Install XFormers
export INSTALL_DEV=true             # Install development tools
export INSTALL_CROCO_ROPE=true      # Compile CroCo RoPE extension

# Run the installation script
./install_offline.sh
```

### Windows

```cmd
REM Set environment variables for optional components
set INSTALL_XFORMERS=true
set INSTALL_DEV=true
set INSTALL_CROCO_ROPE=true

REM Run the installation script
install_offline.bat
```

## Manual Installation

If the scripts don't work, you can install manually:

```bash
# Install base dependencies (includes PyTorch)
pip install --no-index --find-links . -r requirements-base.txt

# Install optional dependencies as needed
pip install --no-index --find-links . -r requirements-xformers.txt
pip install --no-index --find-links . -r requirements-dev.txt

# Install UniCeption package (from parent directory)
cd ..
pip install --no-deps -e .

# Compile CroCo RoPE extension (optional)
cd uniception/models/libs/croco/curope
python setup.py build_ext --inplace
```

## Verification

After installation, verify everything is working:

```bash
cd ..  # Go back to UniCeption root directory
python setup.py check_deps
```

## Notes

- PyTorch, TorchVision, and TorchAudio are now included in the base requirements
- XFormers is optional and only needed for certain performance optimizations
- CroCo RoPE extension compilation requires a CUDA-enabled environment
"""
        )

    print(f"Created offline installation files in {output_dir}")
    print("Files created:")
    print("  - requirements-base.txt (includes PyTorch)")
    print("  - requirements-xformers.txt")
    print("  - requirements-dev.txt")
    print("  - install_offline.sh (Unix/Linux/macOS)")
    print("  - install_offline.bat (Windows)")
    print("  - README_OFFLINE.md")


def create_offline_requirements(output_dir: Path):
    """Create requirements files for offline installation."""
    # This function is now replaced by create_offline_installation_files
    pass


def main():
    parser = argparse.ArgumentParser(description="Prepare UniCeption for offline installation")
    parser.add_argument(
        "--output-dir", type=Path, default="offline_wheels", help="Directory to store downloaded wheels"
    )
    parser.add_argument("--extras", nargs="+", choices=["xformers", "dev", "all"], help="Extra dependencies to include")

    args = parser.parse_args()

    download_wheels(args.output_dir, args.extras)


if __name__ == "__main__":
    main()
