"""Package installation setup."""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install


def install_croco_rope():
    """Install CroCo RoPE extension."""
    try:
        curope_path = Path(__file__).parent / "uniception" / "models" / "libs" / "croco" / "curope"
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
            return False
    except Exception as e:
        print(f"Warning: Error during CroCo RoPE installation: {e}")
        return False


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


class CustomDevelopCommand(develop):
    """Custom development installation command."""

    def run(self):
        develop.run(self)
        # Only install CroCo RoPE if explicitly requested
        if os.getenv("INSTALL_CROCO_ROPE", "false").lower() in ("true", "1", "yes"):
            install_croco_rope()


class CustomInstallCommand(install):
    """Custom installation command."""

    def run(self):
        install.run(self)
        # Only install CroCo RoPE if explicitly requested
        if os.getenv("INSTALL_CROCO_ROPE", "false").lower() in ("true", "1", "yes"):
            install_croco_rope()


class CrocoInstallCommand(install):
    """Install command that includes CroCo RoPE extension."""

    def run(self):
        install.run(self)
        install_croco_rope()


class CheckDependenciesCommand(install):
    """Command to check available dependencies."""

    def run(self):
        check_dependencies()


# Core dependencies (including PyTorch which is essential for UniCeption)
install_requires = [
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

# Optional dependencies
extras_require = {
    "xformers": [
        "xformers",  # Will be installed from PyTorch wheel index
    ],
    "dev": [
        "black",
        "isort",
        "pre-commit",
        "pytest",
    ],
    "minimal": [
        # Minimal dependencies for basic functionality without PyTorch
        "numpy",
        "matplotlib",
        "Pillow",
        "scikit-learn",
        "einops",
    ],
}

# All optional dependencies combined (excluding minimal since it's subset of install_requires)
extras_require["all"] = list(set(extras_require["xformers"] + extras_require["dev"]))

setup(
    name="uniception",
    version="0.1.0",
    description="Generalizable Perception Stack for 3D, 4D, spatial AI and scene understanding",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="AirLab",
    license="BSD Clause-3",
    packages=find_packages(),
    package_dir={"": "."},
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require=extras_require,
    cmdclass={
        "develop": CustomDevelopCommand,
        "install": CustomInstallCommand,
        "install_croco": CrocoInstallCommand,
        "check_deps": CheckDependenciesCommand,
    },
    entry_points={
        "console_scripts": [
            "uniception-download-checkpoints=scripts.download_checkpoints:main",
            "uniception-validate=scripts.validate_installation:main",
            "uniception-prepare-offline=scripts.prepare_offline_install:main",
            "uniception-check-deps=scripts.check_dependencies:main",
            "uniception-install-croco=scripts.install_croco_rope:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="computer-vision, 3d-vision, spatial-ai, perception, deep-learning, pytorch",
)
