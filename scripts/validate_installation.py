"""
Validation script for UniCeption installation.

This script validates that all components of UniCeption are correctly installed
and provides helpful diagnostics.
"""

import importlib
import sys
from pathlib import Path


def check_package_installation():
    """Check if UniCeption package is properly installed."""
    try:
        import uniception

        print("✓ UniCeption package is installed")

        # Check if we can import core modules
        try:
            from uniception.models.encoders import UniCeptionViTEncoderBase

            print("✓ Core encoder modules are available")
        except ImportError as e:
            print(f"✗ Failed to import core encoder modules: {e}")

        return True
    except ImportError as e:
        print(f"✗ UniCeption package not found: {e}")
        return False


def check_dependencies():
    """Check optional dependencies."""
    dependencies = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "torchaudio": "TorchAudio",
        "xformers": "XFormers",
        "einops": "Einops",
        "matplotlib": "Matplotlib",
        "numpy": "NumPy",
        "PIL": "Pillow",
    }

    available = []
    missing = []

    for module, name in dependencies.items():
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "unknown")
            available.append(f"✓ {name}: {version}")
        except ImportError:
            missing.append(f"✗ {name}: not installed")

    print("\nDependency Status:")
    for dep in available:
        print(f"  {dep}")

    if missing:
        print("\nMissing Dependencies:")
        for dep in missing:
            print(f"  {dep}")

    return len(missing) == 0


def check_cuda_support():
    """Check CUDA support."""
    try:
        import torch

        if torch.cuda.is_available():
            print(f"\n✓ CUDA is available")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Available devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    Device {i}: {torch.cuda.get_device_name(i)}")
        else:
            print(f"\n⚠ CUDA is not available (CPU-only mode)")
        return True
    except ImportError:
        print(f"\n⚠ PyTorch not installed - cannot check CUDA support")
        return False


def check_croco_rope():
    """Check CroCo RoPE extension."""
    try:
        from uniception.models.libs.croco.curope import cuRoPE2D

        print("\n✓ CroCo RoPE extension is available")
        return True
    except ImportError:
        print("\n✗ CroCo RoPE extension not available")
        print("  To install: cd uniception/models/libs/croco/curope && python setup.py build_ext --inplace")
        return False


def check_model_availability():
    """Check if models can be loaded."""
    try:
        # Try to check if encoder modules are available
        from uniception.models import encoders

        print(f"\n✓ Encoder module is available")

        # Try to run the encoder list command
        try:
            import subprocess

            result = subprocess.run(
                [sys.executable, "-m", "uniception.models.encoders.list"], capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                encoder_count = len([line for line in lines if line.strip() and not line.startswith("Available")])
                print(f"✓ Available encoders: {encoder_count}")
                return True
            else:
                print(f"⚠ Encoder listing returned non-zero exit code: {result.returncode}")
                return False

        except subprocess.TimeoutExpired:
            print(f"⚠ Encoder listing timed out")
            return False
        except Exception as e:
            print(f"⚠ Could not run encoder listing: {e}")
            return False

    except Exception as e:
        print(f"\n✗ Failed to access encoder modules: {e}")
        return False


def check_file_structure():
    """Check if the project file structure is correct."""
    base_path = Path(__file__).parent.parent
    required_dirs = [
        "uniception",
        "uniception/models",
        "uniception/models/encoders",
        "uniception/models/info_sharing",
        "uniception/models/prediction_heads",
        "scripts",
    ]

    missing_dirs = []
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        if not full_path.exists():
            missing_dirs.append(dir_path)

    if missing_dirs:
        print(f"\n✗ Missing directories:")
        for dir_path in missing_dirs:
            print(f"  - {dir_path}")
        return False
    else:
        print(f"\n✓ Project structure is correct")
        return True


def main():
    """Run all validation checks."""
    print("UniCeption Installation Validation")
    print("=" * 40)

    checks = [
        ("Package Installation", check_package_installation),
        ("Dependencies", check_dependencies),
        ("CUDA Support", check_cuda_support),
        ("CroCo RoPE Extension", check_croco_rope),
        ("Model Availability", check_model_availability),
        ("File Structure", check_file_structure),
    ]

    results = []
    for name, check_func in checks:
        print(f"\nChecking {name}...")
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Error during {name} check: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 40)
    print("Validation Summary:")
    passed = 0
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{len(results)} checks passed")

    if passed == len(results):
        print("🎉 All checks passed! UniCeption is ready to use.")
        return 0
    else:
        print("⚠ Some checks failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
