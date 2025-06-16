# UniCeption

UniCeption houses modular building blocks for developing and training generalizable perception models for all things related to 3D, 4D, spatial AI and scene understanding.
It is designed to be flexible and extensible, allowing researchers to easily experiment with different architectures and configurations.

Please refer to the [Developer Guidelines](#developer-guidelines) for contributing to the project.

## Installation

Clone the repository to your local machine by running the following command:

```bash
git clone git@github.com:castacks/UniCeption.git
cd UniCeption
```

### Standard Installation

Install the `uniception` package in development mode by running the following commands:

```bash
# Please use Conda or Python Virtual Environment based on your preference
# For Conda Environment
conda create --name uniception python=3.12
conda activate uniception
# For Python Virtual Environment
virtualenv uniception
source uniception/bin/activate

# Install UniCeption with base dependencies (includes PyTorch)
pip install -e .

# Optional: Install with XFormers support
pip install -e ".[xformers]"

# Optional: Install with development tools
pip install -e ".[dev]"

# Optional: Install all optional dependencies
pip install -e ".[all]"

# Setup pre-commit hooks for development
pre-commit install
```

### Optional: CroCo RoPE Extension Installation

To use CroCo models with the custom RoPE kernel:

```bash
# Recommended: Use the console script
uniception-install-croco

# Alternative: Set environment variable during installation
INSTALL_CROCO_ROPE=true pip install -e .

# Manual compilation (if needed)
cd uniception/models/libs/croco/curope
python setup.py build_ext --inplace
cd ../../../../../
```

### Installation Validation and Dependency Checking

After installation, use these console scripts to validate your setup:

```bash
# Validate installation and check dependencies
uniception-validate

# Check which optional dependencies are available
uniception-check-deps
```

### Advanced Installation Options

#### Docker Installation (No Internet Access)

If you're working in a Docker container that already has Python dependencies installed but no internet access, you can install UniCeption in development mode without triggering network requests:

```bash
# Install only the package structure without dependencies
pip install --no-deps -e ".[all]"

# Or use the following if the above command still hangs or triggers network requests
pip install --no-index --no-use-pep517 -e ".[all]"
```

**Note:** This command assumes your Docker image already contains all required dependencies (PyTorch, etc.). Use `uniception-validate` after installation to verify all dependencies are available.

#### Offline Installation

For environments without internet access:

```bash
# 1. On a machine with internet access, prepare offline wheels
uniception-prepare-offline --output-dir offline_wheels --extras all

# 2. Copy the offline_wheels directory to your offline environment
# 3. Run the offline installation
cd offline_wheels
INSTALL_CROCO_ROPE=true INSTALL_XFORMERS=true ./install_offline.sh
```

#### Downloading Checkpoints

Download UniCeption format custom checkpoints:

```bash
# Download all available checkpoints
uniception-download-checkpoints

# Download specific folders only (e.g., encoders and prediction heads)
uniception-download-checkpoints --folders encoders prediction_heads

# Specify custom destination
uniception-download-checkpoints --destination /path/to/checkpoints
```

**Available options:**
- `--folders`: Specify which folders to download. Choices: `encoders`, `info_sharing`, `prediction_heads`, `examples` (default: all folders)
- `--destination`: Custom destination folder for downloaded checkpoints (default: current directory)

---

## Currently Supported Components

### Encoders

Please refer to the `uniception/models/encoders` directory for the supported encoders and documentation for adding new encoders. The supported encoders can be listed by running:

```bash
python3 -m uniception.models.encoders.list
```

---

## Information Sharing Blocks

Please refer to the `uniception/models/info_sharing` directory for the supported information sharing blocks.

---

## Prediction Heads

Please refer to the `uniception/models/prediction_heads` directory for the supported prediction heads.

---

## Codebases built on top of UniCeption

Check out our following codebases which build on top of UniCeption:
- [UFM: A Simple Path towards Unified Dense Correspondence with Flow](https://uniflowmatch.github.io/)

## License

The code in this repository is licensed under a fully open-source [BSD 3-Clause License](LICENSE).

## Acknowledgements

We thank the following projects for their open-source code: [DUSt3R](https://github.com/naver/dust3r), [MASt3R](https://github.com/naver/mast3r), [MoGe](https://github.com/microsoft/moge), [HF PyTorch Image Models](https://github.com/huggingface/pytorch-image-models), and all the other pre-trained image encoders featured in this repo.

## Contributing & Developer Guidelines

If you find our work useful, please consider giving it a star ⭐. We welcome contributions to UniCeption! Whether it's fixing bugs, adding new features, or improving documentation, your help is appreciated.

Please follow these guidelines when contributing to UniCeption:
- **Code Style**: Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for code style.
- **Documentation**: Add docstrings to all classes and methods.
- **Unit Tests**: Add necessary unit tests to the `tests` folder.
- **Linting**: Run `black` & `isort` on your code before committing. For example, you can run `black . && isort .`.

Please create a pull request for any changes you make, and ensure that all tests pass before merging. We also encourage you to open issues for discussion before starting on larger features or changes. Also please feel free to add further unit tests to the `tests` folder to ensure the correctness of your changes.

## Maintainers

UniCeption is maintained by the [AirLab](https://theairlab.org/). In particular, feel free to reach out to the following maintainers for any questions or issues (Github issues are preferred):
- [Nikhil Keetha](https://nik-v9.github.io/)
- [Yuchen Zhang](https://infinity1096.github.io/)
