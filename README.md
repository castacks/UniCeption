# UniCeption

A Generalizable Perception Stack for Scene Understanding. 

Please refer to the [Developer Guidelines](#developer-guidelines) for contributing to the project.

## Installation

Clone the repository to your local machine by running the following command:

```bash
git clone git@github.com:castacks/UniCeption.git
```

Install the `uniception` package in development mode by running the following commands:

```bash
# Please use Conda or Python Virtual Environment based on your preference
# For Conda Environment
conda create --name uniception python=3.10
conda activate uniception
# For Python Virtual Environment
virtualenv uniception
source uniception/bin/activate
# Install PyTorch and other dependencies
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
pip install -U xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pre-commit install # Github pre-commit hooks for linting
cd uniception/models/libs/croco/curope # For CroCo
python3 setup.py build_ext --inplace
cd ../../../../../
pip install -e .
```

### Downloading UniCeption Format Checkpoints

To download the UniCeption format custom checkpoints, run the following command:

```bash
python3 scripts/download_checkpoints.py
```

For options to download specific folders please refer to the script's help message:

```
usage: download_checkpoints.py [-h] [--folders FOLDERS [FOLDERS ...]] [--destination DESTINATION]

Download UniCeption format checkpoints from AirLab Data Server

options:
  -h, --help            show this help message and exit
  --folders FOLDERS [FOLDERS ...]
                        List of folders to download (default: all folders). Choices: encoders, info_sharing, prediction_heads, examples
  --destination DESTINATION
                        Destination folder for downloaded checkpoints
```


## Currently Supported Components

### Encoders

Please refer to the `uniception/models/encoders` directory for the supported encoders and documentation for adding new encoders. The supported encoders can be listed by running:

```bash
python3 -m uniception.models.encoders.list
```

## Information Sharing Blocks

Please refer to the `uniception/models/info_sharing` directory for the supported information sharing blocks.

## Prediction Heads

Please refer to the `uniception/models/prediction_heads` directory for the supported prediction heads.

## Developer Guidelines

Please follow these guidelines when contributing to UniCeption:
- **Code Style**: Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for code style.
- **Documentation**: Add docstrings to all classes and methods.
- **Unit Tests**: Add necessary unit tests to the `tests` folder.
- **Linting**: Run `black` & `isort` on your code before committing. For example, you can run `black . && isort .`.

Since UniCeption is currently a private repo, please do not push to the main branch directly. Instead, create a new branch for your changes and submit a pull request for review.