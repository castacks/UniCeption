# UniCeption

Repository to support the AirLab's Generalizable Perception Stack.

## Installation

Clone the repository to your local machine by running the following command:

```bash
git clone git@github.com:castacks/UniCeption.git
```

Install the `uniception` package in development mode by running the following commands:

```bash
virtualenv uniception
source uniception/bin/activate
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
pip install -U xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
cd uniception/models/libs/croco/curope # For CroCo
python3 setup.py build_ext --inplace
cd ../../../../../
pip install -e .
```

## Currently Supported Components

### Encoders

Please refer to the `uniception/models/encoders` directory for the supported encoders and documentation for adding new encoders.

## Developer Guidelines

Please follow these guidelines when contributing to UniCeption:
- **Code Style**: Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for code style.
- **Documentation**: Add docstrings to all classes and methods.
- **Unit Tests**: Add necessary unit tests to the `tests` folder.
- **Linting**: Run `black` on your code before committing. For example, you can run `black uniception`.