# UniCeption

Repository to support the AirLab's Generalizable Perception Stack.

## Installation

Please install the `uniception` package in development mode by running the following command:

```bash
virtualenv uniception
source uniception/bin/activate
pip install torch torchvision torchaudio
pip install -r requirements.txt
cd uniception/models/libs/croco/curope
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