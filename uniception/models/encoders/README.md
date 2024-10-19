# UniCeption Encoders

## Currently Supported Encoders

### UniCeptionViTEncoderBase:

- `CroCoEncoder`
   - `CroCoIntermediateFeatureReturner`
- `DINOv2Encoder`
   - `DINOv2IntermediateFeatureReturner`
- `RADIOEncoder`
   - `RADIOIntermediateFeatureReturner`

# Developer Guidelines for UniCeption Encoders

## Overview

This folder contains the implementation of various UniCeption encoders. Each encoder must adhere to a specific structure and follow certain guidelines to ensure consistency and compatibility across different projects.

## Directory Structure

The encoders and other necessary dependencies/tests for encoders are organized as follows:
```
uniception/
├── models/
│   ├── encoders/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── croco.py
│   │   ├── dinov2.py
│   │   ├── radio.py
│   │   ├── image_normalizations.py
│   └── ...
│   └── libs/
│   │   ├── external_dependency_folders/
|   |   |   ├── external_dependency_files
tests/
├── models/
│   ├── encoders/
│   │   ├── test_encoders.py
│   │   ├── viz_image_encoders.py
│   │   └── ...
|   └── ...
└── ...
```

## Adding a New Encoder

To add a new encoder, follow these steps:

1. **Create a New Encoder File**:
   - Create a new file in the `encoders` directory, e.g., `new_encoder.py`.
   - Define the new encoder class in this file, inheriting from `UniCeptionEncoderBase` or `UniCeptionViTEncoderBase`.
   - Please look at the base class for the necessary attributes and methods to implement.

2. **Define Input Data Normalization**:
   - Add the corresponding normalization for the encoder to respective normalization files, for example, image normalizations should be added to `image_normalizations.py`.
   - Ensure the normalization is added to the dictionaries present in the files, for example, `IMAGE_NORMALIZATION_DICT`.

4. **Implement the Encoder Class**:
   - Inherit from `UniCeptionEncoderBase` or `UniCeptionViTEncoderBase` or other UniCeption base classes.
   - Implement the `forward` method.
   - Ensure the encoder class has the necessary attributes and methods.

4. **Update `__init__.py`**:
   - Import the new encoder class in `__init__.py`.
   - Update the `_make_encoder_test` function to include the new encoder.

5. **Run Encoder Unit Tests**:
   - Add your new encoder to the encoder list in the TestEncoders class in `tests/models/encoders/test_encoders.py`.
   - Run `python3 tests/models/encoders/test_encoders.py` to test the basic expected functionality of UniCeption encoders.
   - Optionally, for image encoders, the unit tests in `tests/models/encoders/viz_image_encoders.py` save PCA visualizations of the encoder outputs to the `local/pca_images` directory.

## Example Encoder Implementation

Here is an example of how to implement a new encoder:

```python
# new_encoder.py
import torch
from uniception.models.encoders.base import UniCeptionEncoderBase, EncoderInput, EncoderOutput

class NewEncoder(UniCeptionEncoderBase):
    def __init__(self, name: str, data_norm_type: str, *args, **kwargs):
        super().__init__(name=name, data_norm_type=data_norm_type, *args, **kwargs)
        # Initialize encoder-specific layers and parameters here

    def forward(self, encoder_input: EncoderInput) -> EncoderOutput:
        self._check_data_normalization_type(encoder_input.data_norm_type)
        # Implement the forward pass
        return EncoderOutput()
```

## Example Normalization

Add the normalization for the new encoder, for example, to `image_normalizations.py`:

```python
# image_normalizations.py
IMAGE_NORMALIZATION_DICT = {
    "dummy": ImageNormalization(mean=torch.tensor([0.0, 0.0, 0.0]), std=torch.tensor([1.0, 1.0, 1.0])),
    "croco": ImageNormalization(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])),
    "dust3r": ImageNormalization(mean=torch.tensor([0.5, 0.5, 0.5]), std=torch.tensor([0.5, 0.5, 0.5])),
    "dinov2": ImageNormalization(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])),
    "radio": ImageNormalization(mean=torch.tensor([0.0, 0.0, 0.0]), std=torch.tensor([1.0, 1.0, 1.0])),
    "new_encoder": ImageNormalization(mean=torch.tensor([0.5, 0.5, 0.5]), std=torch.tensor([0.2, 0.2, 0.2])),
}
```

## Example Unit Testing

Add the new encoder to the encoder factory in `__init__.py` and the encoder list in `tests/models/encoders/test_encoders.py`. Additional tests can also be added as required.

Look at `tests/models/encoders/test_encoders.py` to see what tests are run.

Additionally, if the new encoder is an image encoder, you can add to the encoder list in `tests/models/encoders/viz_image_encoders.py` for saving PCA visualizations of the encoder outputs to the `local/pca_images` directory.

## Developer Guidelines

Please follow these guidelines when contributing to the UniCeption encoders:
- **Consistency**: Ensure that the new encoder follows the structure and naming conventions of existing encoders.
- **Code Style**: Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for code style.
- **Documentation**: Add docstrings to all classes and methods.
- **Unit Tests**: Add necessary unit tests for the encoder class.
- **Linting**: Run `black` on your code before committing. For example, you can run `black uniception`.

## Happy Coding!
