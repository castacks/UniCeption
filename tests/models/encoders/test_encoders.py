import os
import random
from functools import lru_cache
from typing import Tuple

import numpy as np
import pytest
import requests
import torch
from PIL import Image

from uniception.models.encoders import *
from uniception.models.encoders.image_normalizations import *


@pytest.fixture(scope="module")
def norm_types():
    return IMAGE_NORMALIZATION_DICT.keys()


@pytest.fixture(scope="module")
def encoders():
    return [
        "croco",
        "dust3r_224",
        "dust3r_512",
        "dust3r_512_dpt",
        "mast3r_512",
        "dinov2_base",
        "dinov2_large",
        "dinov2_large_reg",
        "dinov2_large_dav2",
        "dinov2_giant",
        "dinov2_giant_reg",
        "radio_v2.5-b",
        "radio_v2.5-l",
        "e-radio_v2",
        "naradio_v2.5-b",
        "naradio_v2.5-l",
    ]


@pytest.fixture(scope="module")
def encoder_configs(encoders):
    # Adjust the number of configs to match the number of encoders
    return [{}] * len(encoders)


@pytest.fixture
def device(request):
    # Access the value of the custom option for device
    device_str = request.config.getoption("--device")
    if device_str == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")  # Use the default CUDA device
    else:
        device = torch.device("cpu")
    print(f"Using device: {device.type.upper()}")
    return device


@pytest.fixture
def example_input(device):
    @lru_cache(maxsize=3)
    def _get_example_input(
        image_size: Tuple[int, int],
        image_norm_type: str = "dummy",
        img_selection: int = 1,
        return_viz_img: bool = False,
    ) -> torch.Tensor:
        url = f"https://raw.githubusercontent.com/naver/croco/d3d0ab2858d44bcad54e5bfc24f565983fbe18d9/assets/Chateau{img_selection}.png"
        image = Image.open(requests.get(url, stream=True).raw)
        image = image.resize(image_size)
        image = image.convert("RGB")

        img = torch.from_numpy(np.array(image))
        viz_img = img.clone()

        # Normalize the image
        image_normalization = IMAGE_NORMALIZATION_DICT[image_norm_type]
        img_mean = image_normalization.mean
        img_std = image_normalization.std
        img = (img.float() / 255.0 - img_mean) / img_std

        # Convert to BCHW format
        img = img.permute(2, 0, 1).unsqueeze(0).to(device)

        if return_viz_img:
            return img, viz_img
        else:
            return img

    return _get_example_input


def inference_encoder(encoder, encoder_input):
    # Encoder expects a ViTEncoderInput object
    return encoder(encoder_input).features


def test_make_dummy_encoder(device):
    print(f"Testing Init of Dummy Encoder on {device.type.upper()}")
    encoder = _make_encoder_test("dummy").to(device)

    # Check if the encoder has parameters
    try:
        params = list(encoder.parameters())
        if not params:
            print("Warning: The encoder has no parameters.")
        else:
            # Verify if the model is on the right device
            assert params[0].is_cuda == (device.type == "cuda")

    except Exception as e:
        print(f"Error: {e}")
        assert False  # Fail the test if any error occurs

    assert encoder is not None


def test_all_encoder_basics(encoders, encoder_configs, norm_types, example_input, encoder_name, device):
    if encoder_name:
        encoders = [encoder_name]  # Override default encoders with the one specified

    for encoder_name, encoder_config in zip(encoders, encoder_configs):
        print(f"Testing encoder: {encoder_name} on {device.type.upper()}")

        encoder = _make_encoder_test(encoder_name, **encoder_config).to(device)
        _check_baseclass_attribute(encoder, norm_types)
        _check_norm_check_function(encoder)

        if isinstance(encoder, UniCeptionViTEncoderBase):
            _check_vit_encoder_attribute(encoder)
            _test_vit_encoder_patch_size(encoder, example_input)


def _check_baseclass_attribute(encoder, norm_types):
    assert hasattr(encoder, "name")
    assert hasattr(encoder, "size")
    assert hasattr(encoder, "data_norm_type")

    assert isinstance(encoder.name, str)
    assert isinstance(encoder.size, str) or encoder.size is None
    assert isinstance(encoder.data_norm_type, str)

    # Check if the data_norm_type is in the list of normalization types
    assert encoder.data_norm_type in norm_types


def _check_norm_check_function(encoder):
    assert hasattr(encoder, "_check_data_normalization_type")

    encoder_notm_type = encoder.data_norm_type

    try:
        encoder._check_data_normalization_type(encoder_notm_type)
    except AssertionError:
        assert False

    try:
        encoder._check_data_normalization_type("some_nonexistent_norm_type")
        assert False
    except AssertionError:
        pass


def _check_vit_encoder_attribute(encoder):
    assert hasattr(encoder, "patch_size")
    assert isinstance(encoder.patch_size, int)
    assert encoder.patch_size > 0


def _test_vit_encoder_patch_size(encoder, example_input):
    print(f"Testing {encoder.name} inference")
    image_size = (14 * encoder.patch_size, 14 * encoder.patch_size)

    img = example_input(image_size, encoder.data_norm_type)
    # Create an instance of ViTEncoderInput with correct attributes
    encoder_input = ViTEncoderInput(
        data_norm_type=encoder.data_norm_type,
        image=img,
    )

    encoder_output = inference_encoder(encoder, encoder_input)

    assert isinstance(encoder_output, torch.Tensor)
    assert encoder_output.shape[2] == 14
    assert encoder_output.shape[3] == 14


@pytest.fixture(scope="session", autouse=True)
def seed_everything():
    seed = 42
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to: {seed} (type: {type(seed)})")

    # Turn XFormers off for testing on CPU
    os.environ["XFORMERS_DISABLED"] = "1"
