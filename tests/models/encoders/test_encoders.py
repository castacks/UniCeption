"""
Tests for all image encoders that force uniception functionalities
"""

import os
import random
import unittest
from functools import lru_cache
from typing import Tuple

import numpy as np
import requests
import torch
from PIL import Image

from uniception.models.encoders import *
from uniception.models.encoders.image_normalizations import *


class TestEncoders(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestEncoders, self).__init__(*args, **kwargs)

        self.norm_types = IMAGE_NORMALIZATION_DICT.keys()

        # self.encoders = [
        #     "croco",
        #     "dust3r_224",
        #     "dust3r_512",
        #     "dust3r_512_dpt",
        #     "mast3r_512",
        #     "dinov2_large",
        #     "dinov2_large_reg",
        #     "dinov2_large_dav2",
        #     "dinov2_giant",
        #     "dinov2_giant_reg",
        #     "radio_v2.5-b",
        #     "radio_v2.5-l",
        #     "e-radio_v2",
        # ]
        self.encoders = [
            "patch_embedder",
        ]

        self.encoder_configs = [{}] * len(self.encoders)

    def inference_encoder(self, encoder, input):
        return encoder(input)

    def test_make_dummy_encoder(self):
        print("Testing Init of Dummy Encoder")
        encoder = _make_encoder_test("dummy")
        self.assertTrue(encoder is not None)

    def test_all_encoder_basics(self):
        for encoder, encoder_config in zip(self.encoders, self.encoder_configs):
            print(f"Testing encoder: {encoder}")

            encoder = _make_encoder_test(encoder, **encoder_config)
            self._check_baseclass_attribute(encoder)
            self._check_norm_check_function(encoder)

            if isinstance(encoder, UniCeptionViTEncoderBase):
                self._check_vit_encoder_attribute(encoder)
                self._test_vit_encoder_patch_size(encoder)

    @lru_cache(maxsize=3)
    def _get_example_input(
        self,
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

        # Normalize the images
        image_normalization = IMAGE_NORMALIZATION_DICT[image_norm_type]

        img_mean, img_std = image_normalization.mean, image_normalization.std

        img = (img.float() / 255.0 - img_mean) / img_std

        # convert to BCHW format
        img = img.permute(2, 0, 1).unsqueeze(0)

        if return_viz_img:
            return img, viz_img
        else:
            return img

    def _test_vit_encoder_patch_size(self, encoder):
        print(f"Testing patch size for encoder: {encoder.name}")
        image_size = (14 * encoder.patch_size, 14 * encoder.patch_size)

        img = self._get_example_input(image_size, encoder.data_norm_type)
        # input and output of the encoder
        encoder_input: ViTEncoderInput = ViTEncoderInput(
            data_norm_type=encoder.data_norm_type,
            image=img,
        )

        encoder_output = self.inference_encoder(encoder, encoder_input).features

        self.assertTrue(isinstance(encoder_output, torch.Tensor))
        self.assertTrue(encoder_output.shape[2] == 14)
        self.assertTrue(encoder_output.shape[3] == 14)

    def _check_baseclass_attribute(self, encoder):
        self.assertTrue(hasattr(encoder, "name"))
        self.assertTrue(hasattr(encoder, "size"))
        self.assertTrue(hasattr(encoder, "data_norm_type"))

        self.assertTrue(isinstance(encoder.name, str))
        self.assertTrue(isinstance(encoder.size, str) or encoder.size is None)
        self.assertTrue(isinstance(encoder.data_norm_type, str))

        # Check if the data_norm_type is in the list of normalization types
        self.assertTrue(encoder.data_norm_type in self.norm_types)

    def _check_norm_check_function(self, encoder):
        self.assertTrue(hasattr(encoder, "_check_data_normalization_type"))

        encoder_notm_type = encoder.data_norm_type

        try:
            encoder._check_data_normalization_type(encoder_notm_type)
        except AssertionError:
            self.assertTrue(False)

        try:
            encoder._check_data_normalization_type("some_nonexistent_norm_type")
            self.assertTrue(False)
        except AssertionError:
            pass

    def _check_vit_encoder_attribute(self, encoder):
        self.assertTrue(hasattr(encoder, "patch_size"))
        self.assertTrue(isinstance(encoder.patch_size, int))
        self.assertTrue(encoder.patch_size > 0)


def seed_everything(seed=42):
    """
    Set the `seed` value for torch and numpy seeds. Also turns on
    deterministic execution for cudnn.

    Parameters:
    - seed:     A hashable seed value
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to: {seed} (type: {type(seed)})")


if __name__ == "__main__":
    # Turn XFormers off for testing on CPU
    os.environ["XFORMERS_DISABLED"] = "1"

    # Seed everything for consistent testing
    seed_everything()

    # Test the Encoders
    unittest.main()
