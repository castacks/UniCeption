# --------------------------------------------------------
# Tests for all encoders that forces uniception functionalities
# --------------------------------------------------------
import unittest
from functools import lru_cache
from typing import Tuple

import numpy as np
import requests
import torch
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from uniception.models.encoders import *
from uniception.models.encoders.image_normalizations import *


class TestEncoders(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestEncoders, self).__init__(*args, **kwargs)

        self.norm_types = ["dummy"]

        self.encoders = [
            "dummy"
        ]

        self.encoder_configs = [
            {}
        ]
    
    def inference_encoder(self, encoder, input):
        return encoder(input)

    def test_make_dummy_encoder(self):
        encoder = _make_encoder("dummy")
        self.assertTrue(encoder is not None)

    def test_make_croco_encoder(self):
        additional_stuff = {}
        encoder = _make_encoder("croco", **additional_stuff)
        self.assertTrue(encoder is not None)

    def test_all_encoder_basics(self):
        for encoder, encoder_config in zip(self.encoders, self.encoder_configs):
            encoder = _make_encoder(encoder, **encoder_config)
            self._check_baseclass_attribute(encoder)
            self._check_norm_check_function(encoder)

            if isinstance(encoder, UniCeptionViTEncoderBase):
                self._check_vit_encoder_attribute(encoder)

            if isinstance(encoder, IntermediateFeatureReturner):
                self._check_intermediate_feature_returner(encoder)

    def test_vit_encoder_patch_size(self):
        for encoder, encoder_config in zip(self.encoders, self.encoder_configs):
            encoder = _make_encoder(encoder, **encoder_config)

            if isinstance(encoder, UniCeptionViTEncoderBase):
                self._test_vit_encoder_patch_size(encoder)

    def _test_vit_encoder_patch_size(self, encoder):
        image_size = (16 * encoder.patch_size, 16 * encoder.patch_size)

        img = self._get_example_input(image_size, encoder.data_norm_type)
        # input and output of the encoder
        encoder_input = {"image": img, "data_norm_type": encoder.data_norm_type}

        # TODO: Add the encoder forward pass here
        encoder_output = self.inference_encoder(encoder, encoder_input)

        self.assertTrue(isinstance(encoder_output, torch.Tensor))
        self.assertTrue(encoder_output.shape[2] == 16)
        self.assertTrue(encoder_output.shape[3] == 16)

    def _visualize_encoder_features(self, encoder, image_size : Tuple[int, int]):
        img = self._get_example_input(image_size, encoder.data_norm_type)
        # input and output of the encoder
        encoder_input = {"image": img, "data_norm_type": encoder.data_norm_type}

        encoder_output = self.inference_encoder(encoder, encoder_input)

        encoder_output : torch.Tensor = torch.randn(1, 384, 14, 14)
        self.assertTrue(isinstance(encoder_output, torch.Tensor))

        # Visualize the features
        rgb_image = render_pca_as_rgb(encoder_output)
        
        plt.imshow(rgb_image)
        plt.savefig("pca_features.png")
    
    @lru_cache(maxsize=3)
    def _get_example_input(self, image_size : Tuple[int, int], image_norm_type : str = "dummy") -> torch.Tensor:
        url = "https://raw.githubusercontent.com/naver/croco/d3d0ab2858d44bcad54e5bfc24f565983fbe18d9/assets/Chateau1.png"
        image = Image.open(requests.get(url, stream=True).raw)
        image = image.resize(image_size)        
        image = image.convert("RGB")

        img = torch.from_numpy(np.array(image))

        # Normalize the images
        img_mean, img_std = IMAGE_NORMALIZATION_DICT[image_norm_type]
        img = (img.float() / 255.0 - img_mean) / img_std

        # convert to BCHW format
        img = img.permute(2, 0, 1).unsqueeze(0)

        return img
    
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
        self.assertTrue(hasattr(encoder, "_check_image_normalization_type"))

        encoder_notm_type = encoder.data_norm_type

        try:
            encoder._check_image_normalization_type(encoder_notm_type)
        except AssertionError:
            self.assertTrue(False)

        try:
            encoder._check_image_normalization_type("some_nonexistent_norm_type")
            self.assertTrue(False)
        except AssertionError:
            pass

    def _check_vit_encoder_attribute(self, encoder):
        self.assertTrue(hasattr(encoder, "patch_size"))
        self.assertTrue(isinstance(encoder.patch_size, int))
        self.assertTrue(encoder.patch_size > 0)

    def _check_intermediate_feature_returner(self, encoder):
        self.assertTrue(hasattr(encoder, "num_intermediate_layers"))
        self.assertTrue(hasattr(encoder, "selected_layers"))

        self.assertTrue(isinstance(encoder.num_intermediate_layers, int))
        self.assertTrue(isinstance(encoder.selected_layers, list))
        self.assertTrue(len(encoder.selected_layers) == encoder.num_intermediate_layers)

        for layer in encoder.selected_layers:
            self.assertTrue(layer < encoder.num_intermediate_layers)
            self.assertTrue(layer >= 0)

def render_pca_as_rgb(features):
    """
    Perform PCA on the given feature tensor and render the first 3 principal components as RGB.

    Args:
        features (torch.Tensor): Feature tensor of shape (B, C, H, W).

    Returns:
        np.ndarray: RGB image of shape (H, W, 3).
    """
    # Ensure input is a 4D tensor
    assert features.dim() == 4, "Input tensor must be 4D (B, C, H, W)"

    B, C, H, W = features.shape

    # Reshape the tensor to (B * H * W, C)
    reshaped_features = features.permute(0, 2, 3, 1).contiguous().view(-1, C).cpu().numpy()

    # Perform PCA
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(reshaped_features)

    # Rescale the principal components to [0, 1]
    principal_components = (principal_components - principal_components.min(axis=0)) / (principal_components.max(axis=0) - principal_components.min(axis=0))

    # Reshape the principal components to (B, H, W, 3)
    principal_components = principal_components.reshape(B, H, W, 3)

    # Convert the principal components to RGB image (take the first batch)
    rgb_image = principal_components[0]

    return rgb_image



if __name__ == "__main__":
    unittest.main()

    # Example usage of the PCA visualization
