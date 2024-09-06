# --------------------------------------------------------
# Tests for all encoders that forces uniception functionalities
# --------------------------------------------------------
import os
import random
import unittest
from functools import lru_cache
from typing import Tuple

import numpy as np
import requests
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from uniception.models.encoders import *
from uniception.models.encoders.image_normalizations import *


class TestEncoders(unittest.TestCase):

    def __init__(self, pca_save_folder, *args, **kwargs):
        super(TestEncoders, self).__init__(*args, **kwargs)

        self.pca_save_folder = pca_save_folder

        self.norm_types = IMAGE_NORMALIZATION_DICT.keys()

        self.encoders = [
            "croco",
            "dust3r_224",
            "dust3r_512",
            "dust3r_512_dpt",
            "mast3r_512",
            "dinov2_large",
            "dinov2_large_reg",
            "dinov2_large_dav2",
            "dinov2_giant",
            "dinov2_giant_reg",
            "radio_v2.5-b",
            "radio_v2.5-l",
            "e-radio_v2",
        ]

        self.encoder_configs = [{}] * len(self.encoders)

    def inference_encoder(self, encoder, input):
        return encoder(input)

    def test_make_dummy_encoder(self):
        encoder = _make_encoder_test("dummy")
        self.assertTrue(encoder is not None)

    def test_make_croco_encoder(self):
        additional_stuff = {}
        encoder = _make_encoder_test("croco", **additional_stuff)
        self.assertTrue(encoder is not None)

    def test_all_encoder_basics(self):
        for encoder, encoder_config in zip(self.encoders, self.encoder_configs):
            print(f"Testing encoder: {encoder}")

            encoder = _make_encoder_test(encoder, **encoder_config)
            self._check_baseclass_attribute(encoder)
            self._check_norm_check_function(encoder)

            if isinstance(encoder, UniCeptionViTEncoderBase):
                self._check_vit_encoder_attribute(encoder)

            if isinstance(encoder, IntermediateFeatureReturner):
                self._check_intermediate_feature_returner(encoder)

    def test_vit_encoder_patch_size(self):
        for encoder, encoder_config in zip(self.encoders, self.encoder_configs):
            encoder = _make_encoder_test(encoder, **encoder_config)

            if isinstance(encoder, UniCeptionViTEncoderBase):
                self._test_vit_encoder_patch_size(encoder)

    def _test_vit_encoder_patch_size(self, encoder):
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

    def visualize_all_encoders(self):
        for encoder, encoder_config in zip(self.encoders, self.encoder_configs):
            encoder = _make_encoder_test(encoder, **encoder_config)
            self._visualize_encoder_features_consistency(encoder, (224, 224))

    def _visualize_encoder_features(self, encoder, image_size: Tuple[int, int]):
        img, viz_img = self._get_example_input(image_size, encoder.data_norm_type, return_viz_img=True)
        # input and output of the encoder
        encoder_input: ViTEncoderInput = ViTEncoderInput(
            data_norm_type=encoder.data_norm_type,
            image=img,
        )

        encoder_output = self.inference_encoder(encoder, encoder_input)
        encoder_output = encoder_output.features

        self.assertTrue(isinstance(encoder_output, torch.Tensor))

        # visualize the features
        pca_viz = get_pca_map(encoder_output.permute(0, 2, 3, 1), image_size, return_pca_stats=False)

        # plot the input image and the PCA features
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(viz_img)
        axs[0].set_title("Input Image")
        axs[0].axis("off")
        axs[1].imshow(pca_viz)
        axs[1].set_title(f"PCA Features of {encoder.name}")
        axs[1].axis("off")
        plt.savefig(f"{self.pca_save_folder}/pca_{encoder.name}.png", bbox_inches="tight")
        plt.close()

    def _visualize_encoder_features_consistency(self, encoder, image_size: Tuple[int, int]):

        img0, viz_img0 = self._get_example_input(
            image_size, encoder.data_norm_type, img_selection=1, return_viz_img=True
        )
        img1, viz_img1 = self._get_example_input(
            image_size, encoder.data_norm_type, img_selection=2, return_viz_img=True
        )
        # input and output of the encoder
        encoder_input0: ViTEncoderInput = ViTEncoderInput(
            data_norm_type=encoder.data_norm_type,
            image=img0,
        )

        encoder_input1: ViTEncoderInput = ViTEncoderInput(
            data_norm_type=encoder.data_norm_type,
            image=img1,
        )

        encoder_output0 = self.inference_encoder(encoder, encoder_input0)
        encoder_output0 = encoder_output0.features

        encoder_output1 = self.inference_encoder(encoder, encoder_input1)
        encoder_output1 = encoder_output1.features

        # get a common PCA codec
        cat_feats = torch.cat([encoder_output0, encoder_output1], dim=3)

        pca_viz = get_pca_map(cat_feats.permute(0, 2, 3, 1), (image_size[0], image_size[1] * 2), return_pca_stats=True)

        # concatenate the input images along the width dimension
        cat_imgs = torch.cat([viz_img0, viz_img1], dim=1)

        # plot the input image and the PCA features
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(cat_imgs)
        axs[0].set_title("Input Images")
        axs[0].axis("off")
        axs[1].imshow(pca_viz[0])
        axs[1].set_title(f"PCA Features of {encoder.name}")
        axs[1].axis("off")
        plt.savefig(f"{self.pca_save_folder}/multi_pca_{encoder.name}.png", bbox_inches="tight")
        plt.close()

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
    principal_components = (principal_components - principal_components.min(axis=0)) / (
        principal_components.max(axis=0) - principal_components.min(axis=0)
    )

    # Reshape the principal components to (B, H, W, 3)
    principal_components = principal_components.reshape(B, H, W, 3)

    # Convert the principal components to RGB image (take the first batch)
    rgb_image = principal_components[0]

    return rgb_image


def get_robust_pca(features: torch.Tensor, m: float = 2, remove_first_component=False):
    # features: (N, C)
    # m: a hyperparam controlling how many std dev outside for outliers
    assert len(features.shape) == 2, "features should be (N, C)"
    reduction_mat = torch.pca_lowrank(features, q=3, niter=20)[2]
    colors = features @ reduction_mat
    if remove_first_component:
        colors_min = colors.min(dim=0).values
        colors_max = colors.max(dim=0).values
        tmp_colors = (colors - colors_min) / (colors_max - colors_min)
        fg_mask = tmp_colors[..., 0] < 0.2
        reduction_mat = torch.pca_lowrank(features[fg_mask], q=3, niter=20)[2]
        colors = features @ reduction_mat
    else:
        fg_mask = torch.ones_like(colors[:, 0]).bool()
    d = torch.abs(colors[fg_mask] - torch.median(colors[fg_mask], dim=0).values)
    mdev = torch.median(d, dim=0).values
    s = d / mdev
    try:
        rins = colors[fg_mask][s[:, 0] < m, 0]
        gins = colors[fg_mask][s[:, 1] < m, 1]
        bins = colors[fg_mask][s[:, 2] < m, 2]
        rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
        rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])
    except:
        rins = colors
        gins = colors
        bins = colors
        rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
        rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])

    return reduction_mat, rgb_min.to(reduction_mat), rgb_max.to(reduction_mat)


def get_pca_map(
    feature_map: torch.Tensor,
    img_size,
    interpolation="bicubic",
    return_pca_stats=False,
    pca_stats=None,
):
    """
    feature_map: (1, h, w, C) is the feature map of a single image.
    """
    if feature_map.shape[0] != 1:
        # make it (1, h, w, C)
        feature_map = feature_map[None]
    if pca_stats is None:
        reduct_mat, color_min, color_max = get_robust_pca(feature_map.reshape(-1, feature_map.shape[-1]))
    else:
        reduct_mat, color_min, color_max = pca_stats
    pca_color = feature_map @ reduct_mat
    pca_color = (pca_color - color_min) / (color_max - color_min)
    pca_color = pca_color.clamp(0, 1)
    pca_color = F.interpolate(
        pca_color.permute(0, 3, 1, 2),
        size=img_size,
        mode=interpolation,
    ).permute(0, 2, 3, 1)
    pca_color = pca_color.detach().cpu().numpy().squeeze(0)
    if return_pca_stats:
        return pca_color, (reduct_mat, color_min, color_max)
    return pca_color


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

    # Create local directory for storing the PCA images
    current_file_path = os.path.abspath(__file__)
    relative_pca_image_folder = os.path.join(os.path.dirname(current_file_path), "../local/pca_images")
    os.makedirs(relative_pca_image_folder, exist_ok=True)

    # Test the Encoders
    test = TestEncoders(pca_save_folder=relative_pca_image_folder)

    # Visualize the PCA of all encoders
    test.visualize_all_encoders()

    print("All encoder tests passed successfully!")
