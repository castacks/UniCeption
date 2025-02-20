"""
PCA Visualization of UniCeption Image Encoders + Multi-View Cross Attention Transformers
"""

import os
import random
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
from uniception.models.info_sharing.base import MultiViewTransformerInput
from uniception.models.info_sharing.cross_attention_transformer import MultiViewCrossAttentionTransformerIFR
from uniception.models.libs.croco.pos_embed import RoPE2D, get_2d_sincos_pos_embed


def _make_mv_cross_attention_transformer_test(model_str: str, **kwargs):
    current_file_path = os.path.abspath(__file__)
    relative_checkpoint_path = os.path.join(
        os.path.dirname(current_file_path), "../../../checkpoints/info_sharing/cross_attn_transformer"
    )
    rope = RoPE2D(float(100))
    if model_str == "croco":
        return MultiViewCrossAttentionTransformerIFR(
            name="croco_base_decoder",
            input_embed_dim=1024,
            num_views=2,
            indices=[12 * 2 // 4, 12 * 3 // 4],
            norm_intermediate=False,
            custom_positional_encoding=rope,
            pretrained_checkpoint_path=f"{relative_checkpoint_path}/Two_View_Cross_Attention_Transformer_CroCo.pth",
            **kwargs,
        )
    elif model_str == "dust3r_224":
        return MultiViewCrossAttentionTransformerIFR(
            name="dust3r_224_base_decoder",
            input_embed_dim=1024,
            num_views=2,
            indices=[12 * 2 // 4, 12 * 3 // 4],
            norm_intermediate=False,
            custom_positional_encoding=rope,
            pretrained_checkpoint_path=f"{relative_checkpoint_path}/Two_View_Cross_Attention_Transformer_DUSt3R_224_linear.pth",
            **kwargs,
        )
    elif model_str == "dust3r_512":
        return MultiViewCrossAttentionTransformerIFR(
            name="dust3r_512_base_decoder",
            input_embed_dim=1024,
            num_views=2,
            indices=[12 * 2 // 4, 12 * 3 // 4],
            norm_intermediate=False,
            custom_positional_encoding=rope,
            pretrained_checkpoint_path=f"{relative_checkpoint_path}/Two_View_Cross_Attention_Transformer_DUSt3R_512_linear.pth",
            **kwargs,
        )
    elif model_str == "dust3r_512_dpt":
        return MultiViewCrossAttentionTransformerIFR(
            name="dust3r_512_dpt_base_decoder",
            input_embed_dim=1024,
            num_views=2,
            indices=[12 * 2 // 4, 12 * 3 // 4],
            norm_intermediate=False,
            custom_positional_encoding=rope,
            pretrained_checkpoint_path=f"{relative_checkpoint_path}/Two_View_Cross_Attention_Transformer_DUSt3R_512_dpt.pth",
            **kwargs,
        )
    elif model_str == "mast3r_512":
        return MultiViewCrossAttentionTransformerIFR(
            name="mast3r_512_base_decoder",
            input_embed_dim=1024,
            num_views=2,
            indices=[12 * 2 // 4, 12 * 3 // 4],
            norm_intermediate=False,
            custom_positional_encoding=rope,
            pretrained_checkpoint_path=f"{relative_checkpoint_path}/Two_View_Cross_Attention_Transformer_MASt3R.pth",
            **kwargs,
        )


class TestMultiViewTransformers:
    def __init__(self, pca_save_folder, *args, **kwargs):
        super(TestMultiViewTransformers, self).__init__(*args, **kwargs)

        self.pca_save_folder = pca_save_folder

        self.norm_types = IMAGE_NORMALIZATION_DICT.keys()

        self.models = [
            "croco",
            "dust3r_224",
            "dust3r_512",
            "dust3r_512_dpt",
            "mast3r_512",
        ]

        self.model_configs = [{}] * len(self.models)

    def inference_encoder(self, encoder, input):
        return encoder(input)

    def inference_info_sharing(self, info_sharing, input):
        return info_sharing(input)

    def visualize_all_models(self):
        for model, model_config in zip(self.models, self.model_configs):
            encoder = _make_encoder_test(model, **model_config)
            info_sharing = _make_mv_cross_attention_transformer_test(model, **model_config)
            self._visualize_model_features_consistency(encoder, info_sharing, (224, 224))

    def _visualize_model_features_consistency(self, encoder, info_sharing, image_size: Tuple[int, int]):
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

        # pass the encoder outputs to the info sharing model
        multi_view_features = [encoder_output0, encoder_output1]
        info_sharing_input = MultiViewTransformerInput(features=multi_view_features)
        info_sharing_output = self.inference_info_sharing(info_sharing, info_sharing_input)
        final_layer_multi_view_features = info_sharing_output[0].features

        # get a common PCA codec
        cat_feats = torch.cat(final_layer_multi_view_features, dim=3)

        pca_viz = get_pca_map(cat_feats.permute(0, 2, 3, 1), (image_size[0], image_size[1] * 2), return_pca_stats=True)

        # concatenate the input images along the width dimension
        cat_imgs = torch.cat([viz_img0, viz_img1], dim=1)

        # plot the input image and the PCA features
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(cat_imgs)
        axs[0].set_title("Input Images")
        axs[0].axis("off")
        axs[1].imshow(pca_viz[0])
        axs[1].set_title(f"PCA Features of {encoder.name} + Base Decoder")
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
    relative_pca_image_folder = os.path.join(
        os.path.dirname(current_file_path), "../../../local/info_sharing/pca_images"
    )
    os.makedirs(relative_pca_image_folder, exist_ok=True)

    # Initialize the test class
    test = TestMultiViewTransformers(pca_save_folder=relative_pca_image_folder)

    # Visualize the PCA of all models
    test.visualize_all_models()

    print(f"The PCA visualizations of all models are saved successfully to {relative_pca_image_folder}!")
