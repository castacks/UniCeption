"""
Encoder Class for NARADIO (RayFronts)
"""

import math
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

from uniception.models.encoders.base import UniCeptionViTEncoderBase, ViTEncoderInput, ViTEncoderOutput
from uniception.models.utils.intermediate_feature_return import FeatureWrapper, IntermediateFeatureReturner


class GaussKernelAttn(nn.Module):
    """Implementation of Gaussian Kernel based Attention using FlexAttention"""

    def __init__(
        self,
        orig_attn,
        input_resolution: tuple,
        gauss_std: float,
        chosen_cls_id: int,
        dim: int,
        qk_norm: bool = False,
        num_prefix_tokens: int = 8,
    ) -> None:
        super().__init__()
        num_heads = orig_attn.num_heads
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.addition_cache = dict()
        self.input_resolution = input_resolution
        self.chosen_cls_id = chosen_cls_id
        self.gauss_std = gauss_std

        self.qkv = orig_attn.qkv
        self.q_norm = orig_attn.q_norm if qk_norm else nn.Identity()
        self.k_norm = orig_attn.k_norm if qk_norm else nn.Identity()
        self.attn_drop = orig_attn.attn_drop
        self.proj = orig_attn.proj
        self.proj_drop = orig_attn.proj_drop
        self.num_prefix_tokens = num_prefix_tokens

        # Cache for Gaussian window addition
        self.cached_addition = None
        self.cached_n_patches = None

    @staticmethod
    def gaussian_window(dim1, dim2, std=7.0):
        constant = 1 / (std * math.sqrt(2))
        ks = list()
        for dim in [dim1, dim2]:
            start = -(dim - 1) / 2.0
            k = torch.linspace(start=start * constant, end=(start + (dim - 1)) * constant, steps=dim, dtype=torch.float)
            ks.append(k)
        dist_square_to_mu = (torch.stack(torch.meshgrid(*ks, indexing="ij")) ** 2).sum(0)

        return torch.exp(-dist_square_to_mu)

    @staticmethod
    def get_attention_addition(dim1, dim2, window, num_prefix_tokens=8):
        m = torch.einsum("ij,kl->ijkl", torch.eye(dim1), torch.eye(dim2))
        m = m.permute((0, 3, 1, 2)).contiguous()
        out = F.conv2d(m.view(-1, dim1, dim2).unsqueeze(1), window.unsqueeze(0).unsqueeze(1), padding="same").squeeze(1)

        out = out.view(dim1 * dim2, dim1 * dim2)
        if num_prefix_tokens > 0:
            v_adjusted = torch.vstack([torch.zeros((num_prefix_tokens, dim1 * dim2)), out])
            out = torch.hstack([torch.zeros((dim1 * dim2 + num_prefix_tokens, num_prefix_tokens)), v_adjusted])

        return out

    def prepare_gaussian_addition(self, n_patches, device):
        """Prepare the Gaussian addition matrix for the current input"""
        if self.cached_n_patches != n_patches:
            window_size = [side * 2 - 1 for side in n_patches]
            window = self.gaussian_window(*window_size, std=self.gauss_std)
            addition = self.get_attention_addition(*n_patches, window, self.num_prefix_tokens).to(device)

            self.cached_addition = addition
            self.cached_n_patches = n_patches

        return self.cached_addition

    def gauss_score_mod(self, score, b, h, q_idx, kv_idx, addition):
        """Score modification function for FlexAttention"""
        # Adding the precomputed Gaussian pattern to the attention score
        return score + addition[q_idx, kv_idx]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        h, w = self.input_resolution
        n_patches = (w // 16, h // 16)

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        addition = self.prepare_gaussian_addition(n_patches, device=x.device)

        # Create a score_mod function with the current addition matrix
        score_mod = lambda score, b, h, q_idx, kv_idx: self.gauss_score_mod(score, b, h, q_idx, kv_idx, addition)

        # Use FlexAttention
        attn_output = flex_attention(q, k, v, score_mod=score_mod)

        # Reshape output and apply projection
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)

        return attn_output


class NARADIOEncoder(UniCeptionViTEncoderBase):
    """UniCeption NARADIO Encoder based on NACLIP + RADIO models

    The model modifies the attention of the last layer of RADIO following the
    example of NACLIP improving spatial structure. And uses the Summary CLS
    projection to project the patch-wise tokens to SIGLIP or CLIP language aligned
    feature spaces. The model computes na-radio spatial or global features by
    default and exposes functions to project those features to Siglip, or CLIP
    feature spaces.
    """

    def __init__(
        self,
        name: str,
        data_norm_type: str = "radio",
        patch_size: int = 16,
        model_version: str = "radio_v2.5-l",
        input_resolution: Tuple[int, int] = [512, 512],
        gauss_std: float = 7.0,
        pretrained_checkpoint_path: str = None,
        torch_hub_force_reload: bool = False,
        *args,
        **kwargs,
    ):
        """
        NARADIO Encoder for extracting spatial features from images.

        Args:
            name (str): Name of the encoder.
            data_norm_type (str): Image normalization type. Default: "radio"
            patch_size (int): Patch size for the encoder. Default: 16
            model_version (str): Version of the RADIO model to load. Default: "radio_v2.5-l"
            input_resolution: Tuple of ints (height, width) of the input images, needed to initialize the guassian attention window.
            gauss_std: Standard deviation of the gaussian kernel.
            pretrained_checkpoint_path (str): Path to the pretrained checkpoint if using custom trained version of RADIO. Default: None
            torch_hub_force_reload (bool): Whether to force reload the model from torch hub. Default: False
        """
        # Init the base class
        super().__init__(
            name=name,
            data_norm_type=data_norm_type,
            patch_size=patch_size,
            *args,
            **kwargs,
        )

        # Init the RADIO Encoder specific attributes
        self.model_version = model_version

        # Load the pretrained RADIO model from torch hub
        print(f"Loading pretrained {self.model_version} from torch hub")
        try:  # Requires internet access
            self.model = torch.hub.load(
                "NVlabs/RADIO",
                "radio_model",
                version=self.model_version,
                progress=True,
                skip_validation=True,
                force_reload=torch_hub_force_reload,
            )
        except:  # Load from cache
            self.model = torch.hub.load(
                "NVlabs/RADIO",
                "radio_model",
                version=self.model_version,
                progress=True,
                skip_validation=True,
            )

        # Load the custom pretrained checkpoint if provided
        if pretrained_checkpoint_path is not None:
            print(f"Loading custom pretrained RADIO checkpoint from {pretrained_checkpoint_path}")
            ckpt = torch.load(pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"]))

        self.enc_embed_dim = self.model.model.embed_dim

        # Replace the attention of the last ViT block with the Gaussian Kernel based attention
        self.model.model.blocks[-1] = GaussKernelAttn(
            self.model.model.blocks[-1].attn,
            input_resolution,
            gauss_std,
            dim=self.enc_embed_dim,
            chosen_cls_id=None,
            num_prefix_tokens=self.model.num_summary_tokens,
        )

    def forward(self, encoder_input: ViTEncoderInput) -> ViTEncoderOutput:
        """
        NARADIO Encoder Forward Pass

        Args:
            encoder_input (ViTEncoderInput): Input data for the encoder. Input data must contain image normalization type and normalized image tensor.

        Returns:
            ViTEncoderOutput: Output data from the encoder.
        """
        # Check image normalization type
        self._check_data_normalization_type(encoder_input.data_norm_type)

        # Check the dtype and shape of the input image
        assert isinstance(encoder_input.image, torch.Tensor), "Input must be a torch.Tensor"
        assert encoder_input.image.ndim == 4, "Input must be of shape (B, C, H, W)"
        batch_size, channels, height, width = encoder_input.image.shape
        assert channels == 3, "Input must have 3 channels"
        assert (
            height % self.patch_size == 0 and width % self.patch_size == 0
        ), f"Input shape must be divisible by patch size: {self.patch_size}"

        # Forward pass throught the RADIO encoder
        summary, features = self.model(encoder_input.image)

        # Resize the features to the expected shape
        # (B x Num_patches x Embed_dim) -> (B x Embed_dim x H / Patch_Size x W / Patch_Size)
        features = features.permute(0, 2, 1)
        features = features.reshape(
            -1, self.enc_embed_dim, height // self.patch_size, width // self.patch_size
        ).contiguous()

        return ViTEncoderOutput(features=features)


class NARADIOIntermediateFeatureReturner(NARADIOEncoder, IntermediateFeatureReturner):
    "Intermediate Feature Returner for UniCeption NARADIO Encoder"

    def __init__(
        self,
        name: str,
        data_norm_type: str = "radio",
        patch_size: int = 16,
        model_version: str = "radio_v2.5-l",
        input_resolution: Tuple[int, int] = [512, 512],
        gauss_std: float = 7.0,
        pretrained_checkpoint_path: str = None,
        indices: Union[int, List[int]] = [-1],
        norm_intermediate: bool = True,
        stop_early: bool = False,
        intermediates_only: bool = True,
        *args,
        **kwargs,
    ):
        """
        Intermediate Feature Returner for the RADIO Encoder.

        Args:
            name (str): Name of the encoder.
            data_norm_type (str): Image normalization type. Default: "radio"
            patch_size (int): Patch size for the encoder. Default: 16
            model_version (str): Version of the RADIO model to load. Default: "radio_v2.5-l"
            pretrained_checkpoint_path (str): Path to the pretrained checkpoint if using custom trained version of RADIO.
            indices (Optional[Union[int, List[int]]], optional): Indices of the layers to return. Defaults to [-1]. Options:
            - int: Return the last n layers.
            - List[int]: Return the intermediate layers at the specified indices.
            norm_intermediate (bool, optional): Whether to normalize the intermediate features. Defaults to True.
            stop_early (bool, optional): Whether to stop early. Defaults to False.
            intermediates_only (bool, optional): Whether to return only the intermediate features. Defaults to True.
        """
        # Init the base classes
        NARADIOEncoder.__init__(
            self,
            name=name,
            data_norm_type=data_norm_type,
            patch_size=patch_size,
            model_version=model_version,
            input_resolution=input_resolution,
            gauss_std=gauss_std,
            pretrained_checkpoint_path=pretrained_checkpoint_path,
            *args,
            **kwargs,
        )
        IntermediateFeatureReturner.__init__(
            self,
            indices=indices,
            norm_intermediate=norm_intermediate,
            stop_early=stop_early,
            intermediates_only=intermediates_only,
        )

    def forward(
        self, encoder_input: ViTEncoderInput
    ) -> Union[List[ViTEncoderOutput], Tuple[ViTEncoderOutput, List[ViTEncoderOutput]]]:
        """
        NARADIO Encoder Forward Pass with Intermediate Feature Return

        Args:
            encoder_input (ViTEncoderInput): Input data for the encoder. Input data must contain image normalization type and normalized image tensor.

        Returns:
            Union[List[ViTEncoderOutput], Tuple[ViTEncoderOutput, List[ViTEncoderOutput]]]: Output data from the encoder.
                If `intermediates_only` is True, returns a list of intermediate features.
                Otherwise, returns a tuple with the final features and a list of intermediate features.
        """
        # Check image normalization type
        self._check_data_normalization_type(encoder_input.data_norm_type)

        # Check the dtype and shape of the input image
        assert isinstance(encoder_input.image, torch.Tensor), "Input must be a torch.Tensor"
        assert encoder_input.image.ndim == 4, "Input must be of shape (B, C, H, W)"
        batch_size, channels, height, width = encoder_input.image.shape
        assert channels == 3, "Input must have 3 channels"
        assert (
            height % self.patch_size == 0 and width % self.patch_size == 0
        ), f"Input shape must be divisible by patch size: {self.patch_size}"

        # Extract the final features and intermediate features accordingly
        model_outputs = self.model.forward_intermediates(
            encoder_input.image,
            indices=self.indices,
            return_prefix_tokens=False,
            norm=self.norm_intermediate,
            stop_early=self.stop_early,
            output_fmt="NCHW",
            intermediates_only=self.intermediates_only,
        )

        if self.intermediates_only:
            outputs = model_outputs
            final_features = None
        else:
            final_output, outputs = model_outputs
            final_features = final_output.features
            final_features = final_features.reshape(
                batch_size, -1, self.enc_embed_dim, height // self.patch_size, width // self.patch_size
            ).contiguous()
            final_features = ViTEncoderOutput(features=final_features)

        outputs = [FeatureWrapper(o) for o in outputs]
        intermediate_features = [
            ViTEncoderOutput(features=intermediate_output.features) for intermediate_output in outputs
        ]

        if self.intermediates_only:
            return intermediate_features
        else:
            return final_features, intermediate_features


if __name__ == "__main__":
    # Init different versions of the RADIO Encoder
    for model_version in ["radio_v2.5-b", "radio_v2.5-l"]:
        naradio_encoder = NARADIOEncoder(name="RADIOv2.5", model_version=model_version)

    print("All NARADIO Encoders have been initialized successfully!")

    # Intermediate Feature Returner Tests
    print("Running Intermediate Feature Returner Tests...")

    # Run the intermediate feature returner with last-n index
    naradio_intermediate_feature_returner = NARADIOIntermediateFeatureReturner(
        name="RADIOv2.5", model_version="radio_v2.5-b", indices=6
    )  # Last 6 layers
    dummy_input = ViTEncoderInput(image=torch.randn(1, 3, 224, 224), data_norm_type="radio")
    output = naradio_intermediate_feature_returner(dummy_input)
    assert isinstance(output, list), "Output must be a list of intermediate features"
    assert isinstance(output[0], ViTEncoderOutput), "Output must be a list of ViTEncoderOutput"
    assert len(output) == 6, "Output must have length of intermediate features equal to the number of indices"

    # Run the intermediate feature returner with specific indices
    naradio_intermediate_feature_returner = NARADIOIntermediateFeatureReturner(
        name="RADIOv2.5", model_version="radio_v2.5-b", indices=[0, 2, 4, 6]
    )  # Specific layers
    dummy_input = ViTEncoderInput(image=torch.randn(1, 3, 224, 224), data_norm_type="radio")
    output = naradio_intermediate_feature_returner(dummy_input)
    assert isinstance(output, list), "Output must be a list of intermediate features"
    assert isinstance(output[0], ViTEncoderOutput), "Output must be a list of ViTEncoderOutput"
    assert len(output) == 4, "Output must have length of intermediate features equal to the number of indices"

    # Test the normalizing of intermediate features
    naradio_intermediate_feature_returner = NARADIOIntermediateFeatureReturner(
        name="RADIOv2.5", model_version="radio_v2.5-b", norm_intermediate=False, intermediates_only=False
    )  # Do not normalize
    dummy_input = ViTEncoderInput(image=torch.randn(1, 3, 224, 224), data_norm_type="radio")
    output = naradio_intermediate_feature_returner(dummy_input)
    assert isinstance(output, tuple), "Output must be a tuple with final features and intermediate features"
    assert isinstance(output[0], ViTEncoderOutput), "First element of output must be the final features"
    assert isinstance(output[1], list), "Second element of output must be a list of intermediate features"
    assert isinstance(output[1][0], ViTEncoderOutput), "Output must be a list of ViTEncoderOutput"
    if not isinstance(naradio_intermediate_feature_returner.model.model.norm, torch.nn.Identity):
        assert not torch.equal(
            output[0].features, output[1][0].features
        ), "Final features and intermediate features must be different"

    naradio_intermediate_feature_returner = NARADIOIntermediateFeatureReturner(
        name="RADIOv2.5", model_version="radio_v2.5-b", norm_intermediate=True, intermediates_only=False
    )
    dummy_input = ViTEncoderInput(image=torch.randn(1, 3, 224, 224), data_norm_type="radio")
    output = naradio_intermediate_feature_returner(dummy_input)
    assert isinstance(output, tuple), "Output must be a tuple with final features and intermediate features"
    assert isinstance(output[0], ViTEncoderOutput), "First element of output must be the final features"
    assert isinstance(output[1], list), "Second element of output must be a list of intermediate features"
    assert isinstance(output[1][0], ViTEncoderOutput), "Output must be a list of ViTEncoderOutput"

    print("All Intermediate Feature Returner Tests have passed successfully!")
