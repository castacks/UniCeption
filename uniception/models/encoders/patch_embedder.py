"""
Encoder class for Patch Embedder
"""

import math
from functools import partial
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from uniception.models.encoders.base import (
    UniCeptionViTEncoderBase,
    ViTEncoderInput,
    ViTEncoderNonImageInput,
    ViTEncoderOutput,
)


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbedder(UniCeptionViTEncoderBase):
    "UniCeption Patch Embedder"

    def __init__(
        self,
        name: str,
        data_norm_type: str = "patch_embedder",
        input_size: Union[int, Tuple[int, int]] = 518,
        patch_size: int = 14,
        in_chans: int = 3,
        enc_embed_dim: int = 1024,
        norm_layer: Optional[Callable] = None,
        post_pe_norm_layer: Optional[Callable] = partial(nn.LayerNorm, eps=1e-6),
        interpolate_antialias: bool = False,
        interpolate_offset: float = 0.1,
        pretrained_checkpoint_path: str = None,
        *args,
        **kwargs,
    ):
        """
        Patch Encoder for extracting patch-wise features from a spatial input of size (B, C, H, W).
        Learnable positional encoding is also applied to the patch-wise features.
        """
        # Init the base class
        super().__init__(
            name=name,
            data_norm_type=data_norm_type,
            patch_size=patch_size,
            *args,
            **kwargs,
        )

        # Init the Patch Embedder specific attributes
        patch_HW = make_2tuple(patch_size)
        self.input_size = make_2tuple(input_size)
        self.patches_resolution = (self.input_size[0] // patch_HW[0], self.input_size[1] // patch_HW[1])
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.enc_embed_dim = enc_embed_dim

        # Init the Patch Embedder layers
        self.proj = nn.Conv2d(in_chans, enc_embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(enc_embed_dim) if norm_layer else nn.Identity()

        # Init the learnable positional encodings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.enc_embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        # Init the norm layer after positional encoding
        self.post_pe_norm = post_pe_norm_layer(enc_embed_dim) if post_pe_norm_layer else nn.Identity()

        # Load the pretrained checkpoint if provided
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        if self.pretrained_checkpoint_path:
            print(f"Loading custom pretrained Patch Embedder checkpoint from {self.pretrained_checkpoint_path} ...")
            ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"]))

    def interpolate_pos_encoding(self, features, height, width):
        """
        Interpolate the positional encoding to the expected size.

        Args:
            features (torch.Tensor): Input tensor of shape (B, N, C).
            height (int, float): Height of the input tensor.
            width (int, float): Width of the input tensor.

        Returns:
            torch.Tensor: Interpolated positional encoding tensor of shape (1, N, C).
        """
        previous_dtype = features.dtype
        npatch = features.shape[1]
        N = self.pos_embed.shape[1]
        if npatch == N and height == width:
            return self.pos_embed
        patch_pos_embed = self.pos_embed.float()
        dim = features.shape[-1]
        height0 = height // self.patch_size
        width0 = width // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sh = float(height0 + self.interpolate_offset) / M
            sw = float(width0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sh, sw)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (height0, width0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (height0, width0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return patch_pos_embed.to(previous_dtype)

    def forward(self, encoder_input: Union[ViTEncoderInput, ViTEncoderNonImageInput]) -> ViTEncoderOutput:
        """
        Patch Embedder Forward Pass

        Args:
            encoder_input (Union[ViTEncoderInput, ViTEncoderNonImageInput]): Input data for the encoder.
                If input type is ViTEncoderInput, input data must contain image normalization type and normalized image tensor.
                If input type is ViTEncoderNonImageInput, input data must contain a tensor of size (B, C, H, W).

        Returns:
            ViTEncoderOutput: Output data from the encoder.
        """
        # Get the input data and verify normalization if the input type is ViTEncoderInput
        if isinstance(encoder_input, ViTEncoderInput):
            self._check_data_normalization_type(encoder_input.data_norm_type)
            input_data = encoder_input.image
        elif isinstance(encoder_input, ViTEncoderNonImageInput):
            input_data = encoder_input.data
        else:
            raise ValueError("Unsupported input type for Patch Embedder.")

        # Check the dtype and shape of the input
        assert isinstance(input_data, torch.Tensor), "Input must be a torch.Tensor"
        assert input_data.ndim == 4, "Input must be of shape (B, C, H, W)"
        batch_size, channels, height, width = input_data.shape
        assert (
            height % self.patch_size == 0 and width % self.patch_size == 0
        ), f"Input shape must be divisible by patch size: {self.patch_size}"

        # Patchify the input data and project into expected latent space
        features = self.proj(input_data)  # (B, C, H, W) -> (B, E, H / Patch_Size, W / Patch_Size)
        features = features.flatten(2).transpose(
            1, 2
        )  # (B, E, H / Patch_Size, W / Patch_Size) -> (B, H / Patch_Size * W / Patch_Size, E)
        features = self.norm(features)  # Normalize the features after patch embedding
        features = features + self.interpolate_pos_encoding(
            features, height, width
        )  # (B, H / Patch_Size * W / Patch_Size, E)
        features = self.post_pe_norm(features)  # Normalize the features after positional encoding

        # Resize the features to the expected shape
        # (B x Num_patches x Embed_dim) -> (B x Embed_dim x H / Patch_Size x W / Patch_Size)
        features = features.permute(0, 2, 1)
        features = features.reshape(
            -1, self.enc_embed_dim, height // self.patch_size, width // self.patch_size
        ).contiguous()

        return ViTEncoderOutput(features=features)


if __name__ == "__main__":
    # Init Patch Embedder for images as input
    patch_embedder = PatchEmbedder(
        name="patch_embedder",
        data_norm_type="patch_embedder",
        input_size=518,
        patch_size=14,
        in_chans=3,
        enc_embed_dim=1024,
    )

    # Test dummy image input
    dummy_image = torch.randn(1, 3, 518, 518)
    patch_embedder_output = patch_embedder(ViTEncoderInput(data_norm_type="patch_embedder", image=dummy_image))
    assert patch_embedder_output.features.shape == (
        1,
        1024,
        37,
        37,
    ), "Output features must have shape (1, 1024, 37, 37)"

    # Init Patch Embedder for non-image data as input
    patch_embedder = PatchEmbedder(
        name="patch_embedder",
        data_norm_type="patch_embedder",
        input_size=518,
        patch_size=14,
        in_chans=6,
        enc_embed_dim=1024,
    )

    # Init Patch Embedder for single channel input
    patch_embedder = PatchEmbedder(
        name="patch_embedder",
        data_norm_type="patch_embedder",
        input_size=518,
        patch_size=14,
        in_chans=1,
        enc_embed_dim=1024,
    )

    # Test dummy non-image input
    dummy_image = torch.randn(1, 1, 518, 518)
    patch_embedder_output = patch_embedder(ViTEncoderNonImageInput(data=dummy_image))
    assert patch_embedder_output.features.shape == (
        1,
        1024,
        37,
        37,
    ), "Output features must have shape (1, 1024, 37, 37)"

    print("All variants of Patch Embedder have been initialized successfully!")
