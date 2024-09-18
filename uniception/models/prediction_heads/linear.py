"""
Linear head implementation
Downstream heads assume inputs of size BCHW (B: batch, C: channels, H: height, W: width);
The linear head implementation is based on DUSt3R and CroCoV2
References: https://github.com/naver/dust3r
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from uniception.models.prediction_heads.base import PixelTaskOutput, PredictionHeadInput


class LinearFeature(nn.Module):
    """
    This class implements a linear mapping from the low resolution patch features
    to pixel-wise features.
    """

    def __init__(
        self,
        input_feature_dim: int,
        output_dim: int,
        patch_size: int,
        pretrained_checkpoint_path: str = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the linear feature mapping.

        Args:
            input_feature_dim : int, the input feature dimension
            output_dim : int, the output feature dimension
            patch_size : int, the patch size
            pretrained_checkpoint_path : str, path to pretrained checkpoint (default: None)
        """

        super().__init__(*args, **kwargs)

        self.input_feature_dim = input_feature_dim
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.pretrained_checkpoint_path = pretrained_checkpoint_path

        self.linear = nn.Conv2d(
            in_channels=self.input_feature_dim,
            out_channels=self.output_dim * (self.patch_size**2),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        if self.pretrained_checkpoint_path is not None:
            print(f"Loading pretrained linear dense feature head from {self.pretrained_checkpoint_path}")
            ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"]))

    def forward(self, feature_input: PredictionHeadInput):
        """
        Forward interface for the linear feature mapping.

        Args:
            feature_input : PredictionHeadInput, the input features
            - last_feature : torch.Tensor, the last feature tensor

        Returns:
            PixelTaskOutput, the output of the linear feature mapping
            - decoded_channels : torch.Tensor, the decoded channels

        """

        x = feature_input.last_feature

        assert (
            x.shape[1] == self.input_feature_dim
        ), f"Input feature dimension mismatch: {x.shape[1]} != {self.input_feature_dim}"

        x = self.linear(x)
        x = F.pixel_shuffle(x, self.patch_size)

        return PixelTaskOutput(decoded_channels=x)


if __name__ == "__main__":
    # Init an example linear feature head
    linear_prediction_head = LinearFeature(input_feature_dim=768, output_dim=4, patch_size=16)

    # Create a dummy input tensor with shape (B, C, H, W)
    dummy_input = torch.randn(1, 768, 14, 14)  # Example input

    # Run dummy forward pass
    output = linear_prediction_head(PredictionHeadInput(last_feature=dummy_input))
