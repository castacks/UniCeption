"""
Linear head implementation
Downstream heads assume inputs of size BCHW (B: batch, C: channels, H: height, W: width);
The DPT head implementation is based on DUSt3R and CroCoV2
References: https://github.com/naver/dust3r
"""

from dataclasses import dataclass
from turtle import forward
from typing import Iterable, List, Optional, Tuple, Union

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
    ):
        """
        Initialize the linear feature mapping.

        Args:
            input_feature_dim : int, the input feature dimension
            output_dim : int, the output feature dimension
            patch_size : int, the patch size
        """

        super().__init__()

        self.input_feature_dim = input_feature_dim
        self.output_dim = output_dim
        self.patch_size = patch_size

        self.linear = nn.Conv2d(
            in_channels=self.input_feature_dim,
            out_channels=self.output_dim * (self.patch_size**2),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

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
