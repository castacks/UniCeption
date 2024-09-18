"""
Base Prediction Head Class for UniCeption
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


@dataclass
class PredictionHeadInput:
    last_feature: Float[Tensor, "batch_size feat_dim feat_height feat_width"]


@dataclass
class PredictionHeadLayeredInput:
    list_features: List[Float[Tensor, "batch_size feat_dim feat_height feat_width"]]
    target_output_shape: Tuple[int, int]


@dataclass
class PixelTaskOutput:
    """
    PixelTaskOutput have dense pixel-wise output in BCHW format,
    with the same spatial resolution as the input image.
    """

    decoded_channels: Float[Tensor, "batch_size output_channels height width"]


@dataclass
class AdaptorInput:
    adaptor_feature: Float[Tensor, "batch_size sliced_channels height width"]
    output_shape_hw: Tuple[int, int]


@dataclass
class AdaptorOutput:
    pass


@dataclass
class PredictionHeadOutput:
    adaptor_output: Dict[str, AdaptorOutput]


@dataclass
class MaskAdaptorOutput:
    logits: Float[Tensor, "batch_size 1 height width"]
    mask: Float[Tensor, "batch_size 1 height width"]


@dataclass
class RegressionAdaptorOutput:
    value: Float[Tensor, "batch_size sliced_channels height width"]


@dataclass
class RegressionWithConfidenceAdaptorOutput:
    value: Float[Tensor, "batch_size sliced_channels height width"]
    confidence: Float[Tensor, "batch_size 1 height width"]


class UniCeptionPredictionHeadBase(nn.Module):
    def __init__(
        self,
        name: str,
        *args,
        **kwargs,
    ):
        """
        Base class for all prediction heads in UniCeption.
        """
        super().__init__(*args, **kwargs)

        self.name: str = name

    def forward(
        self,
        head_input: PredictionHeadInput,
    ) -> PredictionHeadOutput:
        """
        Forward interface for the UniCeption prediction heads.


        Args:
            head_input (PredictionHeadInput): Input to the prediction head.

        Returns:
            head_output (PredictionHeadOutput): Output of the prediction head.
        """

        raise NotImplementedError


class UniCeptionAdaptorBase(nn.Module):
    def __init__(
        self,
        name: str,
        required_channels: int,
        *args,
        **kwargs,
    ):
        """
        Base class for all adaptors in UniCeption.
        """
        super().__init__(*args, **kwargs)

        self.name: str = name
        self.required_channels: int = required_channels

    def forward(
        self,
        adaptor_input: AdaptorInput,
    ) -> AdaptorOutput:
        """
        Forward interface for the UniCeption adaptors.


        Args:
            adaptor_input (AdaptorInput): Input to the adaptor.

        Returns:
            adaptor_output (AdaptorOutput): Output of the adaptor.
        """

        raise NotImplementedError
