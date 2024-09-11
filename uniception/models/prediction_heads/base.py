"""
Base Prediction Head Class for UniCeption
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple


@dataclass
class PredictionHeadInput:
    pass


@dataclass
class AdaptorInput:
    adaptor_feature: torch.Tensor
    output_shape_hw: Tuple[int, int]


@dataclass
class AdaptorOutput:
    pass


@dataclass
class PredictionHeadOutput:
    adaptor_output: Dict[str, AdaptorOutput]


@dataclass
class MaskAdaptorOutput:
    logits: torch.Tensor
    mask: torch.Tensor


@dataclass
class RegressionAdaptorOutput:
    value: torch.Tensor


@dataclass
class RegressionWithConfidenceAdaptorOutput:
    value: torch.Tensor
    confidence: torch.Tensor


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
