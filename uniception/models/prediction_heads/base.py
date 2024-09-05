# --------------------------------------------------------
# Base Prediction Head Class for UniCeption
# --------------------------------------------------------
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PredictionHeadInput:
    pass


@dataclass
class PredictionHeadOutput:
    pass


class UniCeptionPredictionHeadBase(nn.Module):
    def __init__(
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
