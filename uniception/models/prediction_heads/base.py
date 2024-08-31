# --------------------------------------------------------
# Base Prediction Head Class for UniCeption
# --------------------------------------------------------
import torch.nn as nn
from typing import List, Optional

from .image_normalizations import *


class UniCeptionPredictionHeadBase(nn.Module):
    def __init__(
        name: str,
        *args,
        **kwargs,
    ):
        """
        Base class for all prediction heads in UniCeption.
        """
        super(UniCeptionPredictionHeadBase, self).__init__(*args, **kwargs)

        self.name: str = name

    def forward(
        self,
        head_input: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Forward interface for the UniCeption prediction heads.


        Args:
            batch (Dict[str, Any]): Input to the prediction head. 

        Returns:
            Dict[str, Any]: Output of the prediction head.
        """

        raise NotImplementedError

