# --------------------------------------------------------
# Base Encoder Class for UniCeption
# --------------------------------------------------------
import torch.nn as nn
from typing import List, Optional

from .image_normalizations import *


class UniCeptionEncoderBase(nn.Module):
    def __init__(
        name: str,
        data_norm_type: str,
        size: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Base class for all encoders in UniCeption.
        """
        super(UniCeptionEncoderBase, self).__init__(*args, **kwargs)

        self.name: str = name
        self.size: Optional[str] = size

        self.data_norm_type: str = data_norm_type

    def forward(
        self,
        encoder_input: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Forward interface for the UniCeption encoders.

        We expect the "data_norm_type" field to be present in the encoder_input to check for normalization type.

        Args:
            batch (Dict[str, Any]): Input to the encoder. We expect the following fields: "data_norm_type: str".
                This is also includes the other fields that are required by the specific implementation of the encoder.

        Returns:
            Dict[str, Any]: Output of the encoder. This includes the output of the encoder.
        """

        raise NotImplementedError

    def _check_image_normalization_type(self, data_norm_type: str):
        """
        Check if the image normalization type matches the encoder's image normalization type.
        of the data

        Args:
            data_norm_type (str): Image normalization type.

        Raises:
            AssertionError: If the image normalization type does not match the encoder's image normalization.
        """

        assert (
            data_norm_type == self.data_norm_type
        ), f"Input normalization type {data_norm_type} does not match the encoder's normalization type {self.data_norm_type}."


class UniCeptionViTEncoderBase(UniCeptionEncoderBase):
    def __init__(
        self,
        patch_size: int,
        *args,
        **kwargs,
    ):
        """
        Base class for all Vision Transformer encoders in UniCeption.
        """
        super(UniCeptionEncoderBase, self).__init__(*args, **kwargs)

        self.patch_size = patch_size


class IntermediateFeatureReturner():
    
    def __init__(self, num_intermediate_layers : int, selected_layers : List[int]):
        """
        Class to return intermediate features from the encoder.
        """
        self.num_intermediate_layers: int = num_intermediate_layers
        self.selected_layers : List[int] = selected_layers