"""
Base Encoder Class for UniCeption
"""

import torch.nn as nn
from jaxtyping import Float
from dataclasses import dataclass
from torch import Tensor
from typing import List, Optional


@dataclass
class EncoderInput:
    data_norm_type: str
    # Add other fields that are required by the specific implementation of the encoder.


@dataclass
class EncoderOutput:
    pass


class UniCeptionEncoderBase(nn.Module):
    def __init__(
        self,
        name: str,
        data_norm_type: str,
        size: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Base class for all encoders in UniCeption.
        """
        super().__init__(*args, **kwargs)

        self.name: str = name
        self.size: Optional[str] = size

        self.data_norm_type: str = data_norm_type

    def forward(
        self,
        encoder_input: EncoderInput,
    ) -> EncoderOutput:
        """
        Forward interface for the UniCeption encoders.

        We expect the "data_norm_type" field to be present in the encoder_input to check for normalization type.

        Args:
            encoder_input (EncoderInput): Input to the encoder. We expect the following fields: "data_norm_type: str".
                This is also includes the other fields that are required by the specific implementation of the encoder.

        Returns:
            EncoderOutput: Output of the encoder.
        """

        raise NotImplementedError

    def _check_data_normalization_type(self, data_norm_type: str):
        """
        Check if the input normalization type matches the encoder's expected input data normalization type.

        Args:
            data_norm_type (str): Data normalization type.

        Raises:
            AssertionError: If the data normalization type does not match the encoder's expected input data normalization type.
        """

        assert (
            data_norm_type == self.data_norm_type
        ), f"Input normalization type {data_norm_type} does not match the encoder's normalization type {self.data_norm_type}."


@dataclass
class ViTEncoderInput(EncoderInput):
    image: Float[Tensor, "batch channel height width"]


@dataclass
class ViTEncoderOutput(EncoderOutput):
    features: Float[Tensor, "batch enc_embed_dim feat_height feat_width"]


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
        super().__init__(*args, **kwargs)

        self.patch_size = patch_size


class IntermediateFeatureReturner:
    def __init__(self, total_num_layers: int, selected_layers: List[int]):
        """
        Class to return intermediate features from the encoder.
        """
        self.total_num_layers: int = total_num_layers
        self.selected_layers: List[int] = selected_layers
        self.num_intermediate_layers: int = len(selected_layers)


if __name__ == "__main__":
    dummy_model = UniCeptionEncoderBase(name="name", data_norm_type="norm")
    dummy_vit_model = UniCeptionViTEncoderBase(name="name", data_norm_type="norm", patch_size=16)
    print("Dummy Base Encoders created successfully!")
