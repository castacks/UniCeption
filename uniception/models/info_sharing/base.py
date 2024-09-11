"""
Base Information Sharing Class for UniCeption
"""

import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class InfoSharingInput:
    pass


@dataclass
class InfoSharingOutput:
    pass


class UniCeptionInfoSharingBase(nn.Module):
    def __init__(
        self,
        name: str,
        size: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Base class for all models in UniCeption.
        """
        super().__init__(*args, **kwargs)

        self.name: str = name
        self.size: Optional[str] = size

    def forward(
        self,
        model_input: InfoSharingInput,
    ) -> InfoSharingOutput:
        """
        Forward interface for the UniCeption information sharing models.

        Args:
            model_input (InfoSharingInput): Input to the model.
                This is also includes the other fields that are required by the specific implementation of the model.

        Returns:
            InfoSharingOutput: Output of the model.
        """

        raise NotImplementedError


class IntermediateFeatureReturner:
    def __init__(self, total_num_layers: int, selected_layers: List[int]):
        """
        Class to return intermediate features from the encoder.
        """
        self.total_num_layers: int = total_num_layers
        self.selected_layers: List[int] = selected_layers
        self.num_intermediate_layers: int = len(selected_layers)


if __name__ == "__main__":
    dummy_model = UniCeptionInfoSharingBase(name="dummy")
    print("Dummy Base InfoSharing model created successfully!")
