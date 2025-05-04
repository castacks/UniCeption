"""
Global quantity prediction head implementation
Downstream heads assume inputs of size BCHW (B: batch, C: channels, H: height, W: width)
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from uniception.models.prediction_heads.base import PredictionHeadInput, SummaryTaskOutput
from uniception.models.prediction_heads.pose_head import ResConvBlock


class GlobalHead(nn.Module):
    """
    Glboal quantity regression head implementation
    """

    def __init__(
        self,
        patch_size: int,
        input_feature_dim: int,
        num_resconv_block: int = 2,
        output_representation_dim: int = 1,
        pretrained_checkpoint_path: str = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the global head.

        Args:
            patch_size : int, the patch size of the transformer used to generate the input features
            input_feature_dim : int, the input feature dimension
            num_resconv_block : int, the number of residual convolution blocks
            output_representation_dim : int, the dimension of the output representation
            pretrained_checkpoint_path : str, path to pretrained checkpoint (default: None)
        """
        super().__init__()
        self.patch_size = patch_size
        self.input_feature_dim = input_feature_dim
        self.num_resconv_block = num_resconv_block
        self.output_representation_dim = output_representation_dim
        self.pretrained_checkpoint_path = pretrained_checkpoint_path

        # Initialize the hidden dimension of the global head based on the patch size
        self.output_dim = 4 * (self.patch_size**2)

        # Initialize the projection layer for the hidden dimension of the global head
        self.proj = nn.Conv2d(
            in_channels=self.input_feature_dim,
            out_channels=self.output_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # Initialize sequential layers of the global head
        self.res_conv = nn.ModuleList(
            [copy.deepcopy(ResConvBlock(self.output_dim, self.output_dim)) for _ in range(self.num_resconv_block)]
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.more_mlps = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
        )
        self.fc_output = nn.Linear(self.output_dim, self.output_representation_dim)

        # Load the pretrained checkpoint if provided
        if self.pretrained_checkpoint_path is not None:
            print(f"Loading pretrained global head from {self.pretrained_checkpoint_path}")
            ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"]))

    def forward(self, feature_input: PredictionHeadInput):
        """
        Forward interface for the global quantity prediction head.
        The head requires an adapter on the final output.

        Args:
            feature_input : PredictionHeadInput, the input features
            - last_feature : torch.Tensor, the last feature tensor

        Returns:
            SummaryTaskOutput, the output of the global head
            - decoded_channels : torch.Tensor, the decoded channels
        """
        # Get the patch-level features from the input
        feat = feature_input.last_feature  # (B, C, H, W)

        # Check the input dimensions
        assert (
            feat.shape[1] == self.input_feature_dim
        ), f"Input feature dimension {feat.shape[1]} does not match expected dimension {self.input_feature_dim}"

        # Apply the projection layer to the patch-level features
        feat = self.proj(feat)  # (B, PC, H, W)

        # Apply the residual convolution blocks to the projected features
        for i in range(self.num_resconv_block):
            feat = self.res_conv[i](feat)

        # Apply the average pooling layer to the residual convolution output
        feat = self.avgpool(feat)  # (B, PC, 1, 1)

        # Flatten the average pooled features
        feat = feat.view(feat.size(0), -1)  # (B, PC)

        # Apply the more MLPs to the flattened features
        feat = self.more_mlps(feat)  # (B, PC)

        # Apply the final linear layers to the more MLPs output
        output_feat = self.fc_output(feat)  # (B, self.output_representation_dim)

        return SummaryTaskOutput(decoded_channels=output_feat)


if __name__ == "__main__":
    # Init an example global head
    global_head = GlobalHead(
        patch_size=14,
        input_feature_dim=1024,
        num_resconv_block=2,
        output_representation_dim=1,
        pretrained_checkpoint_path=None,
    )

    # Create a dummy input tensor with shape (B, C, H, W)
    dummy_input = torch.randn(4, 1024, 14, 14)  # Example input

    # Run dummy forward pass
    output = global_head(PredictionHeadInput(last_feature=dummy_input))

    # Check the output shape
    assert output.decoded_channels.shape == (4, 1), "Output shape mismatch"

    print("Global head test passed!")
