"""
Cosmos Decoder head implementation
Downstream heads assume inputs of size BCHW (B: batch, C: channels, H: height, W: width);
"""

import torch
import torch.nn as nn

from uniception.models.libs.cosmos_tokenizer.modules import DecoderType
from uniception.models.libs.cosmos_tokenizer.networks import TokenizerConfigs
from uniception.models.prediction_heads.adaptors import (
    Covariance2DAdaptor,
    FlowAdaptor,
    FlowWithConfidenceAdaptor,
    MaskAdaptor,
)
from uniception.models.prediction_heads.base import PixelTaskOutput, PredictionHeadInput

COSMOS_LATENT_CHANNELS = 16

CLASSNAME_TO_ADAPTOR_CLASS = {
    "FlowAdaptor": FlowAdaptor,
    "FlowWithConfidenceAdaptor": FlowWithConfidenceAdaptor,
    "Covariance2DAdaptor": Covariance2DAdaptor,
    "MaskAdaptor": MaskAdaptor,
}


class CosmosSingleChannel(nn.Module):
    """
    This class implements a single cosmos decoder. This decoder takes features and produce
    a single channel output in the range of [-1, 1] (not strictly enforced).
    """

    def __init__(
        self,
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
        """

        super().__init__(*args, **kwargs)

        self.patch_size = patch_size

        assert self.patch_size in [8, 16], f"Invalid patch size: {self.patch_size}"

        # Init Cosmos Encoder sepecific attributes
        tokenizer_config = TokenizerConfigs["CI"].value.copy()
        tokenizer_config.update(dict(spatial_compression=self.patch_size))

        z_channels = tokenizer_config["z_channels"]
        latent_channels = tokenizer_config["latent_channels"]
        del tokenizer_config["z_channels"]
        del tokenizer_config["latent_channels"]

        decoder_name = tokenizer_config.get("decoder", DecoderType.Default.name)
        self.decoder = DecoderType[decoder_name].value(z_channels=z_channels, **tokenizer_config)

        self.post_quant_conv = torch.nn.Conv2d(latent_channels, z_channels, 1)

        if pretrained_checkpoint_path is not None:
            print(f"Loading pretrained cosmos decoder from {pretrained_checkpoint_path}")
            ckpt = torch.load(pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"]))

    def forward(self, x: torch.Tensor):
        """
        Forward interface for the linear feature mapping.

        Args:
            x : torch.Tensor, the input features

        Returns:
            torch.Tensor, the output of the linear feature mapping
        """

        x = self.post_quant_conv(x)
        x = self.decoder(x)

        return x


class CosmosFeature(nn.Module):
    """
    This class implements a linear mapping from the low resolution patch features
    to pixel-wise features.
    """

    def __init__(
        self,
        input_feature_dim: int,
        output_dim: int,
        patch_size: int,
        skip_linear: bool = False,
        single_channel_ckpt: str = None,
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
        self.skip_linear = skip_linear
        self.pretrained_checkpoint_path = pretrained_checkpoint_path

        assert self.patch_size in [8, 16], f"Invalid patch size: {self.patch_size}"

        if not self.skip_linear:
            self.linear = nn.Conv2d(
                in_channels=self.input_feature_dim,
                out_channels=self.output_dim * COSMOS_LATENT_CHANNELS,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )

        self.cosmos_decoders = nn.ModuleList(
            [
                CosmosSingleChannel(
                    patch_size=self.patch_size,
                    pretrained_checkpoint_path=single_channel_ckpt,
                    *args,
                    **kwargs,
                )
                for _ in range(self.output_dim)
            ]
        )

        self.output_scaling = nn.Parameter(torch.ones(1, self.output_dim, 1, 1))
        self.output_bias = nn.Parameter(torch.zeros(1, self.output_dim, 1, 1))

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

        if not self.skip_linear:
            x = self.linear(x)

        x_split = list(torch.split(x, COSMOS_LATENT_CHANNELS, dim=1))

        output = [None] * self.output_dim
        for i, decoder in enumerate(self.cosmos_decoders):
            output[i] = torch.mean(decoder(x_split[i]), dim=1, keepdim=True)

        # Concatenate the decoded channels
        x = torch.cat(output, dim=1)

        # a linear scaling layer to map cosmos output [-1, 1] to arbitrary range
        x = x * self.output_scaling + self.output_bias

        return PixelTaskOutput(decoded_channels=x), x_split


if __name__ == "__main__":

    x_single_channel = torch.randn(1, 16, 8, 8)

    # Test CosmosSingleChannel
    cosmos_single_channel = CosmosSingleChannel(patch_size=8)
    cosmos_single_channel(x_single_channel)

    # Test CosmosFeature
    cosmos_feature = CosmosFeature(input_feature_dim=1024, output_dim=2, patch_size=8)
    x_feature = torch.randn(1, 1024, 8, 8)

    output = cosmos_feature(PredictionHeadInput(last_feature=x_feature))
    print(output.decoded_channels.shape)
