"""
Encoder Class for Cosmos
"""

import torch

from uniception.models.encoders.base import UniCeptionViTEncoderBase, ViTEncoderInput, ViTEncoderOutput
from uniception.models.libs.cosmos_tokenizer.modules import ContinuousFormulation, EncoderType
from uniception.models.libs.cosmos_tokenizer.networks import TokenizerConfigs


class CosmosEncoder(UniCeptionViTEncoderBase):
    "Uniception Cosmos Encoder"

    def __init__(
        self,
        name: str,
        data_norm_type: str = "cosmos",
        patch_size: int = 8,
        pretrained_checkpoint_path: str = None,
        *args,
        **kwargs,
    ):
        """
        Cosmos Encoder for extracting spatial features from images.

        Args:
            name (str): Name of the encoder.
            data_norm_type (str): Image normalization type. Default: "cosmos"
            patch_size (int): Patch size for the encoder. Default: 8
            pretrained_checkpoint_path (str): Path to the pretrained checkpoint. Default: None
        """
        # Init the base class
        super().__init__(name=name, data_norm_type=data_norm_type, patch_size=patch_size, *args, **kwargs)

        # Init Cosmos Encoder sepecific attributes
        tokenizer_config = TokenizerConfigs["CI"].value.copy()
        tokenizer_config.update(dict(spatial_compression=self.patch_size))

        z_factor = tokenizer_config["z_factor"]
        z_channels = tokenizer_config["z_channels"]
        latent_channels = tokenizer_config["latent_channels"]
        encoder_name = kwargs.get("encoder", EncoderType.Default.name)
        print(tokenizer_config)
        del tokenizer_config["z_factor"]
        del tokenizer_config["z_channels"]
        del tokenizer_config["latent_channels"]
        self.encoder = EncoderType[encoder_name].value(z_channels=z_factor * z_channels, **tokenizer_config)
        self.quant_conv = torch.nn.Conv2d(z_factor * z_channels, z_factor * latent_channels, 1)
        formulation_name = kwargs.get("formulation", ContinuousFormulation.AE.name)
        self.distribution = ContinuousFormulation[formulation_name].value()

        # Load the pretrained checkpoint
        if pretrained_checkpoint_path is not None:
            print(f"Loading custom pretrained Cosmos checkpoint from {pretrained_checkpoint_path}")
            ckpt = torch.load(pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"]))

    def encode(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor]:
        """Encodes an image into a latent embedding or code.

        Args:
            input_tensor: The input tensor Bx3xHxW layout, range [-1..1].
        Returns:
            For continuous image (CI) tokenizer, the tuple contains:
                - The latent embedding, Bx16x(h)x(w), where the compression
                rate is (H/h x W/w), and channel dimension of 16.
            For discrete image (DI) tokenizer, the tuple contains:
                - The indices, Bx(h)x(w), from a codebook of size 64K, which
                corresponds to FSQ levels of (8,8,8,5,5,5).
               - The discrete code, Bx6x(h)x(w), where the compression rate is
                again (H/h x W/w), and channel dimension of 6.
        """
        x = self.encoder(input_tensor)
        x = self.quant_conv(x)
        output_latent = self.distribution(x)

        if isinstance(output_latent, torch.Tensor):
            return output_latent
        return output_latent[:-1]

    def forward(self, encoder_input: ViTEncoderInput) -> ViTEncoderOutput:
        """
        Cosmos Encoder Forward Pass

        Args:
            encoder_input (ViTEncoderInput): Input data for the encoder. Input data must contain image normalization type and normalized image tensor.

        Returns:
            ViTEncoderOutput: Output data from the encoder.
        """
        # Check image normalization type
        self._check_data_normalization_type(encoder_input.data_norm_type)

        # Check the dtype and shape of the input image
        assert isinstance(encoder_input.image, torch.Tensor), "Input must be a torch.Tensor"
        assert encoder_input.image.ndim == 4, "Input must be of shape (B, C, H, W)"
        batch_size, channels, height, width = encoder_input.image.shape
        assert channels == 3, "Input must have 3 channels"
        assert (
            height % self.patch_size == 0 and width % self.patch_size == 0
        ), f"Input shape must be divisible by patch size: {self.patch_size}"

        # Extract the features from the DINOv2 model
        features = self.encode(encoder_input.image)[0].contiguous()

        return ViTEncoderOutput(features=features)


if __name__ == "__main__":

    # initialize different variants of the Cosmos Encoder, untrained
    for is_continuous in [True]:
        for patch_size in [8, 16]:
            encoder = CosmosEncoder(name="cosmos", patch_size=patch_size)

    # # initialize from trained checkpoint, with/without jit inference capability
    PRETRAINED_JIT_CHECKPOINTS = {
        ("CI", 8): "../../../checkpoints/encoders/cosmos/Cosmos-Tokenizer-CI8x8/encoder.pth",
        ("CI", 16): "../../../checkpoints/encoders/cosmos/Cosmos-Tokenizer-CI16x16/encoder.pth",
    }

    for patch_size in [8, 16]:

        encoder = CosmosEncoder(
            name="cosmos",
            patch_size=patch_size,
            pretrained_checkpoint_path=PRETRAINED_JIT_CHECKPOINTS[("CI", patch_size)],
        )

    # example inference
    dummy_image = torch.randn(1, 3, 256, 256).cuda()

    encoder_input = ViTEncoderInput(data_norm_type="cosmos", image=dummy_image)

    encoder = encoder.cuda()
    encoder_output = encoder(encoder_input)
