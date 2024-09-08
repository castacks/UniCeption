# --------------------------------------------------------
# Encoder Class for RADIO (Nvidia)
# --------------------------------------------------------
import torch
from typing import Optional

from uniception.models.encoders.base import UniCeptionViTEncoderBase, ViTEncoderInput, ViTEncoderOutput


class RADIOEncoder(UniCeptionViTEncoderBase):
    def __init__(
        self,
        name: str,
        data_norm_type: str = "radio",
        patch_size: int = 16,
        model_version: str = "radio_v2.5-l",
        pretrained_checkpoint_path: str = None,
        eradio_input_shape: Optional[tuple] = None,
        *args,
        **kwargs,
    ):
        """
        RADIO Encoder for extracting spatial features from images.

        Args:
            name (str): Name of the encoder.
            data_norm_type (str): Image normalization type. Default: "radio"
            patch_size (int): Patch size for the encoder. Default: 16
            model_version (str): Version of the RADIO model to load. Default: "radio_v2.5-l"
            pretrained_checkpoint_path (str): Path to the pretrained checkpoint if using custom trained version of RADIO.
            eradio_input_shape (tuple): Input shape (height, width) for E-RADIO models. Default: None
        """
        # Init the base class
        super().__init__(
            name=name,
            data_norm_type=data_norm_type,
            patch_size=patch_size,
            *args,
            **kwargs,
        )

        # Init the RADIO Encoder specific attributes
        self.model_version = model_version
        self.enc_embed_dim = {"radio_v2.5-b": 768, "radio_v2.5-l": 1024, "e-radio_v2": 1536}[self.model_version]

        # Load the pretrained RADIO model from torch hub
        print(f"Loading pretrained {self.model_version} from torch hub")
        try:  # Requires internet access
            self.model = torch.hub.load(
                "NVlabs/RADIO",
                "radio_model",
                version=self.model_version,
                progress=True,
                skip_validation=True,
                force_reload=True,
            )
        except:  # Load from cache
            self.model = torch.hub.load(
                "NVlabs/RADIO",
                "radio_model",
                version=self.model_version,
                progress=True,
                skip_validation=True,
            )

        # Set the optimal window size for E-RADIO models
        if "e-radio" in self.model_version:
            assert eradio_input_shape is not None, "Input shape (height, width) must be provided for E-RADIO models"
            self.model.model.set_optimal_window_size(eradio_input_shape)

        # Load the custom pretrained checkpoint if provided
        if pretrained_checkpoint_path is not None:
            print(f"Loading custom pretrained RADIO checkpoint from {pretrained_checkpoint_path}")
            ckpt = torch.load(pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"]))

    def forward(self, encoder_input: ViTEncoderInput) -> ViTEncoderOutput:
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

        # Forward pass throught the RADIO encoder
        summary, features = self.model(encoder_input.image)

        # Resize the features to the expected shape
        # (B x Num_patches x Embed_dim) -> (B x Embed_dim x H / Patch_Size x W / Patch_Size)
        features = features.permute(0, 2, 1)
        features = features.reshape(-1, self.enc_embed_dim, height // self.patch_size, width // self.patch_size)

        return ViTEncoderOutput(features=features)


if __name__ == "__main__":
    # Init different versions of the RADIO Encoder
    for model_version in ["radio_v2.5-b", "radio_v2.5-l"]:
        radio_encoder = RADIOEncoder(name="RADIOv2.5", model_version=model_version)

    # Init the E-RADIO Encoder
    eradio_input_shape = (512, 512)
    eradio_encoder = RADIOEncoder(name="E-RADIO", model_version="e-radio_v2", eradio_input_shape=eradio_input_shape)

    print("All RADIO Encoders have been initialized successfully!")
