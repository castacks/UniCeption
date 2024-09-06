# --------------------------------------------------------
# Encoder Class for DINOv2
# --------------------------------------------------------
import torch

from uniception.models.encoders.base import UniCeptionViTEncoderBase, ViTEncoderInput, ViTEncoderOutput


class DINOv2Encoder(UniCeptionViTEncoderBase):
    def __init__(
        self,
        name: str,
        data_norm_type: str = "dinov2",
        patch_size: int = 14,
        size: str = "large",
        with_registers: bool = False,
        pretrained_checkpoint_path: str = None,
        *args,
        **kwargs,
    ):
        """
        DINOv2 Encoder for extracting spatial features from images.

        Args:
            name (str): Name of the encoder.
            data_norm_type (str): Image normalization type. Default: "dinov2"
            patch_size (int): Patch size for the encoder. Default: 14
            size (str): Size variant of the DINOv2 model. Options: ["small", "base", "large", "giant"]
            with_registers (bool): Whether to use the DINOv2 model with registers.
            pretrained_checkpoint_path (str): Path to the pretrained checkpoint if using custom trained version of DINOv2.
        """
        # Init the base class
        name = name if not with_registers else f"{name}_reg"
        super().__init__(
            name=name,
            data_norm_type=data_norm_type,
            patch_size=patch_size,
            *args,
            **kwargs,
        )

        # Init the DINOv2 Encoder specific attributes
        self.version = size
        self.with_registers = with_registers
        self.enc_embed_dim = {"small": 384, "base": 768, "large": 1024, "giant": 1536}[self.version]

        # Define DINOv2 model factory
        DINO_MODELS = {
            # No registers
            False: {
                "small": "dinov2_vits14",
                "base": "dinov2_vitb14",
                "large": "dinov2_vitl14",
                "giant": "dinov2_vitg14",
            },
            # With registers
            True: {
                "small": "dinov2_vits14_reg",
                "base": "dinov2_vitb14_reg",
                "large": "dinov2_vitl14_reg",
                "giant": "dinov2_vitg14_reg",
            },
        }

        # Load the pretrained DINOv2 model from torch hub
        print(f"Loading pretrained {DINO_MODELS[self.with_registers][self.version]} from torch hub")
        self.model = torch.hub.load(
            "facebookresearch/dinov2", DINO_MODELS[self.with_registers][self.version], force_reload=True
        )

        # Load the custom pretrained checkpoint if provided
        if pretrained_checkpoint_path:
            print(f"Loading custom pretrained DINOv2 checkpoint from {pretrained_checkpoint_path}")
            ckpt = torch.load(pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"]))

    def forward(self, encoder_input: ViTEncoderInput) -> ViTEncoderOutput:
        # Check image normalization type
        self._check_image_normalization_type(encoder_input.data_norm_type)

        # Check the dtype and shape of the input image
        assert isinstance(encoder_input.image, torch.Tensor), "Input must be a torch.Tensor"
        assert encoder_input.image.ndim == 4, "Input must be of shape (B, C, H, W)"
        batch_size, channels, height, width = encoder_input.image.shape
        assert channels == 3, "Input must have 3 channels"
        assert (
            height % self.patch_size == 0 and width % self.patch_size == 0
        ), f"Input shape must be divisible by patch size: {self.patch_size}"

        # Extract the features from the DINOv2 model
        features = self.model.forward_features(encoder_input.image)["x_norm_patchtokens"]

        # Resize the features to the expected shape
        # (B x Num_patches x Embed_dim) -> (B x Embed_dim x H / Patch_Size x W / Patch_Size)
        features = features.permute(0, 2, 1)
        features = features.reshape(-1, self.enc_embed_dim, height // self.patch_size, width // self.patch_size)

        return ViTEncoderOutput(features=features)


if __name__ == "__main__":
    # Init different variants of DINOv2
    for size in ["small", "base", "large", "giant"]:
        for with_registers in [False, True]:
            name = f"dinov2_{size}"
            dinov2_encoder = DINOv2Encoder(name=name, size=size, with_registers=with_registers)

    # Init the custom pretrained DINOv2 encoders
    for size in ["small", "base", "large"]:
        pretrained_checkpoints_dict = {
            "small": "/ocean/projects/cis220039p/nkeetha/code/UniCeption/checkpoints/encoders/DINOv2_ViTS_DepthAnythingV2.pth",
            "base": "/ocean/projects/cis220039p/nkeetha/code/UniCeption/checkpoints/encoders/DINOv2_ViTB_DepthAnythingV2.pth",
            "large": "/ocean/projects/cis220039p/nkeetha/code/UniCeption/checkpoints/encoders/DINOv2_ViTL_DepthAnythingV2.pth",
        }
        name = f"dinov2_dav2_{size}"
        dinov2_encoder = DINOv2Encoder(
            name=name, size=size, with_registers=False, pretrained_checkpoint_path=pretrained_checkpoints_dict[size]
        )

    print("All DINOv2 Encoders have been initialized successfully!")
