"""
Encoder Class for DUNE

DUNE uses the same implementation as DINOv2, where the only difference is the loaded weights.
"""

from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from uniception.models.encoders.base import UniCeptionViTEncoderBase, ViTEncoderInput, ViTEncoderOutput
from uniception.models.utils.intermediate_feature_return import IntermediateFeatureReturner


class DUNEEncoder(UniCeptionViTEncoderBase):
    "UniCeption DUNE Encoder"

    def __init__(
        self,
        name: str,
        pretrained_checkpoint_path: str,
        data_norm_type: str = "dune",
        patch_size: int = 14,
        vit_size: str = "base",
        pe_image_size: int = 448,
        torch_hub_force_reload: bool = False,
        gradient_checkpointing: bool = False,
        keep_first_n_layers: Optional[int] = None,
        use_pytorch_sdpa=True,
        disable_torch_compile_for_pe=False,
        *args,
        **kwargs,
    ):
        """
        DUNE Encoder for extracting spatial features from images.
        DUNE uses the same implementation as DINOv2 with registers, with custom pretrained weights.

        Args:
            name (str): Name of the encoder.
            pretrained_checkpoint_path (str): Path to the pretrained DUNE checkpoint.
            data_norm_type (str): Image normalization type. Default: "dune"
            patch_size (int): Patch size for the encoder. Default: 14
            vit_size (str): Size variant of the ViT model. Default: "base"
            pe_image_size (int): Image size for position encoding. Default: 448
            torch_hub_force_reload (bool): Whether to force reload the model from torch hub. Default: False
            gradient_checkpointing (bool): Whether to use gradient checkpointing to save GPU memory during backward call. Default: False
            keep_first_n_layers (Optional[int]): If specified, only the first n layers of the model will be kept. Default: None
            use_pytorch_sdpa (bool): Whether to use PyTorch native SDPA for attention layers. Default: True
            disable_torch_compile_for_pe (bool): Whether to disable torch compile for PE interpolation. Default: False
        """
        size = vit_size
        with_registers = True  # all DUNE encoder have registers

        # Init the base class
        name = name if not with_registers else f"{name}_reg"
        super().__init__(
            name=name,
            data_norm_type=data_norm_type,
            patch_size=patch_size,
            gradient_checkpointing=gradient_checkpointing,
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
        try:  # Requires internet access
            self.model = torch.hub.load(
                "facebookresearch/dinov2",
                DINO_MODELS[self.with_registers][self.version],
                force_reload=torch_hub_force_reload,
            )
        except:  # Load from cache
            self.model = torch.hub.load(
                "facebookresearch/dinov2",
                DINO_MODELS[self.with_registers][self.version],
            )

        del (
            self.model.mask_token
        )  # This parameter is unused in producing patch features, and will lead to unused parameters

        # Patch DINOv2 position encoding table to have the correct shape
        num_patches = (pe_image_size // 14) ** 2
        self.model.pos_embed = torch.nn.Parameter(
            torch.zeros(1, num_patches + self.model.num_tokens, self.enc_embed_dim)
        )

        # disable position interpolation at PE interpolation to enable torch compile
        # when training with multiple image shapes.
        if disable_torch_compile_for_pe:
            self.model.interpolate_pos_encoding = torch.compiler.disable(self.model.interpolate_pos_encoding)

        # Keep only the first n layers of the model if keep_first_n_layers is specified
        if keep_first_n_layers is not None:
            self.model.blocks = nn.ModuleList(self.model.blocks[:keep_first_n_layers])

        # Use Native Torch SDPA for attention layers if specified (instead of DINOv2's XFormers)
        if use_pytorch_sdpa:
            self.enable_pytorch_native_sdpa()

        # Wrap the transformer blocks with support for gradient checkpointing if required
        if self.gradient_checkpointing:
            for i in range(len(self.model.blocks)):
                self.model.blocks[i] = self.wrap_module_with_gradient_checkpointing(self.model.blocks[i])

        # Load the custom pretrained checkpoint if provided
        if pretrained_checkpoint_path:
            print(f"Loading DUNE pretrained checkpoint from {pretrained_checkpoint_path}")
            ckpt = torch.load(pretrained_checkpoint_path, weights_only=False)
            # Extract and remap encoder weights from DUNE checkpoint
            encoder_state_dict = self._extract_and_remap_encoder_weights(ckpt["model"])
            print(self.load_state_dict(encoder_state_dict, strict=False))

    def enable_pytorch_native_sdpa(self):
        "Enable PyTorch native SDPA for attention layers"
        for i in range(len(self.model.blocks)):
            self.model.blocks[i].attn = self.wrap_dinov2_attention_with_sdpa(self.model.blocks[i].attn)

    def wrap_dinov2_attention_with_sdpa(self, module: nn.Module):
        "Wrap DINOv2 attention module with PyTorch native SDPA"
        assert torch.__version__ >= "2.0", "SDPA requires PyTorch 2.0 or later"

        class _AttentionWrapper(module.__class__):
            "SDPA Attention Wrapper Class"

            def forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:
                B, N, C = x.shape
                qkv = (
                    self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                )  # (3, B, H, N, C // H)

                q, k, v = torch.unbind(qkv, 0)  # (B, H, N, C // H)

                x = F.scaled_dot_product_attention(q, k, v, attn_bias)
                x = x.permute(0, 2, 1, 3).reshape(B, N, C)

                x = self.proj(x)
                x = self.proj_drop(x)
                return x

        module.__class__ = _AttentionWrapper
        return module

    def _extract_and_remap_encoder_weights(self, checkpoint):
        """
        Extract encoder weights from DUNE checkpoint and remap keys to match DINOv2 model structure.

        DUNE checkpoint structure:
        - Keys are prefixed with "encoder." instead of "model."
        - Blocks are structured as "encoder.blocks.0.0." instead of "model.blocks.0."
        - Checkpoint also contains projectors and teacher_norms that we don't need

        Args:
            checkpoint: Loaded DUNE checkpoint dictionary

        Returns:
            dict: Remapped state dict with "model." prefix and correct block structure
        """
        encoder_state_dict = {}

        for key, value in checkpoint.items():
            # Only process encoder keys, skip projectors and teacher_norms
            if not key.startswith("encoder."):
                continue

            # Remove "encoder." prefix
            new_key = key.replace("encoder.", "", 1)

            # Fix blocks structure: "blocks.0.0." -> "blocks.0."
            # The checkpoint has blocks structured as blocks.0.0, blocks.0.1, etc.
            # We need to flatten this to blocks.0, blocks.1, etc.
            if "blocks." in new_key:
                parts = new_key.split(".")
                if len(parts) >= 3 and parts[0] == "blocks":
                    # Extract block indices (e.g., "0" and "0" from "blocks.0.0")
                    major_idx = parts[1]
                    minor_idx = parts[2]
                    # Combine indices: blocks.0.0 means the first block in the checkpoint
                    # We'll just use the minor index as the actual block number
                    # since blocks are numbered sequentially in the checkpoint
                    parts[1] = minor_idx
                    parts.pop(2)  # Remove the major index
                    new_key = ".".join(parts)

            # Add "model." prefix to match our model structure
            new_key = "model." + new_key
            encoder_state_dict[new_key] = value

        # Remove the mask token from the state dict
        # This parameter is unused in producing patch features, and will lead to unused parameters
        del encoder_state_dict["model.mask_token"]

        return encoder_state_dict

    def forward(self, encoder_input: ViTEncoderInput) -> ViTEncoderOutput:
        """
        DUNE Encoder Forward Pass

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
        features = self.model.forward_features(encoder_input.image)["x_norm_patchtokens"]

        # Resize the features to the expected shape
        # (B x Num_patches x Embed_dim) -> (B x Embed_dim x H / Patch_Size x W / Patch_Size)
        features = features.permute(0, 2, 1)
        features = features.reshape(
            -1, self.enc_embed_dim, height // self.patch_size, width // self.patch_size
        ).contiguous()

        return ViTEncoderOutput(features=features)


class DUNEIntermediateFeatureReturner(DUNEEncoder, IntermediateFeatureReturner):
    "Intermediate Feature Returner for UniCeption DUNE Encoder"

    def __init__(
        self,
        name: str,
        pretrained_checkpoint_path: str,
        data_norm_type: str = "dune",
        patch_size: int = 14,
        vit_size: str = "base",
        pe_image_size: int = 448,
        torch_hub_force_reload: bool = False,
        gradient_checkpointing: bool = False,
        keep_first_n_layers: Optional[int] = None,
        use_pytorch_sdpa=True,
        disable_torch_compile_for_pe=False,
        indices: Optional[Union[int, List[int]]] = 1,
        norm_intermediate: bool = True,
        *args,
        **kwargs,
    ):
        """
        DUNE Encoder for extracting spatial features from images with intermediate feature return.
        DUNE uses the same implementation as DINOv2 with registers, with custom pretrained weights.

        Args:
            name (str): Name of the encoder.
            pretrained_checkpoint_path (str): Path to the pretrained DUNE checkpoint.
            data_norm_type (str): Image normalization type. Default: "dune"
            patch_size (int): Patch size for the encoder. Default: 14
            vit_size (str): Size variant of the ViT model. Default: "base"
            pe_image_size (int): Image size for position encoding. Default: 448
            torch_hub_force_reload (bool): Whether to force reload the model from torch hub. Default: False
            gradient_checkpointing (bool): Whether to use gradient checkpointing to save GPU memory during backward call. Default: False
            keep_first_n_layers (Optional[int]): If specified, only the first n layers of the model will be kept. Default: None
            use_pytorch_sdpa (bool): Whether to use PyTorch native SDPA for attention layers. Default: True
            disable_torch_compile_for_pe (bool): Whether to disable torch compile for PE interpolation. Default: False
            indices (Optional[Union[int, List[int]]], optional): Indices of the layers to return. Defaults to 1. Options:
            - int: Return the last n layers.
            - List[int]: Return the intermediate layers at the specified indices.
            norm_intermediate (bool, optional): Whether to normalize the intermediate features. Defaults to True.
        """
        DUNEEncoder.__init__(
            self,
            name=name,
            pretrained_checkpoint_path=pretrained_checkpoint_path,
            data_norm_type=data_norm_type,
            patch_size=patch_size,
            vit_size=vit_size,
            pe_image_size=pe_image_size,
            torch_hub_force_reload=torch_hub_force_reload,
            gradient_checkpointing=gradient_checkpointing,
            keep_first_n_layers=keep_first_n_layers,
            use_pytorch_sdpa=use_pytorch_sdpa,
            disable_torch_compile_for_pe=disable_torch_compile_for_pe,
            *args,
            **kwargs,
        )

        IntermediateFeatureReturner.__init__(
            self,
            indices=indices,
            norm_intermediate=norm_intermediate,
        )

    def forward(self, encoder_input: ViTEncoderInput) -> List[ViTEncoderOutput]:
        """
        DUNE Encoder Forward Pass with Intermediate Feature Return

        Args:
            encoder_input (ViTEncoderInput): Input data for the encoder. Input data must contain image normalization type and normalized image tensor.

        Returns:
            List[ViTEncoderOutput]: Output data from the encoder. Returns a list of intermediate features.
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

        if self.indices is None:
            self.indices = range(len(self.model.blocks))

        # Extract the intermediate features from the DINOv2 model
        intermediate_features = self.model.get_intermediate_layers(
            encoder_input.image,
            n=self.indices,
            reshape=True,
            norm=self.norm_intermediate,
        )

        # Convert the intermediate features to a list of ViTEncoderOutput
        intermediate_features = [ViTEncoderOutput(features=features) for features in intermediate_features]

        return intermediate_features


if __name__ == "__main__":
    # DUNE only has one variant: ViT-Base at 448 resolution
    # Checkpoint: https://download.europe.naverlabs.com/dune/dune_vitbase14_448.pth
    pretrained_checkpoint_path = "../../../checkpoints/encoders/dune_vitbase14_448.pth"

    # Init DUNE encoder
    dune_encoder = DUNEEncoder(
        name="dune_base",
        vit_size="base",
        pe_image_size=448,
        pretrained_checkpoint_path=pretrained_checkpoint_path,
    )

    print("DUNE Encoder has been initialized successfully!")

    # Intermediate Feature Returner Tests
    print("Running Intermediate Feature Returner Tests...")

    # Run the intermediate feature returner with last-n index
    dune_intermediate_feature_returner = DUNEIntermediateFeatureReturner(
        name="dune_base",
        vit_size="base",
        pe_image_size=448,
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        indices=6,
    )  # Last 6 layers
    dummy_input = ViTEncoderInput(image=torch.randn(1, 3, 448, 448), data_norm_type="dune")
    output = dune_intermediate_feature_returner(dummy_input)
    assert isinstance(output, list), "Output must be a list of intermediate features"
    assert isinstance(output[0], ViTEncoderOutput), "Output must be a list of ViTEncoderOutput"
    assert len(output) == 6, "Output must have length of intermediate features equal to the number of indices"

    # Run the intermediate feature returner with specific indices
    dune_intermediate_feature_returner = DUNEIntermediateFeatureReturner(
        name="dune_base",
        vit_size="base",
        pe_image_size=448,
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        indices=[0, 2, 4, 6],
    )  # Specific layers
    dummy_input = ViTEncoderInput(image=torch.randn(1, 3, 448, 448), data_norm_type="dune")
    output = dune_intermediate_feature_returner(dummy_input)
    assert isinstance(output, list), "Output must be a list of intermediate features"
    assert isinstance(output[0], ViTEncoderOutput), "Output must be a list of ViTEncoderOutput"
    assert len(output) == 4, "Output must have length of intermediate features equal to the number of indices"

    print("All Intermediate Feature Returner Tests have passed successfully!")
