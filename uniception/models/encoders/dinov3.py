"""
Encoder Class for DINOv3 - ViT & ConvNext Variants
"""

import os
from typing import List, Optional, Union

import torch
import torch.nn as nn

from uniception.models.encoders.base import (
    UniCeptionEncoderBase,
    UniCeptionViTEncoderBase,
    ViTEncoderInput,
    ViTEncoderOutput,
)
from uniception.models.utils.intermediate_feature_return import IntermediateFeatureReturner


class DINOv3Encoder(UniCeptionViTEncoderBase):
    "UniCeption DINOv3 ViT Encoder"

    def __init__(
        self,
        name: str,
        dinov3_repo_dir: str,
        data_norm_type: str = "dinov3",
        patch_size: int = 16,
        size: str = "large",
        weights: Optional[str] = None,
        pretrained_checkpoint_path: str = None,
        gradient_checkpointing: bool = False,
        keep_first_n_layers: Optional[int] = None,
        disable_torch_compile_for_pe=False,
        *args,
        **kwargs,
    ):
        """
        DINOv3 ViT Encoders for extracting spatial features from images.

        Args:
            name (str): Name of the encoder.
            dinov3_repo_dir (str): Path to the local directory where the DINOv3 repo was cloned. Required.
            data_norm_type (str): Image normalization type. Default: "dinov3"
            patch_size (int): Patch size for the encoder. Default: 16
            size (str): Size variant of the DINOv3 model. Options: ["small", "small+", "base", "large", "huge+", "7b"]. Default: "large"
            weights (Optional[str]): Path or URL to pretrained DINOv3 backbone checkpoint to pass to torch.hub.load. Default: None
            pretrained_checkpoint_path (str): Path to the pretrained checkpoint if using custom trained version of DINOv3. Default: None
            gradient_checkpointing (bool): Whether to use gradient checkpointing to save GPU memory during backward call. Default: False
            keep_first_n_layers (Optional[int]): If specified, only the first n layers of the model will be kept. Default: None
            disable_torch_compile_for_pe (bool): Whether to disable torch compile for PE interpolation. Default: False
        """
        # Init the base class
        super().__init__(
            name=name,
            data_norm_type=data_norm_type,
            patch_size=patch_size,
            gradient_checkpointing=gradient_checkpointing,
            *args,
            **kwargs,
        )

        # Init the DINOv3 Encoder specific attributes
        self.version = size
        self.enc_embed_dim = {
            "small": 384,
            "small+": 384,
            "base": 768,
            "large": 1024,
            "huge+": 1280,
            "7b": 4096,
        }[self.version]

        # Define DINOv3 model factory
        DINO_MODELS = {
            "small": "dinov3_vits16",
            "small+": "dinov3_vits16plus",
            "base": "dinov3_vitb16",
            "large": "dinov3_vitl16",
            "huge+": "dinov3_vith16plus",
            "7b": "dinov3_vit7b16",
        }

        # Load the pretrained DINOv3 model from local repo
        print(f"Loading {DINO_MODELS[self.version]} from local repo: {dinov3_repo_dir}")
        if weights:
            print(f"Using pretrained weights from: {weights}")
            self.model = torch.hub.load(
                dinov3_repo_dir,
                DINO_MODELS[self.version],
                source="local",
                weights=weights,
            )
        else:
            self.model = torch.hub.load(
                dinov3_repo_dir,
                DINO_MODELS[self.version],
                source="local",
                pretrained=False,
            )

        # Disable position interpolation at PE interpolation to enable torch compile
        # when training with multiple image shapes.
        if disable_torch_compile_for_pe:
            self.model.interpolate_pos_encoding = torch.compiler.disable(self.model.interpolate_pos_encoding)

        # Keep only the first n layers of the model if keep_first_n_layers is specified
        if keep_first_n_layers is not None:
            self.model.blocks = nn.ModuleList(self.model.blocks[:keep_first_n_layers])

        # Wrap the transformer blocks with support for gradient checkpointing if required
        if self.gradient_checkpointing:
            for i in range(len(self.model.blocks)):
                self.model.blocks[i] = self.wrap_module_with_gradient_checkpointing(self.model.blocks[i])

        # Load the custom pretrained checkpoint if provided
        if pretrained_checkpoint_path:
            print(f"Loading custom pretrained DINOv3 checkpoint from {pretrained_checkpoint_path}")
            ckpt = torch.load(pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"]))

    def forward(self, encoder_input: ViTEncoderInput) -> ViTEncoderOutput:
        """
        DINOv3 Encoder Forward Pass

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

        # Extract the features from the DINOv3 model
        features = self.model.forward_features(encoder_input.image)["x_norm_patchtokens"]

        # Resize the features to the expected shape
        # (B x Num_patches x Embed_dim) -> (B x Embed_dim x H / Patch_Size x W / Patch_Size)
        features = features.permute(0, 2, 1)
        features = features.reshape(
            -1, self.enc_embed_dim, height // self.patch_size, width // self.patch_size
        ).contiguous()

        return ViTEncoderOutput(features=features)


class DINOv3IntermediateFeatureReturner(DINOv3Encoder, IntermediateFeatureReturner):
    "Intermediate Feature Returner for UniCeption DINOv3 ViT Encoder"

    def __init__(
        self,
        name: str,
        dinov3_repo_dir: str,
        data_norm_type: str = "dinov3",
        patch_size: int = 16,
        size: str = "large",
        weights: Optional[str] = None,
        pretrained_checkpoint_path: str = None,
        gradient_checkpointing: bool = False,
        keep_first_n_layers: Optional[int] = None,
        disable_torch_compile_for_pe=False,
        indices: Optional[Union[int, List[int]]] = 1,
        norm_intermediate: bool = True,
        *args,
        **kwargs,
    ):
        """
        DINOv3 Encoder for extracting spatial features from images.

        Args:
            name (str): Name of the encoder.
            dinov3_repo_dir (str): Path to the local directory where the DINOv3 repo was cloned. Required.
            data_norm_type (str): Image normalization type. Default: "dinov3"
            patch_size (int): Patch size for the encoder. Default: 16
            size (str): Size variant of the DINOv3 model. Options: ["small", "small+", "base", "large", "huge+", "7b"]. Default: "large"
            weights (Optional[str]): Path or URL to pretrained DINOv3 backbone checkpoint to pass to torch.hub.load. Default: None
            pretrained_checkpoint_path (str): Path to the pretrained checkpoint if using custom trained version of DINOv3. Default: None
            gradient_checkpointing (bool): Whether to use gradient checkpointing to save GPU memory during backward call. Default: False
            keep_first_n_layers (Optional[int]): If specified, only the first n layers of the model will be kept. Default: None
            disable_torch_compile_for_pe (bool): Whether to disable torch compile for PE interpolation. Default: False
            indices (Optional[Union[int, List[int]]], optional): Indices of the layers to return. Defaults to 1. Options:
            - int: Return the last n layers.
            - List[int]: Return the intermediate layers at the specified indices.
            norm_intermediate (bool, optional): Whether to normalize the intermediate features. Defaults to True.
        """
        # Init the base classes
        DINOv3Encoder.__init__(
            self,
            name=name,
            data_norm_type=data_norm_type,
            patch_size=patch_size,
            size=size,
            dinov3_repo_dir=dinov3_repo_dir,
            weights=weights,
            pretrained_checkpoint_path=pretrained_checkpoint_path,
            gradient_checkpointing=gradient_checkpointing,
            keep_first_n_layers=keep_first_n_layers,
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
        DINOv3 Encoder Forward Pass with Intermediate Feature Return

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

        # Extract the intermediate features from the DINOv3 model
        intermediate_features = self.model.get_intermediate_layers(
            encoder_input.image,
            n=self.indices,
            reshape=True,
            norm=self.norm_intermediate,
        )

        # Convert the intermediate features to a list of ViTEncoderOutput
        intermediate_features = [ViTEncoderOutput(features=features) for features in intermediate_features]

        return intermediate_features


class DINOv3ConvNextEncoder(UniCeptionEncoderBase):
    "UniCeption DINOv3 ConvNext Encoder"

    def __init__(
        self,
        name: str,
        dinov3_repo_dir: str,
        data_norm_type: str = "dinov3",
        patch_size: int = 16,
        size: str = "large",
        weights: Optional[str] = None,
        pretrained_checkpoint_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        DINOv3 ConvNext Encoders for extracting spatial features from images.

        Args:
            name (str): Name of the encoder.
            dinov3_repo_dir (str): Path to the local directory where the DINOv3 repo was cloned. Required.
            data_norm_type (str): Image normalization type. Default: "dinov3"
            patch_size (int): Patch size for the encoder. Default: 16
            size (str): Size variant of the DINOv3 model. Options: ["tiny", "small", "base", "large"]. Default: "large"
            weights (Optional[str]): Path or URL to pretrained DINOv3 backbone checkpoint to pass to torch.hub.load. Default: None
            pretrained_checkpoint_path (str): Path to the pretrained checkpoint if using custom trained version of DINOv3. Default: None
        """
        # Init the base class
        super().__init__(name=name, data_norm_type=data_norm_type, *args, **kwargs)

        # Init the DINOv3 ConvNext Encoder specific attributes
        self.patch_size = patch_size
        self.final_layer_patch_size = 32  # DINOv3 ConvNext uses a final patch size of 32
        self.version = size
        self.enc_embed_dim = {
            "tiny": 768,
            "small": 768,
            "base": 1024,
            "large": 1536,
        }[self.version]

        # Define DINOv3 ConvNext factory
        DINO_CONV_MODELS = {
            "tiny": "dinov3_convnext_tiny",
            "small": "dinov3_convnext_small",
            "base": "dinov3_convnext_base",
            "large": "dinov3_convnext_large",
        }

        # Load the pretrained DINOv3 model from local repo
        print(f"Loading {DINO_CONV_MODELS[self.version]} from local repo: {dinov3_repo_dir}")
        if weights:
            print(f"Using pretrained weights from: {weights}")
            self.model = torch.hub.load(
                dinov3_repo_dir,
                DINO_CONV_MODELS[self.version],
                source="local",
                weights=weights,
            )
        else:
            self.model = torch.hub.load(
                dinov3_repo_dir,
                DINO_CONV_MODELS[self.version],
                source="local",
                pretrained=False,
            )

        # Init the patch size arg for the model
        self.model.patch_size = self.patch_size

        # Load the custom pretrained checkpoint if provided
        if pretrained_checkpoint_path:
            print(f"Loading custom pretrained DINOv3 checkpoint from {pretrained_checkpoint_path}")
            ckpt = torch.load(pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"]))

    def forward(self, encoder_input: ViTEncoderInput) -> ViTEncoderOutput:
        """
        DINOv3 Encoder Forward Pass

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

        # Extract the features from the DINOv3 model
        features = self.model.forward_features(encoder_input.image)["x_norm_patchtokens"]

        # Resize the features to the expected shape
        # (B x Num_patches x Embed_dim) -> (B x Embed_dim x H / Patch_Size x W / Patch_Size)
        features = features.permute(0, 2, 1)
        features = features.reshape(
            -1,
            self.enc_embed_dim,
            height // self.final_layer_patch_size,
            width // self.final_layer_patch_size,
        ).contiguous()

        return ViTEncoderOutput(features=features)


class DINOv3ConvNextIntermediateFeatureReturner(DINOv3ConvNextEncoder, IntermediateFeatureReturner):
    "Intermediate Feature Returner for UniCeption DINOv3 ConvNext Encoder"

    def __init__(
        self,
        name: str,
        dinov3_repo_dir: str,
        data_norm_type: str = "dinov3",
        patch_size: int = 16,
        size: str = "large",
        weights: Optional[str] = None,
        pretrained_checkpoint_path: Optional[str] = None,
        indices: Optional[Union[int, List[int]]] = 1,
        norm_intermediate: bool = True,
        *args,
        **kwargs,
    ):
        """
        DINOv3 ConvNext Encoders for extracting spatial features from images.

        Args:
            name (str): Name of the encoder.
            dinov3_repo_dir (str): Path to the local directory where the DINOv3 repo was cloned. Required.
            data_norm_type (str): Image normalization type. Default: "dinov3"
            patch_size (int): Patch size for the encoder. Default: 16
            size (str): Size variant of the DINOv3 model. Options: ["tiny", "small", "base", "large"]. Default: "large"
            weights (Optional[str]): Path or URL to pretrained DINOv3 backbone checkpoint to pass to torch.hub.load. Default: None
            pretrained_checkpoint_path (str): Path to the pretrained checkpoint if using custom trained version of DINOv3. Default: None
            indices (Optional[Union[int, List[int]]], optional): Indices of the layers to return. Defaults to 1. Options:
            - int: Return the last n layers.
            - List[int]: Return the intermediate layers at the specified indices.
            norm_intermediate (bool, optional): Whether to normalize the intermediate features. Defaults to True.
        """
        # Init the base classes
        DINOv3ConvNextEncoder.__init__(
            self,
            name=name,
            data_norm_type=data_norm_type,
            patch_size=patch_size,
            size=size,
            dinov3_repo_dir=dinov3_repo_dir,
            weights=weights,
            pretrained_checkpoint_path=pretrained_checkpoint_path,
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
        DINOv3 ConvNext Encoder Forward Pass with Intermediate Feature Return

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
            self.indices = range(len(self.model.downsample_layers))

        # Extract the intermediate features from the DINOv3 model
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
    # Set the DINOv3 repo directory and checkpoint directory
    dinov3_repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../local/dinov3"))
    checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../checkpoints/encoders"))

    # Init different variants of DINOv3
    # Hash mapping for each size variant
    hash_map = {
        "small": "08c60483",
        "small+": "4057cbaa",
        "base": "73cec8be",
        "large": "8aa4cbdd",
        "huge+": "7c1da9a5",
        "7b": "a955f4ea",
    }

    for size in ["small", "small+", "base", "large", "huge+", "7b"]:
        name = f"dinov3_{size}"
        size_str = size[0] if size != "7b" else "7b"
        size_str_plus = "plus" if "+" in size else ""
        ckpt_name = f"dinov3_vit{size_str}16{size_str_plus}_pretrain_lvd1689m-{hash_map[size]}.pth"
        pretrained_checkpoint_path = os.path.join(checkpoint_dir, ckpt_name)
        dinov3_encoder = DINOv3Encoder(
            name=name,
            dinov3_repo_dir=dinov3_repo_dir,
            size=size,
            weights=pretrained_checkpoint_path,
        )

        if size == "small":
            # Run dummy forward pass
            dummy_input = ViTEncoderInput(image=torch.randn(1, 3, 224, 224), data_norm_type="dinov3")
            dummy_output = dinov3_encoder(dummy_input)

    # Init different variants of DINOv3 - ConvNext
    # Hash mapping for each ConvNext size variant
    convnext_hash_map = {
        "tiny": "21b726bb",
        "small": "296db49d",
        "base": "801f2ba9",
        "large": "61fa432d",
    }

    for size in ["tiny", "small", "base", "large"]:
        ckpt_name = f"dinov3_convnext_{size}_pretrain_lvd1689m-{convnext_hash_map[size]}.pth"
        pretrained_checkpoint_path = os.path.join(checkpoint_dir, ckpt_name)
        dinov3_convnext_encoder = DINOv3ConvNextEncoder(
            name="dino_convnext",
            dinov3_repo_dir=dinov3_repo_dir,
            size=size,
            weights=pretrained_checkpoint_path,
        )

        if size == "tiny":
            # Run dummy forward pass
            dummy_input = ViTEncoderInput(image=torch.randn(1, 3, 224, 224), data_norm_type="dinov3")
            dummy_output = dinov3_convnext_encoder(dummy_input)

    print("All DINOv3 ViT & ConvNext Encoders have been initialized successfully!")

    # Intermediate Feature Returner Tests
    print("Running Intermediate Feature Returner Tests...")

    # Run the intermediate feature returner with last-n index
    dinov3_intermediate_feature_returner = DINOv3IntermediateFeatureReturner(
        name="dinov3_small", dinov3_repo_dir=dinov3_repo_dir, size="small", indices=6
    )  # Last 6 layers
    dummy_input = ViTEncoderInput(image=torch.randn(1, 3, 224, 224), data_norm_type="dinov3")
    output = dinov3_intermediate_feature_returner(dummy_input)
    assert isinstance(output, list), "Output must be a list of intermediate features"
    assert isinstance(output[0], ViTEncoderOutput), "Output must be a list of ViTEncoderOutput"
    assert len(output) == 6, "Output must have length of intermediate features equal to the number of indices"

    # Run the intermediate feature returner with specific indices
    dinov3_intermediate_feature_returner = DINOv3IntermediateFeatureReturner(
        name="dinov3_small",
        dinov3_repo_dir=dinov3_repo_dir,
        size="small",
        indices=[0, 2, 4, 6],
    )  # Specific layers
    dummy_input = ViTEncoderInput(image=torch.randn(1, 3, 224, 224), data_norm_type="dinov3")
    output = dinov3_intermediate_feature_returner(dummy_input)
    assert isinstance(output, list), "Output must be a list of intermediate features"
    assert isinstance(output[0], ViTEncoderOutput), "Output must be a list of ViTEncoderOutput"
    assert len(output) == 4, "Output must have length of intermediate features equal to the number of indices"

    # Run the intermediate feature returner with last-n index
    dinov3_intermediate_feature_returner = DINOv3ConvNextIntermediateFeatureReturner(
        name="dinov3_convnext_small",
        dinov3_repo_dir=dinov3_repo_dir,
        size="small",
        indices=4,
    )  # Last 4 layers
    dummy_input = ViTEncoderInput(image=torch.randn(1, 3, 224, 224), data_norm_type="dinov3")
    output = dinov3_intermediate_feature_returner(dummy_input)
    assert isinstance(output, list), "Output must be a list of intermediate features"
    assert isinstance(output[0], ViTEncoderOutput), "Output must be a list of ViTEncoderOutput"
    assert len(output) == 4, "Output must have length of intermediate features equal to the number of indices"

    # Run the intermediate feature returner with specific indices
    dinov3_intermediate_feature_returner = DINOv3ConvNextIntermediateFeatureReturner(
        name="dinov3_convnext_small",
        dinov3_repo_dir=dinov3_repo_dir,
        size="small",
        indices=[0, 2],
    )  # Specific layers
    dummy_input = ViTEncoderInput(image=torch.randn(1, 3, 224, 224), data_norm_type="dinov3")
    output = dinov3_intermediate_feature_returner(dummy_input)
    assert isinstance(output, list), "Output must be a list of intermediate features"
    assert isinstance(output[0], ViTEncoderOutput), "Output must be a list of ViTEncoderOutput"
    assert len(output) == 2, "Output must have length of intermediate features equal to the number of indices"

    print("All Intermediate Feature Returner Tests have passed successfully!")
