"""
Encoder Class for Meta Perception Encoder (PE)
"""

from typing import List, Optional, Union

import torch

import uniception.models.libs.perception_encoder.vision_encoder.pe as pe
from uniception.models.encoders import UniCeptionViTEncoderBase, ViTEncoderInput, ViTEncoderOutput
from uniception.models.utils.intermediate_feature_return import IntermediateFeatureReturner


class PerceptionEncoder(UniCeptionViTEncoderBase):
    "Transcribed implementation of the Perception Encoder"

    def __init__(
        self,
        name: str,
        data_norm_type: str = "perception_encoder",
        patch_size: int = 14,
        size: str = "large",
        checkpoint_type: str = "core",
        pretrained_checkpoint_path: str = None,
        gradient_checkpointing: bool = False,
        keep_first_n_layers: Optional[int] = None,
        *args,
        **kwargs,
    ):
        """
        Wrapper around the Perception Encoder.

        Args:
            name (str): Name of the encoder.
            data_norm_type (str): Type of data normalization to use.
            patch_size (int): Size of the patches to use.
            size (str): Size of the model, e.g., 'large'.
            checkpoint_type (str): Type of checkpoint to use, e.g., 'core'.
            pretrained_checkpoint_path (str, optional): Path to a pretrained checkpoint.
            gradient_checkpointing (bool): Whether to use gradient checkpointing.
            keep_first_n_layers (Optional[int]): Number of layers to keep from the original model.
        """
        # Init the base class
        super().__init__(
            name=name,
            data_norm_type=data_norm_type,
            size=size,
            patch_size=patch_size,
            gradient_checkpointing=gradient_checkpointing,
            *args,
            **kwargs,
        )

        # Init the encoder specific attributes
        self.enc_embed_dim = {"small": 384, "base": 768, "large": 1024, "giant": 1536}[size]

        # Define PE model factory
        model_ckpt_dict = {
            ("core", "base"): "PE-Core-B16-224",
            ("core", "large"): "PE-Core-L14-336",
            ("core", "giant"): "PE-Core-G14-448",
            ("lang", "base"): "PE-Lang-B16-224",
            ("lang", "large"): "PE-Lang-L14-448",
            ("lang", "giant"): "PE-Lang-G14-448",
            ("spatial", "giant"): "PE-Spatial-G14-448",
        }

        # Get the model type
        model_str = model_ckpt_dict.get((checkpoint_type, size), None)
        if model_str is None:
            raise ValueError(f"Unsupported combination of checkpoint_type '{checkpoint_type}' and size '{size}'.")

        # Init pre-trained model
        self.model = pe.VisionTransformer.from_config(model_str, pretrained=True)

        # Enable gradient checkpointing if required
        self.model.set_grad_checkpointing(gradient_checkpointing)

        # Keep only the first n layers of the model if keep_first_n_layers is specified
        if keep_first_n_layers is not None:
            self.model.truncate(keep_first_n_layers)

        # Load the custom pretrained checkpoint if provided
        if pretrained_checkpoint_path:
            print(f"Loading custom pretrained Perception Encoder checkpoint from {pretrained_checkpoint_path}")
            ckpt = torch.load(pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"]))

    def forward(self, encoder_input: ViTEncoderInput) -> ViTEncoderOutput:
        """
        Perception Encoder Forward Pass

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

        # Extract the features from the PE model
        features = self.model.forward_features(
            encoder_input.image,
            norm=True,
            strip_cls_token=True,  # Strip the CLS token if it exists
        )

        # Permute the features to match the expected output shape
        features = features.permute(0, 2, 1).reshape(
            batch_size, -1, height // self.patch_size, width // self.patch_size
        )

        return ViTEncoderOutput(features=features)


class PerceptionEncoderIntermediateFeatureReturner(PerceptionEncoder, IntermediateFeatureReturner):
    "Intermediate Feature Returner for UniCeption Perception Encoder"

    def __init__(
        self,
        name: str,
        data_norm_type: str = "perception_encoder",
        patch_size: int = 14,
        size: str = "large",
        checkpoint_type: str = "core",
        pretrained_checkpoint_path: str = None,
        gradient_checkpointing: bool = False,
        keep_first_n_layers: Optional[int] = None,
        indices: Optional[Union[int, List[int]]] = 1,
        norm_intermediate: bool = True,
        *args,
        **kwargs,
    ):
        """
        Preception Encoder for extracting intermediate features.

        Args:
            name (str): Name of the encoder.
            data_norm_type (str): Type of data normalization to use.
            patch_size (int): Size of the patches to use.
            size (str): Size of the model, e.g., 'large'.
            checkpoint_type (str): Type of checkpoint to use, e.g., 'core'.
            pretrained_checkpoint_path (str, optional): Path to a pretrained checkpoint.
            gradient_checkpointing (bool): Whether to use gradient checkpointing.
            keep_first_n_layers (Optional[int]): Number of layers to keep from the original model.
            indices (Optional[Union[int, List[int]]], optional): Indices of the layers to return. Defaults to 1. Options:
            - int: Return the last n layers.
            - List[int]: Return the intermediate layers at the specified indices.
            norm_intermediate (bool, optional): Whether to normalize the intermediate features. Defaults to True.
        """
        # Init the base classes
        PerceptionEncoder.__init__(
            self,
            name=name,
            data_norm_type=data_norm_type,
            patch_size=patch_size,
            size=size,
            checkpoint_type=checkpoint_type,
            pretrained_checkpoint_path=pretrained_checkpoint_path,
            gradient_checkpointing=gradient_checkpointing,
            keep_first_n_layers=keep_first_n_layers,
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
        Forward pass for the Perception Encoder with intermediate feature return.

        Args:
            encoder_input (ViTEncoderInput): Input data for the encoder. Input data must contain image normalization type and normalized image tensor.

        Returns:
            List[ViTEncoderOutput]: List of outputs from the encoder at specified indices.
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
            self.indices = range(len(self.model.transformer.resblocks))

        # Extract the intermediate features from the PE model
        intermediate_feats = self.model.get_intermediate_layers(
            x=encoder_input.image,
            n=self.indices,
            norm=self.norm_intermediate,
            strip_cls_token=True,  # Strip the CLS token if it exists
        )

        # reshape the features to match the expected output shape
        outputs = []

        Hp, Wp = height // self.patch_size, width // self.patch_size
        outputs = [
            ViTEncoderOutput(features=feat.permute(0, 2, 1).reshape(batch_size, -1, Hp, Wp))
            for feat in intermediate_feats
        ]

        return outputs


if __name__ == "__main__":
    # Example usage
    encoder = PerceptionEncoderIntermediateFeatureReturner(
        name="PerceptionEncoder", size="giant", checkpoint_type="spatial", indices=[0, 1, 2, 3]
    ).cuda()

    dummy_input = ViTEncoderInput(image=torch.randn(1, 3, 448, 448).cuda(), data_norm_type="perception_encoder")

    output = encoder(dummy_input)

    for x in output:
        print(x.features.shape)  # Should print the shape of the output features
