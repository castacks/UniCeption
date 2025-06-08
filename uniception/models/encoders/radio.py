"""
Encoder Class for RADIO (Nvidia)
"""

from typing import List, Optional, Tuple, Union

import torch

from uniception.models.encoders.base import UniCeptionViTEncoderBase, ViTEncoderInput, ViTEncoderOutput
from uniception.models.utils.intermediate_feature_return import IntermediateFeatureReturner


class RADIOEncoder(UniCeptionViTEncoderBase):
    "UniCeption RADIO Encoder"

    def __init__(
        self,
        name: str,
        data_norm_type: str = "radio",
        patch_size: int = 16,
        model_version: str = "radio_v2.5-l",
        pretrained_checkpoint_path: str = None,
        eradio_input_shape: Optional[tuple] = None,
        torch_hub_force_reload: bool = False,
        keep_first_n_layers: Optional[int] = None,
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
            pretrained_checkpoint_path (str): Path to the pretrained checkpoint if using custom trained version of RADIO. Default: None
            eradio_input_shape (tuple): Input shape (height, width) for E-RADIO models. Default: None
            torch_hub_force_reload (bool): Whether to force reload the model from torch hub. Default: False
            keep_first_n_layers (Optional[int]): Number of layers to keep from the pretrained model. Default: None
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
        self.enc_embed_dim = {
            "radio_v2.5-b": 768,
            "radio_v2.5-l": 1024,
            "radio_v2.5-h": 1280,
            "radio_v2.5-g": 1536,
            "e-radio_v2": 1536,
        }[self.model_version]

        if self.model_version == "radio_v2.5-g":
            assert patch_size == 14, "Patch size must be 14 for RADIO v2.5-g"
        else:
            assert patch_size == 16, "Patch size must be 16 for all other versions of RADIO"

        # Load the pretrained RADIO model from torch hub
        print(f"Loading pretrained {self.model_version} from torch hub")
        try:  # Requires internet access
            self.model = torch.hub.load(
                "NVlabs/RADIO",
                "radio_model",
                version=self.model_version,
                progress=True,
                skip_validation=True,
                force_reload=torch_hub_force_reload,
            )
        except:  # Load from cache
            self.model = torch.hub.load(
                "NVlabs/RADIO",
                "radio_model",
                version=self.model_version,
                progress=True,
                skip_validation=True,
            )

        # Delete the excess blocks if keep_first_n_layers is specified
        if keep_first_n_layers is not None:
            assert keep_first_n_layers < len(
                self.model.model.blocks
            ), "keep_first_n_layers must be less than the number of blocks"
            print(f"Keeping only the first {keep_first_n_layers} layers of the model")
            self.model.model.blocks = torch.nn.ModuleList(self.model.model.blocks[:keep_first_n_layers])

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
        """
        RADIO Encoder Forward Pass

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

        # Forward pass throught the RADIO encoder
        summary, features = self.model(encoder_input.image)

        # Resize the features to the expected shape
        # (B x Num_patches x Embed_dim) -> (B x Embed_dim x H / Patch_Size x W / Patch_Size)
        features = features.permute(0, 2, 1)
        features = features.reshape(
            -1, self.enc_embed_dim, height // self.patch_size, width // self.patch_size
        ).contiguous()

        return ViTEncoderOutput(features=features)


class RADIOIntermediateFeatureReturner(RADIOEncoder, IntermediateFeatureReturner):
    "Intermediate Feature Returner for UniCeption RADIO Encoder"

    def __init__(
        self,
        name: str,
        data_norm_type: str = "radio",
        patch_size: int = 16,
        model_version: str = "radio_v2.5-l",
        pretrained_checkpoint_path: str = None,
        eradio_input_shape: Optional[tuple] = None,
        indices: Union[int, List[int]] = [-1],
        norm_intermediate: bool = True,
        stop_early: bool = False,
        intermediates_only: bool = True,
        feature_adaptor: Optional[str] = None,
        keep_first_n_layers: Optional[int] = None,
        *args,
        **kwargs,
    ):
        """
        Intermediate Feature Returner for the RADIO Encoder.

        Args:
            name (str): Name of the encoder.
            data_norm_type (str): Image normalization type. Default: "radio"
            patch_size (int): Patch size for the encoder. Default: 16
            model_version (str): Version of the RADIO model to load. Default: "radio_v2.5-l"
            pretrained_checkpoint_path (str): Path to the pretrained checkpoint if using custom trained version of RADIO.
            eradio_input_shape (tuple): Input shape (height, width) for E-RADIO models. Default: None
            indices (Optional[Union[int, List[int]]], optional): Indices of the layers to return. Defaults to [-1]. Options:
            - int: Return the last n layers.
            - List[int]: Return the intermediate layers at the specified indices.
            norm_intermediate (bool, optional): Whether to normalize the intermediate features. Defaults to True.
            stop_early (bool, optional): Whether to stop early. Defaults to False.
            intermediates_only (bool, optional): Whether to return only the intermediate features. Defaults to True.
            feature_adaptor (Optional[str], optional): Feature adaptor to use. Defaults to None. Currently supported: "dino_v2".
            keep_first_n_layers (Optional[int], optional): Number of layers to keep from the pretrained model. Defaults to None.
        """
        # Init the base classes
        RADIOEncoder.__init__(
            self,
            name=name,
            data_norm_type=data_norm_type,
            patch_size=patch_size,
            model_version=model_version,
            pretrained_checkpoint_path=pretrained_checkpoint_path,
            eradio_input_shape=eradio_input_shape,
            keep_first_n_layers=keep_first_n_layers,
            *args,
            **kwargs,
        )
        IntermediateFeatureReturner.__init__(
            self,
            indices=indices,
            norm_intermediate=norm_intermediate,
            stop_early=stop_early,
            intermediates_only=intermediates_only,
        )

        # Convert indices to absolute indices if indices is None
        if self.indices is None:
            self.indices = list(range(len(self.model.model.blocks)))

        self.feature_adaptor = feature_adaptor
        if self.feature_adaptor is None:
            pass
        elif self.feature_adaptor == "dino_v2":
            # Initialize a dummy radio encoder with the adaptor setting
            dummy_model = torch.hub.load(
                "NVlabs/RADIO",
                "radio_model",
                version=self.model_version,
                progress=True,
                skip_validation=True,
                adaptor_names="dino_v2",
            )

            # Extract its feature converter weights
            self.spatial_feature_converter = dummy_model.adaptors["dino_v2"].feat_mlp

            # Update the embedding dimension because the features have been projected
            self.enc_embed_dim = self.spatial_feature_converter.final[-1].out_features

            del dummy_model
        else:
            raise ValueError("Unsupported feature adaptor. Supported: dino_v2")

    def forward(
        self, encoder_input: ViTEncoderInput
    ) -> Union[List[ViTEncoderOutput], Tuple[ViTEncoderOutput, List[ViTEncoderOutput]]]:
        """
        RADIO Encoder Forward Pass with Intermediate Feature Return

        Args:
            encoder_input (ViTEncoderInput): Input data for the encoder. Input data must contain image normalization type and normalized image tensor.

        Returns:
            Union[List[ViTEncoderOutput], Tuple[ViTEncoderOutput, List[ViTEncoderOutput]]]: Output data from the encoder.
                If `intermediates_only` is True, returns a list of intermediate features.
                Otherwise, returns a tuple with the final features and a list of intermediate features.
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

        # Extract the final features and intermediate features accordingly
        model_outputs = self.model.forward_intermediates(
            encoder_input.image,
            indices=self.indices,
            return_prefix_tokens=False,
            norm=self.norm_intermediate,
            stop_early=self.stop_early,
            output_fmt="NLC",
            intermediates_only=self.intermediates_only,
        )

        # Extract the final features and intermediate features accordingly
        final_features, intermediate_features = None, None
        if self.intermediates_only:
            intermediate_features = model_outputs
        else:
            final_features = model_outputs[0].features.contiguous()
            intermediate_features = model_outputs[1]

        # Optionally convert the features using the feature adaptor
        Hp, Wp = height // self.patch_size, width // self.patch_size

        # Convert final features
        if final_features is not None:
            if self.feature_adaptor is not None:
                final_features = self.spatial_feature_converter(final_features)

            # Convert to BCHW and package
            final_features = final_features.view(batch_size, Hp, Wp, -1).permute(0, 3, 1, 2)
            final_features = ViTEncoderOutput(features=final_features)

        # Convert intermediate features
        if intermediate_features is not None:
            num_intermediate = len(intermediate_features)
            all_intermediate_feats_tensor = torch.cat(intermediate_features, dim=0)
            if self.feature_adaptor is not None:
                all_intermediate_feats_tensor = self.spatial_feature_converter(all_intermediate_feats_tensor)
            # Convert to BCHW
            all_intermediate_feats_tensor = all_intermediate_feats_tensor.view(
                num_intermediate * batch_size, Hp, Wp, -1
            ).permute(0, 3, 1, 2)
            all_intermediate_feats = torch.chunk(all_intermediate_feats_tensor, num_intermediate, dim=0)
            intermediate_features = [ViTEncoderOutput(features=x) for x in all_intermediate_feats]

        # Return the final features and intermediate features accordingly
        if self.intermediates_only:
            return intermediate_features
        else:
            return final_features, intermediate_features


if __name__ == "__main__":
    # Init different versions of the RADIO Encoder
    for model_version in ["radio_v2.5-b", "radio_v2.5-l"]:
        radio_encoder = RADIOEncoder(name="RADIOv2.5", model_version=model_version)

    # Init the E-RADIO Encoder
    eradio_input_shape = (512, 512)
    eradio_encoder = RADIOEncoder(name="E-RADIO", model_version="e-radio_v2", eradio_input_shape=eradio_input_shape)

    print("All RADIO Encoders have been initialized successfully!")

    # Intermediate Feature Returner Tests
    print("Running Intermediate Feature Returner Tests...")

    # Run the intermediate feature returner with last-n index
    radio_intermediate_feature_returner = RADIOIntermediateFeatureReturner(
        name="RADIOv2.5", model_version="radio_v2.5-b", indices=6
    )  # Last 6 layers
    dummy_input = ViTEncoderInput(image=torch.randn(1, 3, 224, 224), data_norm_type="radio")
    output = radio_intermediate_feature_returner(dummy_input)
    assert isinstance(output, list), "Output must be a list of intermediate features"
    assert isinstance(output[0], ViTEncoderOutput), "Output must be a list of ViTEncoderOutput"
    assert len(output) == 6, "Output must have length of intermediate features equal to the number of indices"

    # Run the intermediate feature returner with specific indices
    radio_intermediate_feature_returner = RADIOIntermediateFeatureReturner(
        name="RADIOv2.5", model_version="radio_v2.5-b", indices=[0, 2, 4, 6]
    )  # Specific layers
    dummy_input = ViTEncoderInput(image=torch.randn(1, 3, 224, 224), data_norm_type="radio")
    output = radio_intermediate_feature_returner(dummy_input)
    assert isinstance(output, list), "Output must be a list of intermediate features"
    assert isinstance(output[0], ViTEncoderOutput), "Output must be a list of ViTEncoderOutput"
    assert len(output) == 4, "Output must have length of intermediate features equal to the number of indices"

    # Test the normalizing of intermediate features
    radio_intermediate_feature_returner = RADIOIntermediateFeatureReturner(
        name="RADIOv2.5", model_version="radio_v2.5-b", norm_intermediate=False, intermediates_only=False
    )  # Do not normalize
    dummy_input = ViTEncoderInput(image=torch.randn(1, 3, 224, 224), data_norm_type="radio")
    output = radio_intermediate_feature_returner(dummy_input)
    assert isinstance(output, tuple), "Output must be a tuple with final features and intermediate features"
    assert isinstance(output[0], ViTEncoderOutput), "First element of output must be the final features"
    assert isinstance(output[1], list), "Second element of output must be a list of intermediate features"
    assert isinstance(output[1][0], ViTEncoderOutput), "Output must be a list of ViTEncoderOutput"
    if not isinstance(radio_intermediate_feature_returner.model.model.norm, torch.nn.Identity):
        assert not torch.equal(
            output[0].features, output[1][0].features
        ), "Final features and intermediate features must be different"

    radio_intermediate_feature_returner = RADIOIntermediateFeatureReturner(
        name="RADIOv2.5", model_version="radio_v2.5-b", norm_intermediate=True, intermediates_only=False
    )
    dummy_input = ViTEncoderInput(image=torch.randn(1, 3, 224, 224), data_norm_type="radio")
    output = radio_intermediate_feature_returner(dummy_input)
    assert isinstance(output, tuple), "Output must be a tuple with final features and intermediate features"
    assert isinstance(output[0], ViTEncoderOutput), "First element of output must be the final features"
    assert isinstance(output[1], list), "Second element of output must be a list of intermediate features"
    assert isinstance(output[1][0], ViTEncoderOutput), "Output must be a list of ViTEncoderOutput"
    assert torch.equal(
        output[0].features, output[1][0].features
    ), "Final features and intermediate features must be same"

    print("All Intermediate Feature Returner Tests have passed successfully!")
