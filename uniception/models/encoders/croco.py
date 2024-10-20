"""
Encoder Class for CroCo & DUSt3R
"""

from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from uniception.models.encoders.base import UniCeptionViTEncoderBase, ViTEncoderInput, ViTEncoderOutput
from uniception.models.libs.croco.blocks import Block
from uniception.models.libs.croco.patch_embed import get_patch_embed
from uniception.models.libs.croco.pos_embed import RoPE2D
from uniception.models.utils.intermediate_feature_return import IntermediateFeatureReturner, feature_take_indices


class CroCoEncoder(UniCeptionViTEncoderBase):
    "UniCeption CroCov2 Encoder"

    def __init__(
        self,
        name: str,
        data_norm_type: str,
        patch_embed_cls: str = "PatchEmbedDust3R",
        img_size: Union[int, Tuple[int, int]] = (224, 224),
        patch_size: int = 16,
        enc_embed_dim: int = 1024,
        enc_depth: int = 24,
        enc_num_heads: int = 16,
        mlp_ratio: int = 4,
        norm_layer: Callable = partial(nn.LayerNorm, eps=1e-6),
        pos_embed: str = "RoPE100",
        pretrained_checkpoint_path: str = None,
        override_checkpoint_attributes: bool = False,
        *args,
        **kwargs,
    ):
        """
        References: https://github.com/naver/dust3r, https://github.com/naver/croco

        Args:
            name (str): Name of the encoder.
            data_norm_type (str): Input data normalization type.
            patch_embed_cls (str, optional): The class to use for patch embedding.
                Defaults to 'PatchEmbedDust3R'. Options: ['PatchEmbedCroCo', 'PatchEmbedDust3R', 'ManyAR_PatchEmbed'].
            img_size (int, optional): The size of the input image. Defaults to 224.
            patch_size (int, optional): The size of the patches to divide the image into. Defaults to 16.
            enc_embed_dim (int, optional): The dimension of the encoder's embedding. Defaults to 768.
            enc_depth (int, optional): The number of encoder layers/transformer blocks. Defaults to 12.
            enc_num_heads (int, optional): The number of encoder heads. Defaults to 12.
            mlp_ratio (int, optional): The MLP ratio used for the CroCo encoder transformer. Defaults to 4.
            norm_layer (nn.Module, optional): The normalization layer to use in the transformer. Defaults to nn.LayerNorm with eps=1e-6.
            pos_embed (str, optional): Positional Embedding. Defaults to 'RoPE100'. Options: ['RoPEfreq'].
            pretrained_checkpoint_path (str, optional): Path to the pretrained checkpoint. Defaults to None.
        """
        # Init the base class
        super().__init__(
            name=name,
            data_norm_type=data_norm_type,
            patch_size=patch_size,
            *args,
            **kwargs,
        )

        # Init the CroCo Encoder specific attributes
        self.patch_embed_cls = patch_embed_cls
        self.img_size = img_size
        self.enc_embed_dim = enc_embed_dim
        self.enc_depth = enc_depth
        self.enc_num_heads = enc_num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.override_checkpoint_attributes = override_checkpoint_attributes

        # Init the positional embedding
        self.pos_embed = pos_embed
        if pos_embed.startswith("RoPE"):  # eg RoPE100
            self.enc_pos_embed = None  # nothing to add in the encoder with RoPE
            self.dec_pos_embed = None  # nothing to add in the decoder with RoPE
            if RoPE2D is None:
                raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(pos_embed[len("RoPE") :])
            self.rope = RoPE2D(freq=freq)
        else:
            raise NotImplementedError("Unknown pos_embed " + pos_embed)

        # Init the patch embedding
        self._set_patch_embed(img_size, patch_size, enc_embed_dim)

        # Init the encoder
        self._set_encoder(enc_depth, enc_embed_dim, enc_num_heads, mlp_ratio, norm_layer, self.rope)

        # Initialize random weights
        self.initialize_weights()

        # Load the pretrained CroCo checkpoint if provided
        if pretrained_checkpoint_path:
            print(f"Loading pretrained CroCo checkpoint from {pretrained_checkpoint_path}")
            ckpt = torch.load(pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"]))
            if not override_checkpoint_attributes:
                ckpt_data_norm_type = ckpt["data_norm_type"]
                ckpt_patch_embed_cls = ckpt["patch_embed_cls"]
                assert (
                    data_norm_type == ckpt_data_norm_type
                ), f"Data normalization type {data_norm_type} does not match the checkpoint {ckpt_data_norm_type}."
                assert (
                    patch_embed_cls == ckpt_patch_embed_cls
                ), f"Patch embedding class {patch_embed_cls} does not match the checkpoint {ckpt_patch_embed_cls}."
        else:
            print("No pretrained checkpoint provided. Randomly initializing the CroCo encoder.")

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        "Set the patch embedding scheme"
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def _set_encoder(self, enc_depth, enc_embed_dim, enc_num_heads, mlp_ratio, norm_layer, rope):
        "Set the encoder"
        self.enc_blocks = nn.ModuleList(
            [
                Block(enc_embed_dim, enc_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, rope=rope)
                for _ in range(enc_depth)
            ]
        )
        self.enc_norm = norm_layer(enc_embed_dim)

    def initialize_weights(self):
        "Initialize the weights of the patch embedding and the transformer encoder"
        # Patch embedding
        self.patch_embed._init_weights()
        # Linears and layer norms
        self.apply(self._init_weights)

    def _init_weights(self, m):
        "Initialize the transformer encoder weights"
        if isinstance(m, nn.Linear):
            # We use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, encoder_input: ViTEncoderInput) -> ViTEncoderOutput:
        """
        CroCov2 Encoder Forward Pass

        Args:
            encoder_input (ViTEncoderInput): Input data for the encoder. Input data must contain image normalization type and normalized image tensor.

        Returns:
            ViTEncoderOutput: Output data from the encoder.
        """
        # Check image normalization type
        self._check_data_normalization_type(encoder_input.data_norm_type)

        # Get the true shape of the image for landscape/portrait mode check in patch embedding
        batch_size, _, height, width = encoder_input.image.shape
        if hasattr(encoder_input, "true_shape"):
            true_shape = encoder_input.true_shape
        else:
            true_shape = torch.tensor([height, width])[None].repeat(batch_size, 1)

        # Embed the image into patches
        features, pos = self.patch_embed(encoder_input.image, true_shape=true_shape)

        # Now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            features = blk(features, pos)
        features = self.enc_norm(features)

        # Resize the features to the expected shape
        # (B x Num_patches x Embed_dim) -> (B x Embed_dim x H / Patch_Size x W / Patch_Size)
        features = features.permute(0, 2, 1)
        features = features.reshape(
            -1, self.enc_embed_dim, height // self.patch_size, width // self.patch_size
        ).contiguous()

        return ViTEncoderOutput(features=features)


class CroCoIntermediateFeatureReturner(CroCoEncoder, IntermediateFeatureReturner):
    "Intermediate Feature Returner for UniCeption CroCo Encoder"

    def __init__(
        self,
        name: str,
        data_norm_type: str,
        patch_embed_cls: str = "PatchEmbedDust3R",
        img_size: Union[int, Tuple[int, int]] = (224, 224),
        patch_size: int = 16,
        enc_embed_dim: int = 1024,
        enc_depth: int = 24,
        enc_num_heads: int = 16,
        mlp_ratio: int = 4,
        norm_layer: Callable = partial(nn.LayerNorm, eps=1e-6),
        pos_embed: str = "RoPE100",
        pretrained_checkpoint_path: str = None,
        indices: Optional[Union[int, List[int]]] = None,
        norm_intermediate: bool = True,
        stop_early: bool = False,
        intermediates_only: bool = True,
        *args,
        **kwargs,
    ):
        """
        Intermediate Feature Returner for the CroCo Encoder.

        Args:
            name (str): Name of the encoder.
            data_norm_type (str): Input data normalization type.
            patch_embed_cls (str, optional): The class to use for patch embedding.
                Defaults to 'PatchEmbedDust3R'. Options: ['PatchEmbedCroCo', 'PatchEmbedDust3R', 'ManyAR_PatchEmbed'].
            img_size (int, optional): The size of the input image. Defaults to 224.
            patch_size (int, optional): The size of the patches to divide the image into. Defaults to 16.
            enc_embed_dim (int, optional): The dimension of the encoder's embedding. Defaults to 768.
            enc_depth (int, optional): The number of encoder layers/transformer blocks. Defaults to 12.
            enc_num_heads (int, optional): The number of encoder heads. Defaults to 12.
            mlp_ratio (int, optional): The MLP ratio used for the CroCo encoder transformer. Defaults to 4.
            norm_layer (nn.Module, optional): The normalization layer to use in the transformer. Defaults to nn.LayerNorm with eps=1e-6.
            pos_embed (str, optional): Positional Embedding. Defaults to 'RoPE100'. Options: ['cosine', 'RoPE100'].
            pretrained_checkpoint_path (str, optional): Path to the pretrained checkpoint. Defaults to None.
            indices (Optional[Union[int, List[int]]], optional): Indices of the layers to return. Defaults to None. Options:
            - None: Return all intermediate layers.
            - int: Return the last n layers.
            - List[int]: Return the intermediate layers at the specified indices.
            norm_intermediate (bool, optional): Whether to normalize the intermediate features. Defaults to True.
            stop_early (bool, optional): Whether to stop early. Defaults to False.
            intermediates_only (bool, optional): Whether to return only the intermediate features. Defaults to True.
        """
        # Init the base classes
        CroCoEncoder.__init__(
            self,
            name=name,
            data_norm_type=data_norm_type,
            patch_embed_cls=patch_embed_cls,
            img_size=img_size,
            patch_size=patch_size,
            enc_embed_dim=enc_embed_dim,
            enc_depth=enc_depth,
            enc_num_heads=enc_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            pos_embed=pos_embed,
            pretrained_checkpoint_path=pretrained_checkpoint_path,
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

    def forward(
        self, encoder_input: ViTEncoderInput
    ) -> Union[List[ViTEncoderOutput], Tuple[ViTEncoderOutput, List[ViTEncoderOutput]]]:
        """
        CroCov2 Encoder Forward Pass with Intermediate Feature Return

        Args:
            encoder_input (ViTEncoderInput): Input data for the encoder. Input data must contain image normalization type and normalized image tensor.

        Returns:
            Union[List[ViTEncoderOutput], Tuple[ViTEncoderOutput, List[ViTEncoderOutput]]]: Output data from the encoder.
                If `intermediates_only` is True, returns a list of intermediate features.
                Otherwise, returns a tuple with the final features and a list of intermediate features.
        """
        # Check image normalization type
        self._check_data_normalization_type(encoder_input.data_norm_type)

        # Get the true shape of the image for landscape/portrait mode check in patch embedding
        batch_size, _, height, width = encoder_input.image.shape
        if hasattr(encoder_input, "true_shape"):
            true_shape = encoder_input.true_shape
        else:
            true_shape = torch.tensor([height, width])[None].repeat(batch_size, 1)

        # Embed the image into patches
        features, pos = self.patch_embed(encoder_input.image, true_shape=true_shape)

        # Get indices of the intermediate features to return
        intermediate_features = []
        take_indices, max_index = feature_take_indices(len(self.enc_blocks), self.indices)

        # Get the blocks based on early stopping
        if torch.jit.is_scripting() or not self.stop_early:  # can't slice blocks in torchscript
            blocks = self.enc_blocks
        else:
            blocks = self.enc_blocks[: max_index + 1]

        # Now apply the transformer encoder and normalization
        for blk_idx, blk in enumerate(blocks):
            features = blk(features, pos)
            if blk_idx in take_indices:
                # Normalize intermediates with final norm layer if enabled
                intermediate_features.append(self.enc_norm(features) if self.norm_intermediate else features)

        # Reshape the intermediate features and convert to ViTEncoderOutput class
        intermediate_features = [
            intermediate.permute(0, 2, 1)
            .reshape(-1, self.enc_embed_dim, height // self.patch_size, width // self.patch_size)
            .contiguous()
            for intermediate in intermediate_features
        ]
        intermediate_features = [ViTEncoderOutput(features=intermediate) for intermediate in intermediate_features]

        # Return only the intermediate features if enabled
        if self.intermediates_only:
            return intermediate_features

        # Normalize and reshape the final features
        features = self.enc_norm(features)
        # Resize the features to the expected shape
        # (B x Num_patches x Embed_dim) -> (B x Embed_dim x H / Patch_Size x W / Patch_Size)
        features = features.permute(0, 2, 1)
        features = features.reshape(
            -1, self.enc_embed_dim, height // self.patch_size, width // self.patch_size
        ).contiguous()
        final_features = ViTEncoderOutput(features=features)

        return final_features, intermediate_features


if __name__ == "__main__":
    # Init the pre-trained CroCo Encoder
    pretrained_checkpoint_path = "../../../checkpoints/encoders/CroCo_Encoder_224.pth"
    croco_encoder = CroCoEncoder(
        name="croco",
        data_norm_type="croco",
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        patch_embed_cls="PatchEmbedCroCo",
    )

    # Init the pre-trained DUSt3R CroCo Encoder
    pretrained_checkpoint_path = "../../../checkpoints/encoders/CroCo_Encoder_224_DUSt3R_linear.pth"
    dust3r_encoder = CroCoEncoder(
        name="dust3r_224",
        data_norm_type="dust3r",
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        patch_embed_cls="PatchEmbedDust3R",
    )

    # Init the pre-trained DUSt3R 512 linear CroCo Encoder
    pretrained_checkpoint_path = "../../../checkpoints/encoders/CroCo_Encoder_512_DUSt3R_linear.pth"
    dust3r_encoder_512 = CroCoEncoder(
        name="dust3r_512",
        data_norm_type="dust3r",
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        patch_embed_cls="ManyAR_PatchEmbed",
        img_size=(512, 512),
    )

    # Init the pre-trained DUSt3R 512 DPT CroCo Encoder
    pretrained_checkpoint_path = "../../../checkpoints/encoders/CroCo_Encoder_512_DUSt3R_dpt.pth"
    dust3r_encoder_512_dpt = CroCoEncoder(
        name="dust3r_512_dpt",
        data_norm_type="dust3r",
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        patch_embed_cls="ManyAR_PatchEmbed",
        img_size=(512, 512),
    )

    # Init the MASt3R 512 CroCo Encoder
    pretrained_checkpoint_path = "../../../checkpoints/encoders/CroCo_Encoder_512_MASt3R.pth"
    mast3r_encoder_512 = CroCoEncoder(
        name="mast3r_512",
        data_norm_type="dust3r",
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        patch_embed_cls="ManyAR_PatchEmbed",
        img_size=(512, 512),
    )

    print("All CroCo & DUSt3R Encoders have been initialized successfully!")

    # Intermediate Feature Returner Tests
    print("Running Intermediate Feature Returner Tests...")
    pretrained_checkpoint_path = "../../../checkpoints/encoders/CroCo_Encoder_512_DUSt3R_dpt.pth"

    # Run the intermediate feature returner with last-n index
    dust3r_intermediate_feature_returner = CroCoIntermediateFeatureReturner(
        name="dust3r_512_dpt",
        data_norm_type="dust3r",
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        patch_embed_cls="ManyAR_PatchEmbed",
        img_size=(512, 512),
        indices=6,  # Last 6 layers
    )
    dummy_input = ViTEncoderInput(image=torch.randn(1, 3, 224, 224), data_norm_type="dust3r")
    output = dust3r_intermediate_feature_returner(dummy_input)
    assert isinstance(output, list), "Output must be a list of intermediate features"
    assert isinstance(output[0], ViTEncoderOutput), "Output must be a list of ViTEncoderOutput"
    assert len(output) == 6, "Output must have length of intermediate features equal to the number of indices"

    # Run the intermediate feature returner with specific indices
    dust3r_intermediate_feature_returner = CroCoIntermediateFeatureReturner(
        name="dust3r_512_dpt",
        data_norm_type="dust3r",
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        patch_embed_cls="ManyAR_PatchEmbed",
        img_size=(512, 512),
        indices=[0, 2, 4, 6],  # Specific layers
    )
    dummy_input = ViTEncoderInput(image=torch.randn(1, 3, 224, 224), data_norm_type="dust3r")
    output = dust3r_intermediate_feature_returner(dummy_input)
    assert isinstance(output, list), "Output must be a list of intermediate features"
    assert isinstance(output[0], ViTEncoderOutput), "Output must be a list of ViTEncoderOutput"
    assert len(output) == 4, "Output must have length of intermediate features equal to the number of indices"

    # Test the normalizing of intermediate features
    dust3r_intermediate_feature_returner = CroCoIntermediateFeatureReturner(
        name="dust3r_512_dpt",
        data_norm_type="dust3r",
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        patch_embed_cls="ManyAR_PatchEmbed",
        img_size=(512, 512),
        indices=[-1],
        norm_intermediate=False,
        intermediates_only=False,
    )
    dummy_input = ViTEncoderInput(image=torch.randn(1, 3, 224, 224), data_norm_type="dust3r")
    output = dust3r_intermediate_feature_returner(dummy_input)
    assert isinstance(output, tuple), "Output must be a tuple with final features and intermediate features"
    assert isinstance(output[0], ViTEncoderOutput), "First element of output must be the final features"
    assert isinstance(output[1], list), "Second element of output must be a list of intermediate features"
    assert isinstance(output[1][0], ViTEncoderOutput), "Output must be a list of ViTEncoderOutput"
    if not isinstance(dust3r_intermediate_feature_returner.enc_norm, torch.nn.Identity):
        assert not torch.equal(
            output[0].features, output[1][0].features
        ), "Final features and intermediate features must be different"

    dust3r_intermediate_feature_returner = CroCoIntermediateFeatureReturner(
        name="dust3r_512_dpt",
        data_norm_type="dust3r",
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        patch_embed_cls="ManyAR_PatchEmbed",
        img_size=(512, 512),
        indices=[-1],
        norm_intermediate=True,
        intermediates_only=False,
    )
    dummy_input = ViTEncoderInput(image=torch.randn(1, 3, 224, 224), data_norm_type="dust3r")
    output = dust3r_intermediate_feature_returner(dummy_input)
    assert isinstance(output, tuple), "Output must be a tuple with final features and intermediate features"
    assert isinstance(output[0], ViTEncoderOutput), "First element of output must be the final features"
    assert isinstance(output[1], list), "Second element of output must be a list of intermediate features"
    assert isinstance(output[1][0], ViTEncoderOutput), "Output must be a list of ViTEncoderOutput"
    assert torch.equal(
        output[0].features, output[1][0].features
    ), "Final features and intermediate features must be same"

    print("All Intermediate Feature Returner Tests have passed successfully!")
