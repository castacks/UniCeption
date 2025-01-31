"""
UniCeption Global-Attention Transformer for Information Sharing
"""

from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from uniception.models.info_sharing.base import (
    MultiViewTransformerInput,
    MultiViewTransformerOutput,
    UniCeptionInfoSharingBase,
)
from uniception.models.utils.intermediate_feature_return import IntermediateFeatureReturner, feature_take_indices
from uniception.models.utils.positional_encoding import PositionGetter
from uniception.models.utils.transformer_blocks import Mlp, SelfAttentionBlock


class MultiViewGlobalAttentionTransformer(UniCeptionInfoSharingBase):
    "UniCeption Multi-View Global-Attention Transformer for information sharing across image features from different views."

    def __init__(
        self,
        name: str,
        input_embed_dim: int,
        num_views: int,
        use_rand_idx_pe_for_non_reference_views: bool = False,
        size: Optional[str] = None,
        depth: int = 12,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Union[Type[nn.Module], Callable[..., nn.Module]] = partial(nn.LayerNorm, eps=1e-6),
        mlp_layer: Type[nn.Module] = Mlp,
        custom_positional_encoding: Optional[Callable] = None,
        pretrained_checkpoint_path: Optional[str] = None,
        gradient_checkpointing: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialize the Multi-View Global-Attention Transformer for information sharing across image features from different views.

        Args:
            input_embed_dim (int): Dimension of input embeddings.
            num_views (int): Maximum number of views for positional encoding.
            use_rand_idx_pe_for_non_reference_views (bool): Whether to use random index positional encoding for non-reference views.
            size (str): String to indicate interpretable size of the transformer (for e.g., base, large, ...). (default: None)
            depth (int): Number of transformer layers. (default: 12, base size)
            dim (int): Dimension of the transformer. (default: 768, base size)
            num_heads (int): Number of attention heads. (default: 12, base size)
            mlp_ratio (float): Ratio of hidden to input dimension in MLP (default: 4.)
            qkv_bias (bool): Whether to include bias in qkv projection (default: True)
            qk_norm (bool): Whether to normalize q and k (default: False)
            proj_drop (float): Dropout rate for output (default: 0.)
            attn_drop (float): Dropout rate for attention weights (default: 0.)
            init_values (float): Initial value for LayerScale gamma (default: None)
            drop_path (float): Dropout rate for stochastic depth (default: 0.)
            act_layer (nn.Module): Activation layer (default: nn.GELU)
            norm_layer (nn.Module): Normalization layer (default: nn.LayerNorm)
            mlp_layer (nn.Module): MLP layer (default: Mlp)
            custom_positional_encoding (Callable): Custom positional encoding function (default: None)
            pretrained_checkpoint_path (str, optional): Path to the pretrained checkpoint. (default: None)
        """
        # Initialize the base class
        super().__init__(name=name, size=size, *args, **kwargs)

        # Initialize the specific attributes of the transformer
        self.input_embed_dim = input_embed_dim
        self.max_num_views = num_views
        self.use_rand_idx_pe_for_non_reference_views = use_rand_idx_pe_for_non_reference_views
        self.depth = depth
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.proj_drop = proj_drop
        self.attn_drop = attn_drop
        self.init_values = init_values
        self.drop_path = drop_path
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.mlp_layer = mlp_layer
        self.custom_positional_encoding = custom_positional_encoding
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.gradient_checkpointing = gradient_checkpointing

        # Initialize the projection layer for input embeddings
        if self.input_embed_dim != self.dim:
            self.proj_embed = nn.Linear(self.input_embed_dim, self.dim, bias=True)
        else:
            self.proj_embed = nn.Identity()

        # Initialize the self-attention blocks which ingest all views at once
        self.self_attention_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=self.dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    qk_norm=self.qk_norm,
                    proj_drop=self.proj_drop,
                    attn_drop=self.attn_drop,
                    init_values=self.init_values,
                    drop_path=self.drop_path,
                    act_layer=self.act_layer,
                    norm_layer=self.norm_layer,
                    mlp_layer=self.mlp_layer,
                    custom_positional_encoding=self.custom_positional_encoding,
                )
                for _ in range(self.depth)
            ]
        )

        # Initialize the final normalization layer
        self.norm = self.norm_layer(self.dim)

        # Initialize the position getter for patch positions if required
        if self.custom_positional_encoding is not None:
            self.position_getter = PositionGetter()

        # Initialize the positional encoding table for the different views
        self.register_buffer(
            "view_pos_table",
            self._get_sinusoid_encoding_table(self.max_num_views, self.dim, 10000),
        )

        # Initialize random weights
        self.initialize_weights()

        # Load pretrained weights if provided
        if self.pretrained_checkpoint_path is not None:
            print(
                f"Loading pretrained multi-view global-attention transformer weights from {self.pretrained_checkpoint_path} ..."
            )
            ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"]))
        
        # apply gradient checkpointing if enabled
        if self.gradient_checkpointing:
            for i, block in enumerate(self.self_attention_blocks):
                self.self_attention_blocks[i] = self.wrap_module_with_gradient_checkpointing(block)

    def _get_sinusoid_encoding_table(self, n_position, d_hid, base):
        "Sinusoid position encoding table"

        def get_position_angle_vec(position):
            return [position / np.power(base, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table)

    def initialize_weights(self):
        "Initialize weights of the transformer."
        # Linears and layer norms
        self.apply(self._init_weights)

    def _init_weights(self, m):
        "Initialize the transformer linear and layer norm weights."
        if isinstance(m, nn.Linear):
            # We use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        model_input: MultiViewTransformerInput,
    ) -> MultiViewTransformerOutput:
        """
        Forward interface for the Multi-View Global-Attention Transformer.

        Args:
            model_input (MultiViewTransformerInput): Input to the model.
                Expects the features to be a list of size (batch, input_embed_dim, height, width),
                where each entry corresponds to a different view.

        Returns:
            MultiViewTransformerOutput: Output of the model post information sharing.
        """
        # Check that the number of views matches the input and the features are of expected shape
        assert (
            len(model_input.features) <= self.max_num_views
        ), f"Expected less than {self.max_num_views} views, got {len(model_input.features)}"
        assert all(
            view_features.shape[1] == self.input_embed_dim for view_features in model_input.features
        ), f"All views must have input dimension {self.input_embed_dim}"
        assert all(
            view_features.ndim == 4 for view_features in model_input.features
        ), "All views must have 4 dimensions (N, C, H, W)"

        # Initialize the multi-view features from the model input and number of views for current input
        multi_view_features = model_input.features
        num_of_views = len(multi_view_features)
        batch_size, _, height, width = multi_view_features[0].shape
        num_of_tokens_per_view = height * width

        # Stack the multi-view features (N, C, H, W) to (N, V, C, H, W) (assumes all V views have same shape)
        multi_view_features = torch.stack(multi_view_features, dim=1)

        # Resize the multi-view features from NVCHW to NLC, where L = V * H * W
        multi_view_features = multi_view_features.permute(0, 1, 3, 4, 2)  # (N, V, H, W, C)
        multi_view_features = multi_view_features.reshape(
            batch_size, num_of_views * height * width, self.input_embed_dim
        ).contiguous()

        # Project input features to the transformer dimension
        multi_view_features = self.proj_embed(multi_view_features)

        # Create patch positions for each view if custom positional encoding is used
        if self.custom_positional_encoding is not None:
            multi_view_positions = [
                self.position_getter(batch_size, height, width, multi_view_features.device)
            ] * num_of_views  # List of length V, where each tensor is (N, H * W, C)
            multi_view_positions = torch.cat(multi_view_positions, dim=1)  # (N, V * H * W, C)
        else:
            multi_view_positions = [None] * num_of_views

        # Add positional encoding for reference view (idx 0)
        ref_view_pe = self.view_pos_table[0].clone().detach()
        ref_view_pe = ref_view_pe.reshape((1, 1, self.dim))
        ref_view_pe = ref_view_pe.repeat(batch_size, num_of_tokens_per_view, 1)
        ref_view_features = multi_view_features[:, :num_of_tokens_per_view, :]
        ref_view_features = ref_view_features + ref_view_pe

        # Add positional encoding for non-reference views (sequential indices starting from idx 1 or random indices which are uniformly sampled)
        if self.use_rand_idx_pe_for_non_reference_views:
            non_ref_view_pe_indices = torch.randint(low=1, high=self.max_num_views, size=(num_of_views - 1,))
        else:
            non_ref_view_pe_indices = torch.arange(1, num_of_views)
        non_ref_view_pe = self.view_pos_table[non_ref_view_pe_indices].clone().detach()
        non_ref_view_pe = non_ref_view_pe.reshape((1, num_of_views - 1, self.dim))
        non_ref_view_pe = non_ref_view_pe.repeat_interleave(num_of_tokens_per_view, dim=1)
        non_ref_view_pe = non_ref_view_pe.repeat(batch_size, 1, 1)
        non_ref_view_features = multi_view_features[:, num_of_tokens_per_view:, :]
        non_ref_view_features = non_ref_view_features + non_ref_view_pe

        # Concatenate the reference and non-reference view features
        multi_view_features = torch.cat([ref_view_features, non_ref_view_features], dim=1)

        # Loop over the depth of the transformer
        for depth_idx in range(self.depth):
            # Apply the self-attention block and update the multi-view features
            multi_view_features = self.self_attention_blocks[depth_idx](multi_view_features, multi_view_positions)

        # Normalize the output features
        output_multi_view_features = self.norm(multi_view_features)

        # Reshape the output multi-view features (N, V * H * W, C) back to (N, V, C, H, W)
        output_multi_view_features = output_multi_view_features.reshape(
            batch_size, num_of_views, height, width, self.dim
        )  # (N, V, H, W, C)
        output_multi_view_features = output_multi_view_features.permute(0, 1, 4, 2, 3).contiguous()

        # Split the output multi-view features into separate views
        output_multi_view_features = output_multi_view_features.split(1, dim=1)
        output_multi_view_features = [
            output_view_features.squeeze(dim=1) for output_view_features in output_multi_view_features
        ]

        # Return the output multi-view features
        return MultiViewTransformerOutput(features=output_multi_view_features)


class MultiViewGlobalAttentionTransformerIFR(MultiViewGlobalAttentionTransformer, IntermediateFeatureReturner):
    "Intermediate Feature Returner for UniCeption Multi-View Global-Attention Transformer"

    def __init__(
        self,
        name: str,
        input_embed_dim: int,
        num_views: int,
        use_rand_idx_pe_for_non_reference_views: bool = False,
        size: Optional[str] = None,
        depth: int = 12,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        mlp_layer: nn.Module = Mlp,
        custom_positional_encoding: Callable = None,
        pretrained_checkpoint_path: str = None,
        indices: Optional[Union[int, List[int]]] = None,
        norm_intermediate: bool = True,
        intermediates_only: bool = False,
        gradient_checkpointing: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialize the Multi-View Global-Attention Transformer for information sharing across image features from different views.
        Extends the base class to return intermediate features.

        Args:
            input_embed_dim (int): Dimension of input embeddings.
            max_num_views (int): Maximum number of views for positional encoding.
            use_rand_idx_pe_for_non_reference_views (bool): Whether to use random index positional encoding for non-reference views.
            size (str): String to indicate interpretable size of the transformer (for e.g., base, large, ...). (default: None)
            depth (int): Number of transformer layers. (default: 12, base size)
            dim (int): Dimension of the transformer. (default: 768, base size)
            num_heads (int): Number of attention heads. (default: 12, base size)
            mlp_ratio (float): Ratio of hidden to input dimension in MLP (default: 4.)
            qkv_bias (bool): Whether to include bias in qkv projection (default: False)
            qk_norm (bool): Whether to normalize q and k (default: False)
            proj_drop (float): Dropout rate for output (default: 0.)
            attn_drop (float): Dropout rate for attention weights (default: 0.)
            init_values (float): Initial value for LayerScale gamma (default: None)
            drop_path (float): Dropout rate for stochastic depth (default: 0.)
            act_layer (nn.Module): Activation layer (default: nn.GELU)
            norm_layer (nn.Module): Normalization layer (default: nn.LayerNorm)
            mlp_layer (nn.Module): MLP layer (default: Mlp)
            custom_positional_encoding (Callable): Custom positional encoding function (default: None)
            pretrained_checkpoint_path (str, optional): Path to the pretrained checkpoint. (default: None)
            indices (Optional[Union[int, List[int]]], optional): Indices of the layers to return. Defaults to None. Options:
            - None: Return all intermediate layers.
            - int: Return the last n layers.
            - List[int]: Return the intermediate layers at the specified indices.
            norm_intermediate (bool, optional): Whether to normalize the intermediate features. Defaults to True.
            intermediates_only (bool, optional): Whether to return only the intermediate features. Defaults to True.
        """
        # Init the base classes
        MultiViewGlobalAttentionTransformer.__init__(
            self,
            name=name,
            input_embed_dim=input_embed_dim,
            num_views=num_views,
            use_rand_idx_pe_for_non_reference_views=use_rand_idx_pe_for_non_reference_views,
            size=size,
            depth=depth,
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            mlp_layer=mlp_layer,
            custom_positional_encoding=custom_positional_encoding,
            pretrained_checkpoint_path=pretrained_checkpoint_path,
            gradient_checkpointing=gradient_checkpointing,
            *args,
            **kwargs,
        )
        IntermediateFeatureReturner.__init__(
            self,
            indices=indices,
            norm_intermediate=norm_intermediate,
            intermediates_only=intermediates_only,
        )

    def forward(
        self,
        model_input: MultiViewTransformerInput,
    ) -> Union[List[MultiViewTransformerOutput], Tuple[MultiViewTransformerOutput, List[MultiViewTransformerOutput]],]:
        """
        Forward interface for the Multi-View Global-Attention Transformer with Intermediate Feature Return.

        Args:
            model_input (MultiViewTransformerInput): Input to the model.
                Expects the features to be a list of size (batch, input_embed_dim, height, width),
                where each entry corresponds to a different view.

        Returns:
            Union[List[MultiViewTransformerOutput], Tuple[MultiViewTransformerOutput, List[MultiViewTransformerOutput]]]:
                Output of the model post information sharing.
                If intermediates_only is True, returns a list of intermediate outputs.
                If intermediates_only is False, returns a tuple of final output and a list of intermediate outputs.
        """
        # Check that the number of views matches the input and the features are of expected shape
        assert (
            len(model_input.features) <= self.max_num_views
        ), f"Expected {self.num_views} views, got {len(model_input.features)}"
        assert all(
            view_features.shape[1] == self.input_embed_dim for view_features in model_input.features
        ), f"All views must have input dimension {self.input_embed_dim}"
        assert all(
            view_features.ndim == 4 for view_features in model_input.features
        ), "All views must have 4 dimensions (N, C, H, W)"

        # Get the indices of the intermediate features to return
        intermediate_multi_view_features = []
        take_indices, _ = feature_take_indices(self.depth, self.indices)

        # Initialize the multi-view features from the model input and number of views for current input
        multi_view_features = model_input.features
        num_of_views = len(multi_view_features)
        batch_size, _, height, width = multi_view_features[0].shape
        num_of_tokens_per_view = height * width

        # Stack the multi-view features (N, C, H, W) to (N, V, C, H, W) (assumes all V views have same shape)
        multi_view_features = torch.stack(multi_view_features, dim=1)

        # Resize the multi-view features from NVCHW to NLC, where L = V * H * W
        multi_view_features = multi_view_features.permute(0, 1, 3, 4, 2)  # (N, V, H, W, C)
        multi_view_features = multi_view_features.reshape(
            batch_size, num_of_views * height * width, self.input_embed_dim
        ).contiguous()

        # Project input features to the transformer dimension
        multi_view_features = self.proj_embed(multi_view_features)

        # Create patch positions for each view if custom positional encoding is used
        if self.custom_positional_encoding is not None:
            multi_view_positions = [
                self.position_getter(batch_size, height, width, multi_view_features.device)
            ] * num_of_views  # List of length V, where each tensor is (N, H * W, C)
            multi_view_positions = torch.cat(multi_view_positions, dim=1)  # (N, V * H * W, C)
        else:
            multi_view_positions = [None] * num_of_views

        # Add positional encoding for reference view (idx 0)
        ref_view_pe = self.view_pos_table[0].clone().detach()
        ref_view_pe = ref_view_pe.reshape((1, 1, self.dim))
        ref_view_pe = ref_view_pe.repeat(batch_size, num_of_tokens_per_view, 1)
        ref_view_features = multi_view_features[:, :num_of_tokens_per_view, :]
        ref_view_features = ref_view_features + ref_view_pe

        # Add positional encoding for non-reference views (sequential indices starting from idx 1 or random indices which are uniformly sampled)
        if self.use_rand_idx_pe_for_non_reference_views:
            non_ref_view_pe_indices = torch.randint(low=1, high=self.max_num_views, size=(num_of_views - 1,))
        else:
            non_ref_view_pe_indices = torch.arange(1, num_of_views)
        non_ref_view_pe = self.view_pos_table[non_ref_view_pe_indices].clone().detach()
        non_ref_view_pe = non_ref_view_pe.reshape((1, num_of_views - 1, self.dim))
        non_ref_view_pe = non_ref_view_pe.repeat_interleave(num_of_tokens_per_view, dim=1)
        non_ref_view_pe = non_ref_view_pe.repeat(batch_size, 1, 1)
        non_ref_view_features = multi_view_features[:, num_of_tokens_per_view:, :]
        non_ref_view_features = non_ref_view_features + non_ref_view_pe

        # Concatenate the reference and non-reference view features
        multi_view_features = torch.cat([ref_view_features, non_ref_view_features], dim=1)

        # Loop over the depth of the transformer
        for depth_idx in range(self.depth):
            # Apply the self-attention block and update the multi-view features
            multi_view_features = self.self_attention_blocks[depth_idx](multi_view_features, multi_view_positions)
            if depth_idx in take_indices:
                # Normalize the intermediate features with final norm layer if enabled
                intermediate_multi_view_features.append(
                    self.norm(multi_view_features) if self.norm_intermediate else multi_view_features
                )

        # Reshape the intermediate features and convert to MultiViewTransformerOutput class
        for idx in range(len(intermediate_multi_view_features)):
            # Reshape the intermediate multi-view features (N, V * H * W, C) back to (N, V, C, H, W)
            intermediate_multi_view_features[idx] = intermediate_multi_view_features[idx].reshape(
                batch_size, num_of_views, height, width, self.dim
            )  # (N, V, H, W, C)
            intermediate_multi_view_features[idx] = (
                intermediate_multi_view_features[idx].permute(0, 1, 4, 2, 3).contiguous()
            )
            # Split the intermediate multi-view features into separate views
            intermediate_multi_view_features[idx] = intermediate_multi_view_features[idx].split(1, dim=1)
            intermediate_multi_view_features[idx] = [
                intermediate_view_features.squeeze(dim=1)
                for intermediate_view_features in intermediate_multi_view_features[idx]
            ]
            intermediate_multi_view_features[idx] = MultiViewTransformerOutput(
                features=intermediate_multi_view_features[idx]
            )

        # Return only the intermediate features if enabled
        if self.intermediates_only:
            return intermediate_multi_view_features

        # Normalize the output features
        output_multi_view_features = self.norm(multi_view_features)

        # Reshape the output multi-view features (N, V * H * W, C) back to (N, V, C, H, W)
        output_multi_view_features = output_multi_view_features.reshape(
            batch_size, num_of_views, height, width, self.dim
        )  # (N, V, H, W, C)
        output_multi_view_features = output_multi_view_features.permute(0, 1, 4, 2, 3).contiguous()

        # Split the output multi-view features into separate views
        output_multi_view_features = output_multi_view_features.split(1, dim=1)
        output_multi_view_features = [
            output_view_features.squeeze(dim=1) for output_view_features in output_multi_view_features
        ]
        output_multi_view_features = MultiViewTransformerOutput(features=output_multi_view_features)

        return output_multi_view_features, intermediate_multi_view_features


def dummy_positional_encoding(x, xpos):
    "Dummy function for positional encoding of tokens"
    x = x
    xpos = xpos
    return x


if __name__ == "__main__":
    # Init multi-view global-attention transformer with no custom positional encoding and run a forward pass
    for num_views in [2, 3, 4]:
        print(f"Testing MultiViewGlobalAttentionTransformer with {num_views} views ...")
        # Sequential idx based positional encoding
        model = MultiViewGlobalAttentionTransformer(
            name="MV-GAT", input_embed_dim=1024, num_views=1000, use_rand_idx_pe_for_non_reference_views=False
        )
        model_input = [torch.rand(1, 1024, 14, 14) for _ in range(num_views)]
        model_input = MultiViewTransformerInput(features=model_input)
        model_output = model(model_input)
        assert len(model_output.features) == num_views
        assert all(f.shape == (1, model.dim, 14, 14) for f in model_output.features)
        # Random idx based positional encoding
        model = MultiViewGlobalAttentionTransformer(
            name="MV-GAT", input_embed_dim=1024, num_views=1000, use_rand_idx_pe_for_non_reference_views=True
        )
        model_input = [torch.rand(1, 1024, 14, 14) for _ in range(num_views)]
        model_input = MultiViewTransformerInput(features=model_input)
        model_output = model(model_input)
        assert len(model_output.features) == num_views
        assert all(f.shape == (1, model.dim, 14, 14) for f in model_output.features)

    # Init multi-view global-attention transformer with custom positional encoding and run a forward pass
    for num_views in [2, 3, 4]:
        print(f"Testing MultiViewGlobalAttentionTransformer with {num_views} views and custom positional encoding ...")
        model = MultiViewGlobalAttentionTransformer(
            name="MV-GAT",
            input_embed_dim=1024,
            num_views=1000,
            use_rand_idx_pe_for_non_reference_views=True,
            custom_positional_encoding=dummy_positional_encoding,
        )
        model_input = [torch.rand(1, 1024, 14, 14) for _ in range(num_views)]
        model_input = MultiViewTransformerInput(features=model_input)
        model_output = model(model_input)
        assert len(model_output.features) == num_views
        assert all(f.shape == (1, model.dim, 14, 14) for f in model_output.features)

    print("All multi-view global-attention transformers initialized and tested successfully!")

    # Intermediate Feature Returner Tests
    print("Running Intermediate Feature Returner Tests ...")

    # Run the intermediate feature returner with last-n index
    model_intermediate_feature_returner = MultiViewGlobalAttentionTransformerIFR(
        name="MV-GAT-IFR",
        input_embed_dim=1024,
        num_views=1000,
        use_rand_idx_pe_for_non_reference_views=True,
        indices=6,  # Last 6 layers
    )
    model_input = [torch.rand(1, 1024, 14, 14) for _ in range(2)]
    model_input = MultiViewTransformerInput(features=model_input)
    output = model_intermediate_feature_returner(model_input)
    assert isinstance(output, tuple)
    assert isinstance(output[0], MultiViewTransformerOutput)
    assert len(output[1]) == 6
    assert all(isinstance(intermediate, MultiViewTransformerOutput) for intermediate in output[1])
    assert len(output[1][0].features) == 2

    # Run the intermediate feature returner with specific indices
    model_intermediate_feature_returner = MultiViewGlobalAttentionTransformerIFR(
        name="MV-GAT-IFR",
        input_embed_dim=1024,
        num_views=1000,
        use_rand_idx_pe_for_non_reference_views=True,
        indices=[0, 2, 4, 6],  # Specific indices
    )
    model_input = [torch.rand(1, 1024, 14, 14) for _ in range(2)]
    model_input = MultiViewTransformerInput(features=model_input)
    output = model_intermediate_feature_returner(model_input)
    assert isinstance(output, tuple)
    assert isinstance(output[0], MultiViewTransformerOutput)
    assert len(output[1]) == 4
    assert all(isinstance(intermediate, MultiViewTransformerOutput) for intermediate in output[1])
    assert len(output[1][0].features) == 2

    # Test the normalizing of intermediate features
    model_intermediate_feature_returner = MultiViewGlobalAttentionTransformerIFR(
        name="MV-GAT-IFR",
        input_embed_dim=1024,
        num_views=1000,
        use_rand_idx_pe_for_non_reference_views=True,
        indices=[-1],  # Last layer
        norm_intermediate=False,  # Disable normalization
    )
    model_input = [torch.rand(1, 1024, 14, 14) for _ in range(2)]
    model_input = MultiViewTransformerInput(features=model_input)
    output = model_intermediate_feature_returner(model_input)
    for view_idx in range(2):
        assert not torch.equal(
            output[0].features[view_idx], output[1][-1].features[view_idx]
        ), "Final features and intermediate features (last layer) must be different."

    model_intermediate_feature_returner = MultiViewGlobalAttentionTransformerIFR(
        name="MV-GAT-IFR",
        input_embed_dim=1024,
        num_views=1000,
        use_rand_idx_pe_for_non_reference_views=True,
        indices=[-1],  # Last layer
        norm_intermediate=True,
    )
    model_input = [torch.rand(1, 1024, 14, 14) for _ in range(2)]
    model_input = MultiViewTransformerInput(features=model_input)
    output = model_intermediate_feature_returner(model_input)
    for view_idx in range(2):
        assert torch.equal(
            output[0].features[view_idx], output[1][-1].features[view_idx]
        ), "Final features and intermediate features (last layer) must be same."

    print("All Intermediate Feature Returner Tests passed!")
