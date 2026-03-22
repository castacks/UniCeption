from functools import partial
from typing import Callable, Type, Union, Tuple, Optional
from itertools import repeat
import collections.abc

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, list_repo_files
from torch.utils.checkpoint import checkpoint

from uniception.models.encoders.base import (
    UniCeptionViTEncoderBase,
    ViTEncoderInput,
    ViTEncoderOutput,
)


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


class LayerScale(nn.Module):
    """Layer scale module.

    References:
      - https://arxiv.org/abs/2103.17239
    """

    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        """Initialize LayerScale module.

        Args:
            dim: Dimension.
            init_values: Initial value for scaling.
            inplace: If True, perform inplace operations.
        """
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer scaling."""
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class SelfAttention(nn.Module):
    """Standard Multi-head Self Attention module with QKV projection.

    This module implements the standard multi-head attention mechanism used in transformers.
    It supports both the fused attention implementation (scaled_dot_product_attention) for
    efficiency when available, and a manual implementation otherwise. The module includes
    options for QK normalization, attention dropout, and projection dropout.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        """Initialize the Attention module.

        Args:
            dim: Input dimension of the token embeddings
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in the query, key, value projections
            qk_norm: Whether to apply normalization to query and key vectors
            proj_bias: Whether to use bias in the output projection
            attn_drop: Dropout rate applied to the attention weights
            proj_drop: Dropout rate applied after the output projection
            norm_layer: Normalization layer constructor for QK normalization if enabled
        """
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        if qk_norm or scale_norm:
            assert norm_layer is not None, (
                "norm_layer must be provided if qk_norm or scale_norm is True"
            )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(dim) if scale_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, L, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        x = x.transpose(1, 2).reshape(B, L, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        bias: bool = True,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def _init_img_size(self, img_size: Union[int, Tuple[int, int]]):
        assert self.patch_size
        if img_size is None:
            return None, None, None
        img_size = to_2tuple(img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)

        x = self.norm(x)
        return x

class SelfAttentionBlock(nn.Module):
    """Transformer block with pre-normalization."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_attn_norm: bool = False,
        scale_mlp_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        """Initialize Block.

        Args:
            dim: Number of input channels.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            qk_norm: If True, apply normalization to query and key.
            proj_bias: If True, add bias to output projection.
            proj_drop: Projection dropout rate.
            attn_drop: Attention dropout rate.
            init_values: Initial values for layer scale.
            drop_path: Stochastic depth rate.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            mlp_layer: MLP layer.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            scale_norm=scale_attn_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer if scale_mlp_norm else None,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class PixioEncoder(UniCeptionViTEncoderBase):
    def __init__(
        self,
        name: str,
        data_norm_type: str,
        img_size: int = 256,
        patch_size: int = 16,
        embed_dim: int = 1280,
        depth: int = 32,
        in_chans: int = 3,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        n_cls_tokens: int = 8,
        norm_layer: Union[Type[nn.Module], Callable[..., nn.Module]] = partial(
            nn.LayerNorm, eps=1e-6
        ),
        hf_model_name: str = "facebook/pixio-vith16",
        gradient_checkpointing: bool = True,
        *args,
        **kwargs,
    ):
        """
        Base class for all Vision Transformer encoders in UniCeption.
        """
        super().__init__(
            name=name,
            data_norm_type=data_norm_type,
            patch_size=patch_size,
            *args,
            **kwargs,
        )

        self.n_cls_tokens = n_cls_tokens

        self.patch_size = patch_size

        self.enc_embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, self.enc_embed_dim
        )

        self.cls_token = nn.Parameter(torch.zeros(1, n_cls_tokens, self.enc_embed_dim))

        self.pos_embed = nn.Parameter(
            torch.zeros(
                1, self.patch_embed.num_patches + n_cls_tokens, self.enc_embed_dim
            )
        )

        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    self.enc_embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    mlp_layer=Mlp,
                )
                for _ in range(depth)
            ]
        )

        self.norm = norm_layer(self.enc_embed_dim)

        ckpt_path = self.get_pth_file(repo_id=hf_model_name)
        print(f"Loading pretrained Pixio Encoder from {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, weights_only=False)
        print(self.load_state_dict(ckpt, strict=False))

        if gradient_checkpointing:
            for i in range(len(self.blocks)):
                self.blocks[i] = self.wrap_module_with_gradient_checkpointing(
                    self.blocks[i]
                )
    def wrap_module_with_gradient_checkpointing(self, module: nn.Module):
        class _CheckpointingWrapper(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, *args, **kwargs):
                return checkpoint(
                    self.inner.forward, *args, use_reentrant=False, **kwargs
                )

        return _CheckpointingWrapper(module)

    def _interpolate_pos_emb(self, x):
        """
        Interpolate the positional embeddings to match the input x.
        """
        assert (
            x.shape[-2] % self.patch_embed.patch_size[0] == 0
        ), f"height {x.shape[-2]} must be divisible by patch size {self.patch_embed.patch_size[0]}"
        assert (
            x.shape[-1] % self.patch_embed.patch_size[1] == 0
        ), f"width {x.shape[-1]} must be divisible by patch size {self.patch_embed.patch_size[1]}"

        H = x.shape[-2] // self.patch_embed.patch_size[0]
        W = x.shape[-1] // self.patch_embed.patch_size[1]

        cls_pos_embed = self.pos_embed[:, : self.n_cls_tokens]
        patch_pos_embed = self.pos_embed[:, self.n_cls_tokens :]

        pt_size = int(patch_pos_embed.shape[1] ** 0.5)

        if pt_size == H == W:
            return self.pos_embed

        patch_pos_embed = patch_pos_embed.reshape(1, pt_size, pt_size, -1).permute(
            0, 3, 1, 2
        )
        patch_pos_embed = torch.nn.functional.interpolate(
            patch_pos_embed, size=(H, W), mode="bicubic", align_corners=False
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, H * W, -1)

        new_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed), dim=1)

        return new_pos_embed

    def forward(self, encoder_input):
        assert isinstance(
            encoder_input.image, torch.Tensor
        ), "Input must be a torch.Tensor"
        assert encoder_input.image.ndim == 4, "Input must be of shape (B, C, H, W)"
        _, channels, height, width = encoder_input.image.shape
        assert channels == 3, "Input must have 3 channels"
        assert (
            height % self.patch_size == 0 and width % self.patch_size == 0
        ), f"Input shape must be divisible by patch size: {self.patch_size}"

        pos_embed = self._interpolate_pos_emb(encoder_input.image)

        x = self.patch_embed(encoder_input.image)

        x = x + pos_embed[:, self.n_cls_tokens :, :]

        cls_token = self.cls_token + pos_embed[:, : self.n_cls_tokens, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        layers = list(range(len(self.blocks)))
        for i, blk in enumerate(self.blocks):
            x = blk(x)

            if i in layers:
                x_norm = self.norm(x)
                features = x_norm[:, self.n_cls_tokens :]

        features = features.permute(0, 2, 1)

        features = features.reshape(
            -1, self.enc_embed_dim, height // self.patch_size, width // self.patch_size
        ).contiguous()

        return ViTEncoderOutput(features=features)
    
    def get_pth_file(self, repo_id: str) -> str:
        files = list_repo_files(repo_id)
        pth_files = [f for f in files if f.endswith(".pth")]
        if not pth_files:
            raise FileNotFoundError(f"No .pth file found in {repo_id}")
        if len(pth_files) > 1:
            raise ValueError(f"Multiple .pth files found: {pth_files}")
        return hf_hub_download(repo_id=repo_id, filename=pth_files[0])


def pixio_vitb16(pretrained=None):
    model = PixioEncoder(
        img_size=256,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        n_cls_tokens=8,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )

    if pretrained:
        state_dict = torch.load(pretrained, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)

    return model


def pixio_vitl16(pretrained=None):
    model = PixioEncoder(
        img_size=256,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        n_cls_tokens=8,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )

    if pretrained:
        state_dict = torch.load(pretrained, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)

    return model


def pixio_vith16(pretrained=None):
    model = PixioEncoder(
        img_size=256,
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        n_cls_tokens=8,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )

    if pretrained:
        state_dict = torch.load(pretrained, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)

    return model


def pixio_vit1b16(pretrained=None):
    model = PixioEncoder(
        img_size=256,
        patch_size=16,
        embed_dim=1536,
        depth=48,
        num_heads=24,
        mlp_ratio=4,
        n_cls_tokens=8,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )

    if pretrained:
        state_dict = torch.load(pretrained, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)

    return model


def pixio_vit5b16(pretrained=None):
    model = PixioEncoder(
        img_size=256,
        patch_size=16,
        embed_dim=3072,
        depth=48,
        num_heads=32,
        mlp_ratio=4,
        n_cls_tokens=8,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )

    if pretrained:
        state_dict = torch.load(pretrained, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)

    return model
