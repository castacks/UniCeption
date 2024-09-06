# --------------------------------------------------------
# Encoder Class for CroCo & DUSt3R
# --------------------------------------------------------
import torch
import torch.nn as nn
from functools import partial
from typing import Callable, Union, Tuple

from uniception.models.encoders.base import UniCeptionViTEncoderBase, ViTEncoderInput, ViTEncoderOutput
from uniception.models.libs.croco.blocks import Block
from uniception.models.libs.croco.patch_embed import get_patch_embed
from uniception.models.libs.croco.pos_embed import get_2d_sincos_pos_embed, RoPE2D


class CroCoEncoder(UniCeptionViTEncoderBase):
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
        *args,
        **kwargs,
    ):
        """
        CroCoV2 Encoder
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
            pos_embed (str, optional): Positional Embedding. Defaults to 'RoPE100'. Options: ['cosine', 'RoPE100'].
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

        # Init the positional embedding
        self.pos_embed = pos_embed
        if pos_embed == "cosine":
            enc_pos_embed = get_2d_sincos_pos_embed(
                enc_embed_dim, int(self.patch_embed.num_patches**0.5), n_cls_token=0
            )
            self.register_buffer("enc_pos_embed", torch.from_numpy(enc_pos_embed).float())
            self.rope = None  # nothing for cosine
        elif pos_embed.startswith("RoPE"):  # eg RoPE100
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
            ckpt_data_norm_type = ckpt["data_norm_type"]
            ckpt_patch_embed_cls_opts = ckpt["patch_embed_cls"]
            print(self.load_state_dict(ckpt["model"]))
            assert (
                data_norm_type == ckpt_data_norm_type
            ), f"Data normalization type {data_norm_type} does not match the checkpoint {ckpt_data_norm_type}."
            assert (
                patch_embed_cls in ckpt_patch_embed_cls_opts
            ), f"Patch embedding class {patch_embed_cls} does not match the checkpoint options {ckpt_patch_embed_cls_opts}."
        else:
            print("No pretrained checkpoint provided. Randomly initializing the CroCo encoder.")

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def _set_encoder(self, enc_depth, enc_embed_dim, enc_num_heads, mlp_ratio, norm_layer, rope):
        self.enc_blocks = nn.ModuleList(
            [
                Block(enc_embed_dim, enc_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, rope=rope)
                for _ in range(enc_depth)
            ]
        )
        self.enc_norm = norm_layer(enc_embed_dim)

    def initialize_weights(self):
        # Patch embedding
        self.patch_embed._init_weights()
        # Linears and layer norms
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # We use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, encoder_input: ViTEncoderInput) -> ViTEncoderOutput:
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
        features = features.reshape(-1, self.enc_embed_dim, height // self.patch_size, width // self.patch_size)

        return ViTEncoderOutput(features=features)


if __name__ == "__main__":
    # Init a dummy input dataclass for 224 resolution
    encoder_input = ViTEncoderInput(image=torch.randn(1, 3, 224, 224), data_norm_type="croco")

    # Init the pre-trained CroCo Encoder
    pretrained_checkpoint_path = "/ocean/projects/cis220039p/nkeetha/code/UniCeption/checkpoints/CroCo_Encoder_224.pth"
    croco_encoder = CroCoEncoder(
        name="croco",
        data_norm_type="croco",
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        patch_embed_cls="PatchEmbedCroCo",
    )

    # Forward pass the dummy input through the CroCo Encoder
    encoder_output = croco_encoder(encoder_input)

    # Init a dummy input dataclass for 224 resolution
    encoder_input = ViTEncoderInput(image=torch.randn(1, 3, 224, 224), data_norm_type="dust3r")

    # Init the pre-trained DUSt3R CroCo Encoder
    pretrained_checkpoint_path = (
        "/ocean/projects/cis220039p/nkeetha/code/UniCeption/checkpoints/CroCo_Encoder_224_DUSt3R_linear.pth"
    )
    dust3r_encoder = CroCoEncoder(
        name="dust3r_224",
        data_norm_type="dust3r",
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        patch_embed_cls="PatchEmbedDust3R",
    )

    # Forward pass the dummy input through the DUSt3R CroCo Encoder
    encoder_output = dust3r_encoder(encoder_input)

    # Init a dummy input dataclass for 512 resolution
    encoder_input = ViTEncoderInput(image=torch.randn(1, 3, 384, 512), data_norm_type="dust3r")

    # Init the pre-trained DUSt3R 512 linear CroCo Encoder
    pretrained_checkpoint_path = (
        "/ocean/projects/cis220039p/nkeetha/code/UniCeption/checkpoints/CroCo_Encoder_512_DUSt3R_linear.pth"
    )
    dust3r_encoder_512 = CroCoEncoder(
        name="dust3r_512",
        data_norm_type="dust3r",
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        patch_embed_cls="ManyAR_PatchEmbed",
        img_size=(512, 512),
    )

    # Forward pass the dummy input through the DUSt3R 512 linear CroCo Encoder
    encoder_output = dust3r_encoder_512(encoder_input)

    # Init the pre-trained DUSt3R 512 DPT CroCo Encoder
    pretrained_checkpoint_path = (
        "/ocean/projects/cis220039p/nkeetha/code/UniCeption/checkpoints/CroCo_Encoder_512_DUSt3R_dpt.pth"
    )
    dust3r_encoder_512_dpt = CroCoEncoder(
        name="dust3r_512_dpt",
        data_norm_type="dust3r",
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        patch_embed_cls="ManyAR_PatchEmbed",
        img_size=(512, 512),
    )

    # Forward pass the dummy input through the DUSt3R 512 DPT CroCo Encoder
    encoder_output = dust3r_encoder_512_dpt(encoder_input)

    # Init the MASt3R 512 CroCo Encoder
    pretrained_checkpoint_path = (
        "/ocean/projects/cis220039p/nkeetha/code/UniCeption/checkpoints/CroCo_Encoder_512_MASt3R.pth"
    )
    mast3r_encoder_512 = CroCoEncoder(
        name="mast3r_512",
        data_norm_type="dust3r",
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        patch_embed_cls="ManyAR_PatchEmbed",
        img_size=(512, 512),
    )

    # Forward pass the dummy input through the MASt3R 512 CroCo Encoder
    encoder_output = mast3r_encoder_512(encoder_input)

    print("All CroCo & DUSt3R Encoders have been initialized successfully!")
