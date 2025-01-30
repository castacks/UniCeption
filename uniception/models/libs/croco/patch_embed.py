# --------------------------------------------------------
# Patch Embedding for CroCo and DUSt3R
# Adopted from DUSt3R (Naver Corporation, CC BY-NC-SA 4.0 (non-commercial use only))
# --------------------------------------------------------
import torch
import torch.nn as nn

from uniception.models.libs.croco.blocks import to_2tuple

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12


def get_patch_embed(patch_embed_cls, img_size, patch_size, enc_embed_dim):
    assert patch_embed_cls in ["PatchEmbedCroCo", "PatchEmbedDust3R", "ManyAR_PatchEmbed"]
    patch_embed = eval(patch_embed_cls)(img_size, patch_size, 3, enc_embed_dim)
    return patch_embed


class PositionGetter(object):
    """Return positions of patches"""

    def __init__(self):
        self.cache_positions = {}

    def __call__(self, b, h, w, device):
        if not (h, w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            self.cache_positions[h, w] = torch.cartesian_prod(y, x)  # (h, w, 2)
        pos = self.cache_positions[h, w].view(1, h * w, 2).expand(b, -1, 2).clone()
        return pos


class PatchEmbedCroCo(nn.Module):
    """Just adding _init_weights + position getter compared to timm.models.layers.patch_embed.PatchEmbed"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.position_getter = PositionGetter()

    def forward(self, x, **kw):
        B, C, H, W = x.shape
        torch._assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        torch._assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        pos = self.position_getter(B, x.size(2), x.size(3), x.device)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, pos

    def _init_weights(self):
        w = self.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))


class PatchEmbedDust3R(PatchEmbedCroCo):
    def forward(self, x, **kw):
        B, C, H, W = x.shape
        assert (
            H % self.patch_size[0] == 0
        ), f"Input image height ({H}) is not a multiple of patch size ({self.patch_size[0]})."
        assert (
            W % self.patch_size[1] == 0
        ), f"Input image width ({W}) is not a multiple of patch size ({self.patch_size[1]})."
        x = self.proj(x)
        pos = self.position_getter(B, x.size(2), x.size(3), x.device)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, pos


class ManyAR_PatchEmbed(PatchEmbedCroCo):
    """Handle images with non-square aspect ratio.
    All images in the same batch have the same aspect ratio.
    true_shape = [(height, width) ...] indicates the actual shape of each image.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        self.embed_dim = embed_dim
        super().__init__(img_size, patch_size, in_chans, embed_dim, norm_layer, flatten)

    def forward(self, img, true_shape):
        B, C, H, W = img.shape
        assert W >= H, f"img should be in landscape mode, but got {W=} {H=}"
        assert (
            H % self.patch_size[0] == 0
        ), f"Input image height ({H}) is not a multiple of patch size ({self.patch_size[0]})."
        assert (
            W % self.patch_size[1] == 0
        ), f"Input image width ({W}) is not a multiple of patch size ({self.patch_size[1]})."
        assert true_shape.shape == (B, 2), f"true_shape has the wrong shape={true_shape.shape}"

        # size expressed in tokens
        W //= self.patch_size[0]
        H //= self.patch_size[1]
        n_tokens = H * W

        height, width = true_shape.T
        is_landscape = width >= height
        is_portrait = ~is_landscape

        # allocate result
        x = img.new_zeros((B, n_tokens, self.embed_dim))
        pos = img.new_zeros((B, n_tokens, 2), dtype=torch.int64)

        # linear projection, transposed if necessary
        x[is_landscape] = self.proj(img[is_landscape]).permute(0, 2, 3, 1).flatten(1, 2).float()
        x[is_portrait] = self.proj(img[is_portrait].swapaxes(-1, -2)).permute(0, 2, 3, 1).flatten(1, 2).float()

        pos[is_landscape] = self.position_getter(1, H, W, pos.device)
        pos[is_portrait] = self.position_getter(1, W, H, pos.device)

        x = self.norm(x)
        return x, pos
