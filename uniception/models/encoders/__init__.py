# --------------------------------------------------------
# Encoder Factory for UniCeption
# --------------------------------------------------------
import os
from uniception.models.encoders.base import (
    UniCeptionEncoderBase,
    UniCeptionViTEncoderBase,
    IntermediateFeatureReturner,
    EncoderInput,
    ViTEncoderInput,
    ViTEncoderOutput,
)

from uniception.models.encoders.croco import CroCoEncoder
from uniception.models.encoders.dinov2 import DINOv2Encoder


def _make_encoder_test(encoder_str: str, **kwargs) -> UniCeptionEncoderBase:
    current_file_path = os.path.abspath(__file__)
    relative_checkpoint_path = os.path.join(os.path.dirname(current_file_path), "../../../checkpoints/encoders")
    if encoder_str == "dummy":
        return UniCeptionEncoderBase(name="dummy", data_norm_type="dummy")
    elif encoder_str == "croco":
        return CroCoEncoder(
            name="croco",
            data_norm_type="croco",
            pretrained_checkpoint_path=f"{relative_checkpoint_path}/CroCo_Encoder_224.pth",
            patch_embed_cls="PatchEmbedCroCo",
        )
    elif encoder_str == "dust3r_224":
        return CroCoEncoder(
            name="dust3r_224",
            data_norm_type="dust3r",
            pretrained_checkpoint_path=f"{relative_checkpoint_path}/CroCo_Encoder_224_DUSt3R_linear.pth",
            patch_embed_cls="PatchEmbedDust3R",
        )
    elif encoder_str == "dust3r_512":
        return CroCoEncoder(
            name="dust3r_512",
            data_norm_type="dust3r",
            pretrained_checkpoint_path=f"{relative_checkpoint_path}/CroCo_Encoder_512_DUSt3R_linear.pth",
            patch_embed_cls="ManyAR_PatchEmbed",
            img_size=(512, 512),
        )
    elif encoder_str == "dust3r_512_dpt":
        return CroCoEncoder(
            name="dust3r_512_dpt",
            data_norm_type="dust3r",
            pretrained_checkpoint_path=f"{relative_checkpoint_path}/CroCo_Encoder_512_DUSt3R_dpt.pth",
            patch_embed_cls="ManyAR_PatchEmbed",
            img_size=(512, 512),
        )
    elif encoder_str == "mast3r_512":
        return CroCoEncoder(
            name="mast3r_512",
            data_norm_type="dust3r",
            pretrained_checkpoint_path=f"{relative_checkpoint_path}/CroCo_Encoder_512_MASt3R.pth",
            patch_embed_cls="ManyAR_PatchEmbed",
            img_size=(512, 512),
        )
    elif "dinov2" in encoder_str:
        size = encoder_str.split("_")[1]
        size_single_cap_letter = size[0].upper()
        if "reg" in encoder_str:
            with_registers = True
            pretrained_checkpoint_path = None
        elif "dav2" in encoder_str:
            with_registers = False
            pretrained_checkpoint_path = (
                f"{relative_checkpoint_path}/DINOv2_ViT{size_single_cap_letter}_DepthAnythingV2.pth"
            )
        else:
            with_registers = False
            pretrained_checkpoint_path = None
        return DINOv2Encoder(
            name=encoder_str.replace("_reg", ""),
            size=size,
            with_registers=with_registers,
            pretrained_checkpoint_path=pretrained_checkpoint_path,
        )
    else:
        raise ValueError(f"Unknown encoder: {encoder_str}")


__all__ = [
    "_make_encoder_test",
    "UniCeptionEncoderBase",
    "UniCeptionViTEncoderBase",
    "IntermediateFeatureReturner",
    "EncoderInput",
    "ViTEncoderInput",
    "ViTEncoderOutput",
]
