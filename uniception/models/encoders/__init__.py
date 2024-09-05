from uniception.models.encoders.base import (
    UniCeptionEncoderBase,
    UniCeptionViTEncoderBase,
    IntermediateFeatureReturner,
    EncoderInput,
    ViTEncoderInput,
    ViTEncoderOutput,
)

from uniception.models.encoders.croco import CroCoEncoder


def _make_encoder(encoder_str: str, **kwargs) -> UniCeptionEncoderBase:

    if encoder_str == "dummy":
        return UniCeptionEncoderBase(name="dummy", data_norm_type="dummy")
    elif encoder_str == "croco":
        return CroCoEncoder(
            name="croco",
            data_norm_type="croco",
            pretrained_checkpoint_path="/home/inf/shared_workspace/UniCeption/checkpoints/CroCo_Encoder_224.pth",
            patch_embed_cls="PatchEmbedCroCo",
        )
    elif encoder_str == "dust3r_224":
        return CroCoEncoder(
            name="dust3r_224",
            data_norm_type="dust3r",
            pretrained_checkpoint_path="/home/inf/shared_workspace/UniCeption/checkpoints/CroCo_Encoder_224_DUSt3R_linear.pth",
            patch_embed_cls="PatchEmbedDust3R",
        )
    elif encoder_str == "dust3r_512":
        return CroCoEncoder(
            name="dust3r_512",
            data_norm_type="dust3r",
            pretrained_checkpoint_path="/home/inf/shared_workspace/UniCeption/checkpoints/CroCo_Encoder_512_DUSt3R_linear.pth",
            patch_embed_cls="ManyAR_PatchEmbed",
            img_size=(512, 512),
        )
    elif encoder_str == "dust3r_512_dpt":
        return CroCoEncoder(
            name="dust3r_512_dpt",
            data_norm_type="dust3r",
            pretrained_checkpoint_path="/home/inf/shared_workspace/UniCeption/checkpoints/CroCo_Encoder_512_DUSt3R_dpt.pth",
            patch_embed_cls="ManyAR_PatchEmbed",
            img_size=(512, 512),
        )
    elif encoder_str == "mast3r_512":
        return CroCoEncoder(
            name="mast3r_512",
            data_norm_type="dust3r",
            pretrained_checkpoint_path="/home/inf/shared_workspace/UniCeption/checkpoints/CroCo_Encoder_512_MASt3R.pth",
            patch_embed_cls="ManyAR_PatchEmbed",
            img_size=(512, 512),
        )
    else:
        raise ValueError(f"Unknown encoder: {encoder_str}")


__all__ = [
    "_make_encoder",
    "UniCeptionEncoderBase",
    "UniCeptionViTEncoderBase",
    "IntermediateFeatureReturner",
    "EncoderInput",
    "ViTEncoderInput",
    "ViTEncoderOutput",
]
