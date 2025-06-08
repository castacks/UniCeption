"""
Encoder Factory for UniCeption
"""

import os

from uniception.models.encoders.base import (
    EncoderGlobalRepInput,
    EncoderInput,
    UniCeptionEncoderBase,
    UniCeptionViTEncoderBase,
    ViTEncoderInput,
    ViTEncoderNonImageInput,
    ViTEncoderOutput,
)

from uniception.models.encoders.cosmos import CosmosEncoder
from uniception.models.encoders.croco import CroCoEncoder, CroCoIntermediateFeatureReturner
from uniception.models.encoders.dinov2 import DINOv2Encoder, DINOv2IntermediateFeatureReturner
from uniception.models.encoders.radio import RADIOEncoder, RADIOIntermediateFeatureReturner
from uniception.models.encoders.dense_rep_encoder import DenseRepresentationEncoder
from uniception.models.encoders.global_rep_encoder import GlobalRepresentationEncoder
from uniception.models.encoders.naradio import NARADIOEncoder
from uniception.models.encoders.patch_embedder import PatchEmbedder


# Define encoder configurations
ENCODER_CONFIGS = {
    "croco": {
        "class": CroCoEncoder,
        "intermediate_feature_returner_class": CroCoIntermediateFeatureReturner,
        "supported_models": ["CroCov2", "DUSt3R", "MASt3R"],
    },
    "dense_rep_encoder": {
        "class": DenseRepresentationEncoder,
        "supported_models": ["Dense-Representation-Encoder"],
    },
    "dinov2": {
        "class": DINOv2Encoder,
        "intermediate_feature_returner_class": DINOv2IntermediateFeatureReturner,
        "supported_models": ["DINOv2", "DINOv2-Registers", "DINOv2-Depth-Anythingv2"],
    },
    "dinov2_intermediate_feature_returner": {
        "class": DINOv2IntermediateFeatureReturner,
        "supported_models": ["DINOv2-Intermediate-Feature-Returner"],
    },
    "global_rep_encoder": {
        "class": GlobalRepresentationEncoder,
        "supported_models": ["Global-Representation-Encoder"],
    },
    "patch_embedder": {
        "class": PatchEmbedder,
        "supported_models": ["Patch-Embedder"],
    },
    "radio": {
        "class": RADIOEncoder,
        "intermediate_feature_returner_class": RADIOIntermediateFeatureReturner,
        "supported_models": ["RADIO", "E-RADIO"],
    },
    "cosmos": {
        "class": CosmosEncoder,
        "supported_models": ["Cosmos-Tokenizer CI8x8", "Cosmos-Tokenizer CI16x16"],
    },
    "naradio": {
        "class": NARADIOEncoder,
        "supported_models": ["RADIO"],
    },
    # Add other encoders here
}


def encoder_factory(encoder_str: str, **kwargs) -> UniCeptionEncoderBase:
    """
    Encoder factory for UniCeption.
    Please use python3 -m uniception.models.encoders.list to see available encoders.

    Args:
        encoder_str (str): Name of the encoder to create.
        **kwargs: Additional keyword arguments to pass to the encoder constructor.

    Returns:
        UniCeptionEncoderBase: An instance of the specified encoder.
    """
    if encoder_str not in ENCODER_CONFIGS:
        raise ValueError(
            f"Unknown encoder: {encoder_str}. For valid encoder_str options, please use python3 -m uniception.models.encoders.list"
        )

    encoder_config = ENCODER_CONFIGS[encoder_str]
    encoder_class = encoder_config["class"]

    return encoder_class(**kwargs)


def feature_returner_encoder_factory(encoder_str: str, **kwargs) -> UniCeptionEncoderBase:
    """
    Encoder factory for UniCeption.
    Please use python3 -m uniception.models.encoders.list to see available encoders.

    Args:
        encoder_str (str): Name of the encoder to create.
        **kwargs: Additional keyword arguments to pass to the encoder constructor.

    Returns:
        UniCeptionEncoderBase: An instance of the specified encoder.
    """
    if encoder_str not in ENCODER_CONFIGS:
        raise ValueError(
            f"Unknown encoder: {encoder_str}. For valid encoder_str options, please use python3 -m uniception.models.encoders.list"
        )

    encoder_config = ENCODER_CONFIGS[encoder_str]
    encoder_class = encoder_config["intermediate_feature_returner_class"]

    return encoder_class(**kwargs)


def get_available_encoders() -> list:
    """
    Get a list of available encoders in UniCeption.

    Returns:
        list: A list of available encoder names.
    """
    return list(ENCODER_CONFIGS.keys())


def print_available_encoder_models():
    """
    Print the currently supported encoders in UniCeption.
    """
    print("Currently Supported Encoders in UniCeption:\nFormat -> encoder_str: supported_models")
    for encoder_name, config in ENCODER_CONFIGS.items():
        print(f"{encoder_name}: {', '.join(config['supported_models'])}")


def _make_encoder_test(encoder_str: str, **kwargs) -> UniCeptionEncoderBase:
    "Function to create encoders for testing purposes."
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
    elif "naradio" in encoder_str:
        return NARADIOEncoder(
            name=encoder_str,
            model_version=encoder_str.replace("na", ""),
        )
    elif "radio" in encoder_str:
        if "e-radio" in encoder_str:
            eradio_input_shape = (224, 224)
        else:
            eradio_input_shape = None
        return RADIOEncoder(
            name=encoder_str,
            model_version=encoder_str,
            eradio_input_shape=eradio_input_shape,
        )
    elif "cosmos" in encoder_str:
        patch_size = int(encoder_str.split("x")[-1])
        return CosmosEncoder(
            name=encoder_str,
            patch_size=patch_size,
            pretrained_checkpoint_path=f"{relative_checkpoint_path}/Cosmos-Tokenizer-CI{patch_size}x{patch_size}/encoder.pth",
        )
    elif "patch_embedder" in encoder_str:
        return PatchEmbedder(
            name=encoder_str,
        )
    else:
        raise ValueError(f"Unknown encoder: {encoder_str}")


__all__ = [
    "encoder_factory",
    "get_available_encoders",
    "print_available_encoder_models",
    "_make_encoder_test",
    "UniCeptionEncoderBase",
    "UniCeptionViTEncoderBase",
    "EncoderInput",
    "ViTEncoderInput",
    "ViTEncoderOutput",
]
