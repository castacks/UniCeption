"""
Script to convert DINOv2 weights to UniCeption info sharing classes.
"""

import argparse
import os

import torch

from uniception.models.encoders import _make_encoder_test
from uniception.models.encoders.base import ViTEncoderInput
from uniception.models.info_sharing.alternating_attention_transformer import MultiViewAlternatingAttentionTransformer
from uniception.models.info_sharing.base import MultiViewTransformerInput
from uniception.models.info_sharing.global_attention_transformer import MultiViewGlobalAttentionTransformer
from uniception.models.utils.transformer_blocks import Mlp, SwiGLUFFNFused


def get_parser():
    parser = argparse.ArgumentParser(description="Convert DINOv2 weights for info sharing classes.")
    parser.add_argument(
        "--encoder_str", type=str, default="dinov2_large", help="DINOv2 encoder string (e.g., dinov2_large)."
    )
    parser.add_argument("--start", type=int, default=12, help="Starting block index (including)")
    parser.add_argument(
        "--end", type=int, default=-1, help="Ending block index (excluding), use -1 to indicate the last block"
    )
    parser.add_argument(
        "--info_sharing_layers",
        type=int,
        default=-1,
        help="Depth of info sharing transformer. (-1) to determine from start and end. \
        If different then (to - from), will left empty layers or not copy all layers from the encoder.",
    )
    parser.add_argument("--info_sharing_class", type=str, default="global", help="'global' or 'alternating'")
    parser.add_argument("--output_path", type=str, help="Path to save the converted info sharing model.")
    parser.add_argument(
        "--skip_verify", action="store_true", help="Whether to skip verifying the converted model by loading it back."
    )
    return parser


def main(args):
    # Construct UniCeption encoder
    print(f"Loading encoder: {args.encoder_str}")
    encoder = _make_encoder_test(args.encoder_str)

    # Decide the length of layers to extract
    encoder_length = len(encoder.model.blocks)
    if args.end == -1:
        args.end = encoder_length

    layers_to_extract = list(range(args.start, args.end))
    layers_to_extract = layers_to_extract[: min(len(layers_to_extract), encoder_length)]

    if args.info_sharing_layers == -1:
        args.info_sharing_layers = len(layers_to_extract)

    layers_to_extract = layers_to_extract[: min(len(layers_to_extract), args.info_sharing_layers)]

    print("-" * 40)
    print(
        f"Extracting blocks from {layers_to_extract[0]} to {layers_to_extract[-1] + 1} (encoder length: {encoder_length})"
    )
    print(f"Filling into {len(layers_to_extract)}/{args.info_sharing_layers} info sharing layers")

    # Determine the MLP class based on the encoder
    if "giant" in args.encoder_str:
        mlp_layer = SwiGLUFFNFused
    else:
        mlp_layer = Mlp

    # Construct info sharing model
    target_blocks: torch.nn.ModuleList

    if args.info_sharing_class == "global":
        info_sharing_model = MultiViewGlobalAttentionTransformer(
            name="info_sharing",
            input_embed_dim=encoder.model.embed_dim,
            distinguish_ref_and_non_ref_views=False,  # for ease of validation
            use_pe_for_non_reference_views=False,
            max_num_views_for_pe=1000,  # dummy value
            use_rand_idx_pe_for_non_reference_views=False,  # dummy value
            depth=args.info_sharing_layers,
            dim=encoder.model.embed_dim,
            num_heads=encoder.model.num_heads,
            mlp_layer=mlp_layer,
            mlp_ratio=4,  # the same for all DINOv2
            qkv_bias=True,
            qk_norm=False,
            init_values=1e-5,  # for enabling LayerScale
        )

        target_blocks = info_sharing_model.self_attention_blocks
    elif args.info_sharing_class == "alternating":
        info_sharing_model = MultiViewAlternatingAttentionTransformer(
            name="info_sharing",
            input_embed_dim=encoder.model.embed_dim,
            distinguish_ref_and_non_ref_views=False,  # for ease of validation
            use_pe_for_non_reference_views=False,
            max_num_views_for_pe=1000,  # dummy value
            use_rand_idx_pe_for_non_reference_views=False,  # dummy value
            depth=args.info_sharing_layers,
            dim=encoder.model.embed_dim,
            num_heads=encoder.model.num_heads,
            mlp_layer=mlp_layer,
            mlp_ratio=4,  # the same for all DINOv2
            qkv_bias=True,
            qk_norm=False,
            init_values=1e-5,  # for enabling LayerScale
        )

        target_blocks = info_sharing_model.self_attention_blocks
    else:
        raise ValueError(f"Unknown info sharing class: {args.info_sharing_class}")

    # Fill in the weights
    for i, layer_idx in enumerate(layers_to_extract):
        print(f"Copying encoder block {layer_idx} to info sharing block {i}")
        target_blocks[i].load_state_dict(encoder.model.blocks[layer_idx].state_dict())

    # Also copy weight for the final layernorm
    print("Copying encoder final norm to info sharing norm")
    info_sharing_model.norm.load_state_dict(encoder.model.norm.state_dict())

    # Save the converted model
    print(f"Saving to {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save({"model": info_sharing_model.state_dict()}, args.output_path)


def verify(args):
    print("Verifying the converted model...")

    # Construct original encoder
    encoder_reference = _make_encoder_test(
        args.encoder_str,
        norm_returned_features=True,  # keep final norm for reference encoder
    )
    encoder_test = _make_encoder_test(
        args.encoder_str,
        norm_returned_features=False,  # drop final norm for truncated encoder
        keep_first_n_layers=args.start,  # keep only the first 'start' layers
    )

    encoder_length = len(encoder_reference.model.blocks)
    if args.end == -1:
        args.end = encoder_length

    layers_to_extract = list(range(args.start, args.end))
    layers_to_extract = layers_to_extract[: min(len(layers_to_extract), encoder_length)]

    if args.info_sharing_layers == -1:
        args.info_sharing_layers = len(layers_to_extract)

    assert args.start > 0, "Verification only works for start > 0"
    assert args.info_sharing_layers == (
        args.end - args.start
    ), "Verification only works for info_sharing_layers == (end - start)"
    assert args.end <= len(encoder_reference.model.blocks), "Verification only works for end <= encoder length"

    encoder_reference.model.blocks = encoder_reference.model.blocks[: args.end]  # keep the full encoder up to "end"

    # Determine the MLP class based on the encoder
    if "giant" in args.encoder_str:
        mlp_layer = SwiGLUFFNFused
    else:
        mlp_layer = Mlp

    # Construct info sharing model
    info_sharing_model: torch.nn.Module
    if args.info_sharing_class == "global":
        info_sharing_model = MultiViewGlobalAttentionTransformer(
            name="info_sharing",
            input_embed_dim=encoder_reference.model.embed_dim,
            distinguish_ref_and_non_ref_views=False,  # for ease of validation
            use_pe_for_non_reference_views=False,  # necessary for preserving DINO features
            max_num_views_for_pe=1000,  # dummy value
            use_rand_idx_pe_for_non_reference_views=False,  # dummy value
            depth=args.info_sharing_layers,
            dim=encoder_reference.model.embed_dim,
            num_heads=encoder_reference.model.num_heads,
            mlp_layer=mlp_layer,
            mlp_ratio=4,  # the same for all DINOv2
            qkv_bias=True,
            qk_norm=False,
            init_values=1e-5,  # for enabling LayerScale
            pretrained_checkpoint_path=args.output_path,
        )
    elif args.info_sharing_class == "alternating":
        info_sharing_model = MultiViewAlternatingAttentionTransformer(
            name="info_sharing",
            input_embed_dim=encoder_reference.model.embed_dim,
            distinguish_ref_and_non_ref_views=False,  # for ease of validation
            use_pe_for_non_reference_views=False,  # necessary for preserving DINO features
            max_num_views_for_pe=1000,  # dummy value
            use_rand_idx_pe_for_non_reference_views=False,  # dummy value
            depth=args.info_sharing_layers,
            dim=encoder_reference.model.embed_dim,
            num_heads=encoder_reference.model.num_heads,
            mlp_layer=mlp_layer,
            mlp_ratio=4,  # the same for all DINOv2
            qkv_bias=True,
            qk_norm=False,
            init_values=1e-5,  # for enabling LayerScale
            pretrained_checkpoint_path=args.output_path,
        )
    else:
        raise ValueError(f"Unknown info sharing class: {args.info_sharing_class}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move models to device
    encoder_reference = encoder_reference.to(device)
    encoder_test = encoder_test.to(device)
    info_sharing_model = info_sharing_model.to(device)

    torch.manual_seed(42)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    with torch.no_grad():
        # Forward pass through the reference encoder
        reference_feature = encoder_reference(ViTEncoderInput(image=dummy_input, data_norm_type="dinov2")).features

        # Forward pass through the test encoder + info sharing(without any PE in info_sharing)
        # Pass through the truncated encoder
        encoder_result = encoder_test(ViTEncoderInput(image=dummy_input, data_norm_type="dinov2"))

        features_test_encoder = encoder_result.features
        registers = encoder_result.registers

        # Passing in the registers is ABSOLUTELY ESSENTIAL for correct dino feature to appear in the info-sharing module.
        # if not passed, would result in 1.15 relative error! which totally changes the feature distribution.
        info_sharing_output = info_sharing_model(
            MultiViewTransformerInput(
                features=[features_test_encoder], additional_input_tokens_per_view=[registers]  # single view
            )
        ).features[0]

        relative_error = torch.linalg.norm(info_sharing_output - reference_feature) / torch.linalg.norm(
            reference_feature
        )
        assert relative_error < 1e-3, f"Verification failed: Relative error {relative_error} is too high!"
        print(
            f"Relative error between {len(encoder_reference.model.blocks)} layer encoder to {len(encoder_test.model.blocks)} layer encoder + {len(info_sharing_model.self_attention_blocks)} layer info-sharing: {relative_error:.3e}"
        )

    print("Verification successful: The features from the reference encoder and the encoder + info sharing match!")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)

    if not args.skip_verify:
        verify(args)
