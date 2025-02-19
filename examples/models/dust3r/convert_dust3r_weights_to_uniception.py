"""
This file extracts the cross-attention transformer & prediction head weights from dust3r checkpoints into uniception format.

Special Notice: dust3r have changed their released weights before/after CVPR, and
uniception uses the checkpoint BEFORE CVPR (they perform better). So please make sure you are not converting
the newly downloaded weights. Consult Yuchen and Nikhil on where to find the old weights.
"""

import argparse
import os

import torch
from torch import nn

from uniception.models.info_sharing.cross_attention_transformer import MultiViewCrossAttentionTransformerIFR
from uniception.models.prediction_heads.dpt import DPTFeature, DPTRegressionProcessor
from uniception.models.prediction_heads.linear import LinearFeature


def extract_cross_attention_weights(checkpoint_path, output_folder, output_filename):
    "Extract the UniCeption format cross attention weights from the original CroCoV2/DUSt3R/MASt3R checkpoints."
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Filter the relevant keys for the cross attention model and duplicate if necessary
    filtered_checkpoint = checkpoint["model"]
    filtered_checkpoint = {k: v for k, v in filtered_checkpoint.items() if "dec" in k}
    duplicate_checkpoint = {}
    if not any(k.startswith("dec_blocks2") for k in filtered_checkpoint):
        print("Duplicating dec_blocks to dec_blocks2")
        for key, value in filtered_checkpoint.items():
            if key.startswith("dec_blocks"):
                duplicate_checkpoint[key.replace("dec_blocks", "dec_blocks2")] = value
        filtered_checkpoint = {**filtered_checkpoint, **duplicate_checkpoint}
    new_checkpoint = {}
    for k, v in filtered_checkpoint.items():
        if "decoder_embed" in k:
            new_key = k.replace("decoder_embed", "proj_embed")
            new_checkpoint[new_key] = v
        elif "dec_blocks." in k:
            new_key = k.replace("dec_blocks.", "multi_view_branches.0.")
            new_checkpoint[new_key] = v
        elif "dec_blocks2." in k:
            new_key = k.replace("dec_blocks2.", "multi_view_branches.1.")
            new_checkpoint[new_key] = v
        elif "dec_norm" in k:
            new_key = k.replace("dec_norm", "norm")
            new_checkpoint[new_key] = v

    # Init model
    model = MultiViewCrossAttentionTransformerIFR(
        name="MV-CAT-IFR",
        input_embed_dim=1024,
        num_views=2,
        indices=[5, 8],
        norm_intermediate=False,
    )

    # Load new checkpoint
    print(model.load_state_dict(new_checkpoint))

    # Save the checkpoint
    save_checkpoint = {}
    save_checkpoint["model"] = model.state_dict()
    os.makedirs(os.path.join(output_folder, "cross_attn_transformer"), exist_ok=True)
    save_path = os.path.join(output_folder, "cross_attn_transformer", output_filename)
    torch.save(save_checkpoint, save_path)


def extract_dust3r_dpt_checkpoints(checkpoint_path, output_folder, output_filename):
    "Extract the UniCeption format DPT head weights from the original DUSt3R checkpoint."
    source_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    for head in ["head1", "head2"]:
        # Extract head weights from the checkpoint
        dpt_head_weights = {k: v for k, v in source_ckpt["model"].items() if k.startswith(f"downstream_{head}")}
        dpt_head_weights = {k.replace(f"downstream_{head}.dpt.", ""): v for k, v in dpt_head_weights.items()}
        dpt_feature_weights = {k: v for k, v in dpt_head_weights.items() if not (k.startswith("head"))}

        # Construct the DPTFeature module and load the weights
        dpt = DPTFeature(
            patch_size=16,
            hooks=[0, 1, 2, 3],
            input_feature_dims=[1024, 768, 768, 768],
            layer_dims=[96, 192, 384, 768],
            feature_dim=256,
            use_bn=False,
            output_width_ratio=1,
        )

        dpt.load_state_dict(dpt_feature_weights, strict=True)

        # Construct the dpt processor module and load the weights
        dpt_processor_weights = {k.replace("head.", ""): v for k, v in dpt_head_weights.items() if k.startswith("head")}

        # Replace the keys according to:
        key_replace_dict = {
            "0.weight": "conv1.weight",
            "0.bias": "conv1.bias",
            "2.weight": "conv2.0.weight",
            "2.bias": "conv2.0.bias",
            "4.weight": "conv2.2.weight",
            "4.bias": "conv2.2.bias",
        }

        dpt_processor_weights = {key_replace_dict.get(k, k): v for k, v in dpt_processor_weights.items()}

        dpt_reg_processor = DPTRegressionProcessor(input_feature_dim=256, output_dim=4, hidden_dims=[128, 128])

        dpt_reg_processor.load_state_dict(dpt_processor_weights, strict=True)

        # Save the state_dicts of the DPTFeature and DPTRegressionProcessor
        dpt_feature_path = os.path.join(output_folder, "dpt_feature_head", output_filename + f"_feature_{head}.pth")
        dpt_reg_processor_path = os.path.join(
            output_folder, "dpt_reg_processor", output_filename + f"_reg_processor{head[-1]}.pth"
        )

        os.makedirs(os.path.dirname(dpt_feature_path), exist_ok=True)
        os.makedirs(os.path.dirname(dpt_reg_processor_path), exist_ok=True)

        torch.save({"model": dpt.state_dict()}, dpt_feature_path)
        torch.save({"model": dpt_reg_processor.state_dict()}, dpt_reg_processor_path)


def extract_dust3r_linear_checkpoints(checkpoint_path, output_folder, output_filename):
    "Extract the UniCeption format linear head weights from the original DUSt3R checkpoint."
    test_linear_to_conv()

    source_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    for head in ["head1", "head2"]:
        linear_head_params = {k: v for k, v in source_ckpt["model"].items() if k.startswith(f"downstream_{head}")}
        linear_head_params = {k.replace(f"downstream_{head}.proj.", ""): v for k, v in linear_head_params.items()}

        assert set(linear_head_params.keys()) == {"weight", "bias"}

        input_feature_dim = 768
        output_dim = 4
        patch_size = 16

        linear = nn.Linear(input_feature_dim, output_dim * patch_size * patch_size, bias=True)
        linear.load_state_dict(linear_head_params, strict=True)

        conv_layer = linear_to_conv2d(linear)

        linear_feature = LinearFeature(input_feature_dim, 4, patch_size)
        linear_feature.linear.load_state_dict(conv_layer.state_dict(), strict=True)

        linear_feature_path = os.path.join(
            output_folder, "linear_feature_head", output_filename + f"_feature_{head}.pth"
        )
        os.makedirs(os.path.dirname(linear_feature_path), exist_ok=True)
        torch.save({"model": linear_feature.state_dict()}, linear_feature_path)


def extract_mast3r_dpt_checkpoints(checkpoint_path, output_folder, output_filename):
    "Extract the UniCeption format DPT head weights from the original MASt3R checkpoint."
    source_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    for head in ["head1", "head2"]:
        dpt_head = {k: v for k, v in source_ckpt["model"].items() if k.startswith(f"downstream_{head}")}
        dpt_head = {k.replace(f"downstream_{head}.", ""): v for k, v in dpt_head.items()}
        dpt_head = {k.replace("dpt.", ""): v for k, v in dpt_head.items()}

        dpt_feature_weights = {
            k: v for k, v in dpt_head.items() if not (k.startswith("head") or k.startswith("head_local_features"))
        }

        dpt = DPTFeature(
            patch_size=16,
            hooks=[0, 1, 2, 3],
            input_feature_dims=[1024, 768, 768, 768],
            layer_dims=[96, 192, 384, 768],
            feature_dim=256,
            use_bn=False,
            output_width_ratio=1,
        )

        dpt.load_state_dict(dpt_feature_weights, strict=True)

        dpt_processor_weights = {
            k.replace("head.", ""): v
            for k, v in dpt_head.items()
            if (k.startswith("head") and not k.startswith("head_local_features"))
        }

        # Replace the keys according to:
        key_replace_dict = {
            "0.weight": "conv1.weight",
            "0.bias": "conv1.bias",
            "2.weight": "conv2.0.weight",
            "2.bias": "conv2.0.bias",
            "4.weight": "conv2.2.weight",
            "4.bias": "conv2.2.bias",
        }

        dpt_processor_weights = {key_replace_dict.get(k, k): v for k, v in dpt_processor_weights.items()}

        dpt_reg_processor = DPTRegressionProcessor(input_feature_dim=256, output_dim=4, hidden_dims=[128, 128])

        dpt_reg_processor.load_state_dict(dpt_processor_weights, strict=True)

        # Save the state_dicts of the DPTFeature and DPTRegressionProcessor
        dpt_feature_path = os.path.join(output_folder, "dpt_feature_head", output_filename + f"_feature_{head}.pth")
        dpt_reg_processor_path = os.path.join(
            output_folder, "dpt_reg_processor", output_filename + f"_reg_processor{head[-1]}.pth"
        )

        os.makedirs(os.path.dirname(dpt_feature_path), exist_ok=True)
        os.makedirs(os.path.dirname(dpt_reg_processor_path), exist_ok=True)

        torch.save({"model": dpt.state_dict()}, dpt_feature_path)
        torch.save({"model": dpt_reg_processor.state_dict()}, dpt_reg_processor_path)


def linear_to_conv2d(linear_layer):
    """
    Converts a nn.Linear layer to an equivalent nn.Conv2d layer with a 1x1 kernel.

    Parameters:
    - linear_layer (nn.Linear): The Linear layer to convert.

    Returns:
    - conv_layer (nn.Conv2d): The equivalent Conv2d layer.
    """
    # Extract in_features and out_features from the Linear layer
    in_features = linear_layer.in_features
    out_features = linear_layer.out_features
    bias = linear_layer.bias is not None

    # Create a Conv2d layer with a 1x1 kernel
    conv_layer = nn.Conv2d(
        in_channels=in_features, out_channels=out_features, kernel_size=1, stride=1, padding=0, bias=bias
    )

    # Reshape Linear weights to match Conv2d weights
    conv_weight = linear_layer.weight.data.view(out_features, in_features, 1, 1).clone()
    conv_layer.weight.data = conv_weight

    # Copy bias if it exists
    if bias:
        conv_layer.bias.data = linear_layer.bias.data.clone()

    return conv_layer


def test_linear_to_conv():
    "Test the linear_to_conv2d function."
    batch_size = 4
    height = 16
    width = 24
    in_channels = 3
    out_channels = 5

    # Sample input tensor in BHWC format
    x_linear = torch.randn(batch_size, height, width, in_channels)

    # Define Linear layer
    linear_layer = nn.Linear(in_channels, out_channels)
    output_linear = linear_layer(x_linear)

    # Transpose input tensor to BCHW format for Conv2d
    x_conv = x_linear.permute(0, 3, 1, 2)

    # Define Conv2d layer
    conv_layer = linear_to_conv2d(linear_layer)

    # Get Conv2d output and transpose back to BHWC format
    output_conv = conv_layer(x_conv).permute(0, 2, 3, 1)

    # Verify that outputs are the same
    assert torch.allclose(output_linear, output_conv, atol=1e-6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract dust3r checkpoints to uniception format")

    parser.add_argument(
        "-dcf", "--dust3r_checkpoints_folder", type=str, required=True, help="Path to the dust3r checkpoints folder"
    )
    parser.add_argument("-of", "--output_folder", type=str, required=True, help="Path to the output folder")

    args = parser.parse_args()

    output_folder = args.output_folder
    info_sharing_output_folder = os.path.join(output_folder, "info_sharing")
    pred_head_output_folder = os.path.join(output_folder, "prediction_heads")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(info_sharing_output_folder, exist_ok=True)
    os.makedirs(pred_head_output_folder, exist_ok=True)

    # Extract croco checkpoint
    print("Extracting CroCo checkpoint...")
    croco_ckpt_filepath = os.path.join(args.dust3r_checkpoints_folder, "CroCo_V2_ViTLarge_BaseDecoder.pth")
    extract_cross_attention_weights(
        croco_ckpt_filepath, info_sharing_output_folder, "Two_View_Cross_Attention_Transformer_CroCo.pth"
    )

    # Extract dust3r 224 linear checkpoint
    print("Extracting DUSt3R 224 linear checkpoint...")
    dust3r_ckpt_filepath = os.path.join(args.dust3r_checkpoints_folder, "DUSt3R_ViTLarge_BaseDecoder_224_linear.pth")
    extract_cross_attention_weights(
        dust3r_ckpt_filepath, info_sharing_output_folder, "Two_View_Cross_Attention_Transformer_DUSt3R_224_linear.pth"
    )
    extract_dust3r_linear_checkpoints(dust3r_ckpt_filepath, pred_head_output_folder, "DUSt3R_224_linear")

    # Extract dust3r 512 linear checkpoint
    print("Extracting DUSt3R 512 linear checkpoint...")
    dust3r_ckpt_filepath = os.path.join(args.dust3r_checkpoints_folder, "DUSt3R_ViTLarge_BaseDecoder_512_linear.pth")
    extract_cross_attention_weights(
        dust3r_ckpt_filepath, info_sharing_output_folder, "Two_View_Cross_Attention_Transformer_DUSt3R_512_linear.pth"
    )
    extract_dust3r_linear_checkpoints(dust3r_ckpt_filepath, pred_head_output_folder, "DUSt3R_512_linear")

    # Extract dust3r 512 dpt checkpoint
    print("Extracting DUSt3R 512 dpt checkpoint...")
    dust3r_ckpt_filepath = os.path.join(args.dust3r_checkpoints_folder, "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
    extract_cross_attention_weights(
        dust3r_ckpt_filepath, info_sharing_output_folder, "Two_View_Cross_Attention_Transformer_DUSt3R_512_dpt.pth"
    )
    extract_dust3r_dpt_checkpoints(dust3r_ckpt_filepath, pred_head_output_folder, "DUSt3R_512_dpt")

    # Extract mast3r 512 dpt checkpoint
    print("Extracting MASt3R 512 dpt checkpoint...")
    mast3r_ckpt_path = os.path.join(
        args.dust3r_checkpoints_folder, "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    )
    extract_cross_attention_weights(
        mast3r_ckpt_path, info_sharing_output_folder, "Two_View_Cross_Attention_Transformer_MASt3R_512_dpt.pth"
    )
    extract_mast3r_dpt_checkpoints(mast3r_ckpt_path, pred_head_output_folder, "MASt3R_512_dpt")
