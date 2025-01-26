"""
Initalizing Pre-trained DUSt3R using UniCeption
"""

import argparse
import os
from io import BytesIO

import numpy as np
import requests
import rerun as rr
import torch
from PIL import Image
from uniception.models.factory import DUSt3R
from uniception.utils.viz import script_add_rerun_args


def get_model_configurations_and_checkpoints():
    """
    Get different DUSt3R model configurations and paths to refactored checkpoints.

    Returns:
        Tuple[List[str], dict]: A tuple containing the model configurations and paths to refactored checkpoints.
    """
    # Initialize model configurations
    model_configurations = ["dust3r_224_linear", "dust3r_512_linear", "dust3r_512_dpt", "dust3r_512_dpt_mast3r"]

    # Get paths to pretrained checkpoints
    current_file_path = os.path.abspath(__file__)
    relative_checkpoint_path = os.path.join(os.path.dirname(current_file_path), "../../../checkpoints")

    # Initialize model configurations
    model_to_checkpoint_path = {
        "dust3r_512_dpt": {
            "encoder": f"{relative_checkpoint_path}/encoders/CroCo_Encoder_512_DUSt3R_dpt.pth",
            "info_sharing": f"{relative_checkpoint_path}/info_sharing/cross_attn_transformer/Two_View_Cross_Attention_Transformer_DUSt3R_512_dpt.pth",
            "feature_head": [
                f"{relative_checkpoint_path}/prediction_heads/dpt_feature_head/DUSt3R_512_dpt_feature_head1.pth",
                f"{relative_checkpoint_path}/prediction_heads/dpt_feature_head/DUSt3R_512_dpt_feature_head2.pth",
            ],
            "regressor": [
                f"{relative_checkpoint_path}/prediction_heads/dpt_reg_processor/DUSt3R_512_dpt_reg_processor1.pth",
                f"{relative_checkpoint_path}/prediction_heads/dpt_reg_processor/DUSt3R_512_dpt_reg_processor2.pth",
            ],
            "ckpt_path": f"{relative_checkpoint_path}/examples/original_dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        },
        "dust3r_512_dpt_mast3r": {
            "encoder": f"{relative_checkpoint_path}/encoders/CroCo_Encoder_512_MASt3R.pth",
            "info_sharing": f"{relative_checkpoint_path}/info_sharing/cross_attn_transformer/Two_View_Cross_Attention_Transformer_MASt3R_512_dpt.pth",
            "feature_head": [
                f"{relative_checkpoint_path}/prediction_heads/dpt_feature_head/MASt3R_512_dpt_feature_head1.pth",
                f"{relative_checkpoint_path}/prediction_heads/dpt_feature_head/MASt3R_512_dpt_feature_head2.pth",
            ],
            "regressor": [
                f"{relative_checkpoint_path}/prediction_heads/dpt_reg_processor/MASt3R_512_dpt_reg_processor1.pth",
                f"{relative_checkpoint_path}/prediction_heads/dpt_reg_processor/MASt3R_512_dpt_reg_processor2.pth",
            ],
            "ckpt_path": f"{relative_checkpoint_path}/examples/original_dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt_mast3r.pth",
        },
        "dust3r_512_linear": {
            "encoder": f"{relative_checkpoint_path}/encoders/CroCo_Encoder_512_DUSt3R_linear.pth",
            "info_sharing": f"{relative_checkpoint_path}/info_sharing/cross_attn_transformer/Two_View_Cross_Attention_Transformer_DUSt3R_512_linear.pth",
            "feature_head": [
                f"{relative_checkpoint_path}/prediction_heads/linear_feature_head/DUSt3R_512_linear_feature_head1.pth",
                f"{relative_checkpoint_path}/prediction_heads/linear_feature_head/DUSt3R_512_linear_feature_head2.pth",
            ],
            "regressor": None,
            "ckpt_path": f"{relative_checkpoint_path}/examples/original_dust3r/DUSt3R_ViTLarge_BaseDecoder_512_linear.pth",
        },
        "dust3r_224_linear": {
            "encoder": f"{relative_checkpoint_path}/encoders/CroCo_Encoder_224_DUSt3R_linear.pth",
            "info_sharing": f"{relative_checkpoint_path}/info_sharing/cross_attn_transformer/Two_View_Cross_Attention_Transformer_DUSt3R_224_linear.pth",
            "feature_head": [
                f"{relative_checkpoint_path}/prediction_heads/linear_feature_head/DUSt3R_224_linear_feature_head1.pth",
                f"{relative_checkpoint_path}/prediction_heads/linear_feature_head/DUSt3R_224_linear_feature_head2.pth",
            ],
            "regressor": None,
            "ckpt_path": f"{relative_checkpoint_path}/examples/original_dust3r/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth",
        },
    }
    return model_configurations, model_to_checkpoint_path


def get_parser():
    "Argument parser for the script."
    parser = argparse.ArgumentParser()
    parser.add_argument("--viz", action="store_true")

    return parser


if __name__ == "__main__":
    # Parse arguments
    parser = get_parser()
    script_add_rerun_args(parser)  # Options: --addr
    args = parser.parse_args()

    # Set up Rerun for visualization
    if args.viz:
        rr.script_setup(args, f"UniCeption_DUSt3R_Inference")
        rr.set_time_seconds("stable_time", 0)

    # the reference data are collected under this setting.
    # may use (False, "high") to test the relative error at TF32 precision
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    # Get paths to pretrained checkpoints
    current_file_path = os.path.abspath(__file__)
    relative_checkpoint_path = os.path.join(os.path.dirname(current_file_path), "../../../checkpoints")
    model_configurations, model_to_checkpoint_path = get_model_configurations_and_checkpoints()

    MODEL_TO_VERIFICATION_PATH = {
        "dust3r_512_dpt": {
            "head_output": os.path.join(
                os.path.dirname(current_file_path),
                "../../../reference_data/dust3r_pre_cvpr",
                "DUSt3R_512_dpt",
                "03_head_output.npz",
            )
        },
        "dust3r_512_dpt_mast3r": {
            "head_output": os.path.join(
                os.path.dirname(current_file_path),
                "../../../reference_data/dust3r_pre_cvpr",
                "MASt3R_512_dpt",
                "03_head_output.npz",
            )
        },
        "dust3r_512_linear": {
            "head_output": os.path.join(
                os.path.dirname(current_file_path),
                "../../../reference_data/dust3r_pre_cvpr",
                "DUSt3R_512_linear",
                "03_head_output.npz",
            )
        },
        "dust3r_224_linear": {
            "head_output": os.path.join(
                os.path.dirname(current_file_path),
                "../../../reference_data/dust3r_pre_cvpr",
                "DUSt3R_224_linear",
                "03_head_output.npz",
            )
        },
    }

    # Test different DUSt3R models using UniCeption modules
    for model_name in model_configurations:
        dust3r_model = DUSt3R(
            name=model_name,
            img_size=(512, 512) if "512" in model_name else (224, 224),
            patch_embed_cls="PatchEmbedDust3R",
            pred_head_type="linear" if "linear" in model_name else "dpt",
            pretrained_checkpoint_path=model_to_checkpoint_path[model_name]["ckpt_path"],
            # pretrained_encoder_checkpoint_path=model_to_checkpoint_path[model_name]["encoder"],
            # pretrained_info_sharing_checkpoint_path=model_to_checkpoint_path[model_name]["info_sharing"],
            # pretrained_pred_head_checkpoint_paths=model_to_checkpoint_path[model_name]["feature_head"],
            # pretrained_pred_head_regressor_checkpoint_paths=model_to_checkpoint_path[model_name]["regressor"],
            # override_encoder_checkpoint_attributes=True,
        )
        print("DUSt3R model initialized successfully!")

        # Initalize device
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        dust3r_model.to(device)

        # Initalize two example images
        img0_url = (
            "https://raw.githubusercontent.com/naver/croco/d3d0ab2858d44bcad54e5bfc24f565983fbe18d9/assets/Chateau1.png"
        )
        img1_url = (
            "https://raw.githubusercontent.com/naver/croco/d3d0ab2858d44bcad54e5bfc24f565983fbe18d9/assets/Chateau2.png"
        )
        response = requests.get(img0_url)
        img0 = Image.open(BytesIO(response.content))
        response = requests.get(img1_url)
        img1 = Image.open(BytesIO(response.content))
        img0_tensor = torch.from_numpy(np.array(img0))[..., :3].permute(2, 0, 1).unsqueeze(0).float() / 255
        img1_tensor = torch.from_numpy(np.array(img1))[..., :3].permute(2, 0, 1).unsqueeze(0).float() / 255

        # Normalize images according to DUSt3R's normalization
        img0_tensor = (img0_tensor - 0.5) / 0.5
        img1_tensor = (img1_tensor - 0.5) / 0.5
        img_tensor = torch.cat((img0_tensor, img1_tensor), dim=0).to(device)

        # Run a forward pass
        view1 = {"img": img_tensor, "instance": [0, 1], "data_norm_type": "dust3r"}
        view2 = {"img": view1["img"][[1, 0]].clone().to(device), "instance": [1, 0], "data_norm_type": "dust3r"}

        res1, res2 = dust3r_model(view1, view2)
        print("Forward pass completed successfully!")

        # Automatically test the results against the reference result from vanilla dust3r code if they exist
        reference_output_path = MODEL_TO_VERIFICATION_PATH[model_name]["head_output"]
        if os.path.exists(reference_output_path):
            reference_output_data = np.load(reference_output_path)

            # Check against the reference output
            check_dict = {
                "head1_pts3d": (
                    res1["pts3d"].detach().cpu().numpy(),
                    reference_output_data["head1_pts3d"],
                ),
                "head2_pts3d": (
                    res2["pts3d_in_other_view"].detach().cpu().numpy(),
                    reference_output_data["head2_pts3d"],
                ),
                "head1_conf": (
                    res1["conf"].detach().squeeze(-1).cpu().numpy(),
                    reference_output_data["head1_conf"],
                ),
                "head2_conf": (
                    res2["conf"].detach().squeeze(-1).cpu().numpy(),
                    reference_output_data["head2_conf"],
                ),
            }

            compute_abs_and_rel_error = lambda x, y: (np.abs(x - y).max(), np.linalg.norm(x - y) / np.linalg.norm(x))

            print(f"===== Checking for {model_name} model =====")
            for key, (output, reference) in check_dict.items():
                abs_error, rel_error = compute_abs_and_rel_error(output, reference)
                print(f"{key} abs_error: {abs_error}, rel_error: {rel_error}")

                assert abs_error < 1e-2 and rel_error < 1e-3, f"Error in {key} output"

        points1 = res1["pts3d"][0].detach().cpu().numpy()
        points2 = res2["pts3d_in_other_view"][0].detach().cpu().numpy()
        conf_mask1 = res1["conf"][0].squeeze(-1).detach().cpu().numpy() > 3.0
        conf_mask2 = res2["conf"][0].squeeze(-1).detach().cpu().numpy() > 3.0

        if args.viz:
            rr.log(f"{model_name}", rr.ViewCoordinates.RDF, static=True)
            filtered_pts3d1 = points1[conf_mask1]
            filtered_pts3d1_colors = np.array(img0)[..., :3][conf_mask1] / 255
            filtered_pts3d2 = points2[conf_mask2]
            filtered_pts3d2_colors = np.array(img1)[..., :3][conf_mask2] / 255
            rr.log(
                f"{model_name}/view1",
                rr.Points3D(
                    positions=filtered_pts3d1.reshape(-1, 3),
                    colors=filtered_pts3d1_colors.reshape(-1, 3),
                ),
            )
            rr.log(
                f"{model_name}/view2",
                rr.Points3D(
                    positions=filtered_pts3d2.reshape(-1, 3),
                    colors=filtered_pts3d2_colors.reshape(-1, 3),
                ),
            )
            print(
                "Visualizations logged to Rerun: http://localhost:<rr-viewer-port>?url=ws://localhost:<ws-server-port>. "
                "Replace <rr-viewer-port> and <ws-server-port> with the actual ports."
            )
