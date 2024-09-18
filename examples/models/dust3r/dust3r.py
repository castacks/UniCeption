"""
Initalizing Pre-trained DUSt3R using UniCeption
"""

import argparse
import numpy as np
import os
import requests
import rerun as rr
from PIL import Image
from io import BytesIO

import torch
import torch.nn as nn
from typing import List, Tuple

from uniception.models.encoders import ViTEncoderInput
from uniception.models.encoders.croco import CroCoEncoder
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
from uniception.models.libs.croco.pos_embed import get_2d_sincos_pos_embed, RoPE2D

from uniception.models.info_sharing.cross_attention_transformer import (
    MultiViewCrossAttentionTransformer,
    MultiViewCrossAttentionTransformerIFR,
    MultiViewCrossAttentionTransformerInput,
)

from uniception.models.prediction_heads.adaptors import PointMapWithConfidenceAdaptor
from uniception.models.prediction_heads.base import AdaptorInput, PredictionHeadInput, PredictionHeadLayeredInput
from uniception.models.prediction_heads.dpt import DPTFeature, DPTRegressionProcessor
from uniception.models.prediction_heads.linear import LinearFeature

from uniception.utils.viz import script_add_rerun_args


def is_symmetrized(gt1, gt2):
    "Function to check if input pairs are symmetrized, i.e., (a, b) and (b, a) always exist in the input"
    x = gt1["instance"]
    y = gt2["instance"]
    if len(x) == len(y) and len(x) == 1:
        return False  # special case of batchsize 1
    ok = True
    for i in range(0, len(x), 2):
        ok = ok and (x[i] == y[i + 1]) and (x[i + 1] == y[i])
    return ok


def interleave(tensor1, tensor2):
    "Interleave two tensors along the first dimension (used to avoid redundant encoding for symmetrized pairs)"
    res1 = torch.stack((tensor1, tensor2), dim=1).flatten(0, 1)
    res2 = torch.stack((tensor2, tensor1), dim=1).flatten(0, 1)
    return res1, res2


class DUSt3R(nn.Module):
    "DUSt3R defined with UniCeption Modules"

    def __init__(
        self,
        name: str,
        data_norm_type: str = "dust3r",
        img_size: tuple = (224, 224),
        patch_embed_cls: str = "PatchEmbedDust3R",
        pred_head_type: str = "linear",
        pred_head_output_dim: int = 4,
        pred_head_feature_dim: int = 256,
        depth_mode: Tuple[str, float, float] = ("exp", -float("inf"), float("inf")),
        conf_mode: Tuple[str, float, float] = ("exp", 1, float("inf")),
        pos_embed: str = "RoPE100",
        pretrained_checkpoint_path: str = None,
        pretrained_encoder_checkpoint_path: str = None,
        pretrained_decoder_checkpoint_path: str = None,
        pretrained_pred_head_checkpoint_paths: List[str] = [None, None],
        pretrained_pred_head_regressor_checkpoint_paths: List[str] = [None, None],
        override_encoder_checkpoint_attributes: bool = False,
        *args,
        **kwargs,
    ):
        """
        Two-view model containing siamese encoders followed by a two-view cross-attention transformer and respective downstream heads.
        The goal is to output scene representation directly, both images in view1's frame (hence the asymmetry).

        Args:
            name (str): Name of the model.
            data_norm_type (str): Type of data normalization. (default: "dust3r")
            img_size (tuple): Size of input images. (default: (224, 224))
            patch_embed_cls (str): Class for patch embedding. (default: "PatchEmbedDust3R"). Options:
            - "PatchEmbedDust3R"
            - "ManyAR_PatchEmbed"
            pred_head_type (str): Type of prediction head. (default: "linear"). Options:
            - "linear"
            - "dpt"
            pred_head_output_dim (int): Output dimension of prediction head. (default: 4)
            pred_head_feature_dim (int): Feature dimension of prediction head. (default: 256)
            depth_mode (Tuple[str, float, float]): Depth mode settings (mode=['linear', 'square', 'exp'], vmin, vmax). (default: ('exp', -inf, inf))
            conf_mode (Tuple[str, float, float]): Confidence mode settings (mode=['linear', 'square', 'exp'], vmin, vmax). (default: ('exp', 1, inf))
            pos_embed (str): Position embedding type. (default: 'RoPE100')
            landscape_only (bool): Run downstream head only in landscape orientation. (default: True)
            pretrained_checkpoint_path (str): Path to pretrained checkpoint. (default: None)
            pretrained_encoder_checkpoint_path (str): Path to pretrained encoder checkpoint. (default: None)
            pretrained_decoder_checkpoint_path (str): Path to pretrained decoder checkpoint. (default: None)
            pretrained_pred_head_checkpoint_paths (List[str]): Paths to pretrained prediction head checkpoints. (default: None)
            pretrained_pred_head_regressor_checkpoint_paths (List[str]): Paths to pretrained prediction head regressor checkpoints. (default: None)
            override_encoder_checkpoint_attributes (bool): Whether to override encoder checkpoint attributes. (default: False)
        """
        super().__init__(*args, **kwargs)

        # Initalize the attributes
        self.name = name
        self.data_norm_type = data_norm_type
        self.img_size = img_size
        self.patch_embed_cls = patch_embed_cls
        self.pred_head_type = pred_head_type
        self.pred_head_output_dim = pred_head_output_dim
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pos_embed = pos_embed
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.pretrained_encoder_checkpoint_path = pretrained_encoder_checkpoint_path
        self.pretrained_decoder_checkpoint_path = pretrained_decoder_checkpoint_path
        self.pretrained_pred_head_checkpoint_paths = pretrained_pred_head_checkpoint_paths
        self.pretrained_pred_head_regressor_checkpoint_paths = pretrained_pred_head_regressor_checkpoint_paths
        self.override_encoder_checkpoint_attributes = override_encoder_checkpoint_attributes

        # Initialize RoPE for the CroCo Encoder & Two-View Cross Attention Transformer
        freq = float(pos_embed[len("RoPE") :])
        self.rope = RoPE2D(freq=freq)

        # Initialize Encoder
        self.encoder = CroCoEncoder(
            name=name,
            data_norm_type=data_norm_type,
            patch_embed_cls=patch_embed_cls,
            img_size=img_size,
            pretrained_checkpoint_path=pretrained_encoder_checkpoint_path,
            override_checkpoint_attributes=override_encoder_checkpoint_attributes,
        )

        # Initialize Multi-View Cross Attention Transformer
        if self.pred_head_type == "linear":
            # Returns only normalized last layer features
            self.decoder = MultiViewCrossAttentionTransformer(
                name="base_decoder",
                input_embed_dim=self.encoder.enc_embed_dim,
                num_views=2,
                custom_positional_encoding=self.rope,
                pretrained_checkpoint_path=pretrained_decoder_checkpoint_path,
            )
        elif self.pred_head_type == "dpt":
            # Returns intermediate features and normalized last layer features
            self.decoder = MultiViewCrossAttentionTransformerIFR(
                name="base_decoder",
                input_embed_dim=self.encoder.enc_embed_dim,
                num_views=2,
                indices=[5, 8],
                norm_intermediate=False,
                custom_positional_encoding=self.rope,
                pretrained_checkpoint_path=pretrained_decoder_checkpoint_path,
            )
        else:
            raise ValueError(f"Invalid prediction head type: {pred_head_type}. Must be 'linear' or 'dpt'.")

        # Initialize Prediction Heads
        if pred_head_type == "linear":
            # Initialize Prediction Head 1
            self.head1 = LinearFeature(
                input_feature_dim=self.decoder.dim,
                output_dim=pred_head_output_dim,
                patch_size=self.encoder.patch_size,
                pretrained_checkpoint_path=pretrained_pred_head_checkpoint_paths[0],
            )
            # Initialize Prediction Head 2
            self.head2 = LinearFeature(
                input_feature_dim=self.decoder.dim,
                output_dim=pred_head_output_dim,
                patch_size=self.encoder.patch_size,
                pretrained_checkpoint_path=pretrained_pred_head_checkpoint_paths[1],
            )
        elif pred_head_type == "dpt":
            # Initialze Predction Head 1
            self.dpt_feature_head1 = DPTFeature(
                patch_size=self.encoder.patch_size,
                hooks=[0, 1, 2, 3],
                input_feature_dims=[self.encoder.enc_embed_dim] + [self.decoder.dim] * 3,
                feature_dim=pred_head_feature_dim,
                pretrained_checkpoint_path=pretrained_pred_head_checkpoint_paths[0],
            )
            self.dpt_regressor_head1 = DPTRegressionProcessor(
                input_feature_dim=pred_head_feature_dim,
                output_dim=pred_head_output_dim,
                pretrained_checkpoint_path=pretrained_pred_head_regressor_checkpoint_paths[0],
            )
            self.head1 = nn.Sequential(self.dpt_feature_head1, self.dpt_regressor_head1)
            # Initialize Prediction Head 2
            self.dpt_feature_head2 = DPTFeature(
                patch_size=self.encoder.patch_size,
                hooks=[0, 1, 2, 3],
                input_feature_dims=[self.encoder.enc_embed_dim] + [self.decoder.dim] * 3,
                feature_dim=pred_head_feature_dim,
                pretrained_checkpoint_path=pretrained_pred_head_checkpoint_paths[1],
            )
            self.dpt_regressor_head2 = DPTRegressionProcessor(
                input_feature_dim=pred_head_feature_dim,
                output_dim=pred_head_output_dim,
                pretrained_checkpoint_path=pretrained_pred_head_regressor_checkpoint_paths[1],
            )
            self.head2 = nn.Sequential(self.dpt_feature_head2, self.dpt_regressor_head2)

        # Initialize Final Output Adaptor
        self.adaptor = PointMapWithConfidenceAdaptor(
            name="pointmap",
            pointmap_mode=depth_mode[0],
            pointmap_vmin=depth_mode[1],
            pointmap_vmax=depth_mode[2],
            confidence_type=conf_mode[0],
            confidence_vmin=conf_mode[1],
            confidence_vmax=conf_mode[2],
        )

        # Load pretrained weights
        if self.pretrained_checkpoint_path is not None:
            print(f"Loading pretrained DUSt3R weights from {self.pretrained_checkpoint_path} ...")
            ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"]))

    def _encode_image_pairs(self, img1, img2, data_norm_type):
        "Encode two different batches of images (each batch can have different image shape)"
        if img1.shape[-2:] == img2.shape[-2:]:
            encoder_input = ViTEncoderInput(image=torch.cat((img1, img2), dim=0), data_norm_type=data_norm_type)
            encoder_output = self.encoder(encoder_input)
            out, out2 = encoder_output.features.chunk(2, dim=0)
        else:
            encoder_input = ViTEncoderInput(image=img1, data_norm_type=data_norm_type)
            out = self.encoder(encoder_input)
            out = out.features
            encoder_input2 = ViTEncoderInput(image=img2)
            out2 = self.encoder(encoder_input2)
            out2 = out2.features

        return out, out2

    def _encode_symmetrized(self, view1, view2):
        "Encode image pairs accounting for symmetrization, i.e., (a, b) and (b, a) always exist in the input"
        img1 = view1["img"]
        img2 = view2["img"]
        if is_symmetrized(view1, view2):
            # Computing half of forward pass'
            feat1, feat2 = self._encode_image_pairs(img1[::2], img2[::2], data_norm_type=view1["data_norm_type"])
            feat1, feat2 = interleave(feat1, feat2)
        else:
            feat1, feat2 = self._encode_image_pairs(img1, img2, data_norm_type=view1["data_norm_type"])

        return feat1, feat2

    def _downstream_head(self, head_num, decout, img_shape):
        "Run the respective prediction heads"
        head = getattr(self, f"head{head_num}")
        if self.pred_head_type == "linear":
            head_input = PredictionHeadInput(last_feature=decout[f"{head_num}"])
        elif self.pred_head_type == "dpt":
            head_input = PredictionHeadLayeredInput(list_features=decout[f"{head_num}"], target_output_shape=img_shape)

        return head(head_input)

    def forward(self, view1, view2):
        """
        Forward pass for DUSt3R performing the following operations:
        1. Encodes the two input views (images).
        2. Combines the encoded features using a two-view cross-attention transformer.
        3. Passes the combined features through the respective prediction heads.
        4. Returns the processed final outputs for both views.

        Args:
            view1 (dict): Dictionary containing the first view's images and instance information.
                          "img" is a required key and value is a tensor of shape (B, C, H, W).
            view2 (dict): Dictionary containing the second view's images and instance information.
                          "img" is a required key and value is a tensor of shape (B, C, H, W).

        Returns:
            Tuple[dict, dict]: A tuple containing the final outputs for both views.
        """
        # Get input shapes
        _, _, height1, width1 = view1["img"].shape
        _, _, height2, width2 = view2["img"].shape
        shape1 = (int(height1), int(width1))
        shape2 = (int(height2), int(width2))

        # Encode the two images --> Each feat output: BCHW features (batch_size, feature_dim, feature_height, feature_width)
        feat1, feat2 = self._encode_symmetrized(view1, view2)

        # Combine all images into view-centric representation
        decoder_input = MultiViewCrossAttentionTransformerInput(features=[feat1, feat2])
        if self.pred_head_type == "linear":
            final_decoder_multi_view_feat = self.decoder(decoder_input)
        elif self.pred_head_type == "dpt":
            final_decoder_multi_view_feat, intermediate_decoder_multi_view_feat = self.decoder(decoder_input)

        if self.pred_head_type == "linear":
            # Define feature dictionary for linear head
            decoder_outputs = {
                "1": final_decoder_multi_view_feat.features[0].float(),
                "2": final_decoder_multi_view_feat.features[1].float(),
            }
        elif self.pred_head_type == "dpt":
            # Define feature dictionary for DPT head
            decoder_outputs = {
                "1": [
                    feat1.float(),
                    intermediate_decoder_multi_view_feat[0].features[0].float(),
                    intermediate_decoder_multi_view_feat[1].features[0].float(),
                    final_decoder_multi_view_feat.features[0].float(),
                ],
                "2": [
                    feat2.float(),
                    intermediate_decoder_multi_view_feat[0].features[1].float(),
                    intermediate_decoder_multi_view_feat[1].features[1].float(),
                    final_decoder_multi_view_feat.features[1].float(),
                ],
            }

        # Downstream task prediction
        with torch.autocast("cuda", enabled=False):
            # Prediction heads
            head_output1 = self._downstream_head(1, decoder_outputs, shape1)
            head_output2 = self._downstream_head(2, decoder_outputs, shape2)

            # Post-process outputs
            final_output1 = self.adaptor(
                AdaptorInput(adaptor_feature=head_output1.decoded_channels, output_shape_hw=shape1)
            )
            final_output2 = self.adaptor(
                AdaptorInput(adaptor_feature=head_output2.decoded_channels, output_shape_hw=shape2)
            )

            # Convert outputs to dictionary
            res1 = {
                "pts3d": final_output1.value.permute(0, 2, 3, 1).contiguous(),
                "conf": final_output1.confidence.permute(0, 2, 3, 1).contiguous(),
            }
            res2 = {
                "pts3d_in_other_view": final_output2.value.permute(0, 2, 3, 1).contiguous(),
                "conf": final_output2.confidence.permute(0, 2, 3, 1).contiguous(),
            }

        return res1, res2


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

    # Initialize model configurations
    MODEL_TO_CHECKPOINT_PATH = {
        "dust3r_512_dpt": {
            "encoder": f"{relative_checkpoint_path}/encoders/CroCo_Encoder_512_DUSt3R_dpt.pth",
            "decoder": f"{relative_checkpoint_path}/info_sharing/cross_attn_transformer/Two_View_Cross_Attention_Transformer_DUSt3R_512_dpt.pth",
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
            "decoder": f"{relative_checkpoint_path}/info_sharing/cross_attn_transformer/Two_View_Cross_Attention_Transformer_MASt3R_512_dpt.pth",
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
            "decoder": f"{relative_checkpoint_path}/info_sharing/cross_attn_transformer/Two_View_Cross_Attention_Transformer_DUSt3R_512_linear.pth",
            "feature_head": [
                f"{relative_checkpoint_path}/prediction_heads/linear_feature_head/DUSt3R_512_linear_feature_head1.pth",
                f"{relative_checkpoint_path}/prediction_heads/linear_feature_head/DUSt3R_512_linear_feature_head2.pth",
            ],
            "regressor": None,
            "ckpt_path": f"{relative_checkpoint_path}/examples/original_dust3r/DUSt3R_ViTLarge_BaseDecoder_512_linear.pth",
        },
        "dust3r_224_linear": {
            "encoder": f"{relative_checkpoint_path}/encoders/CroCo_Encoder_224_DUSt3R_linear.pth",
            "decoder": f"{relative_checkpoint_path}/info_sharing/cross_attn_transformer/Two_View_Cross_Attention_Transformer_DUSt3R_224_linear.pth",
            "feature_head": [
                f"{relative_checkpoint_path}/prediction_heads/linear_feature_head/DUSt3R_224_linear_feature_head1.pth",
                f"{relative_checkpoint_path}/prediction_heads/linear_feature_head/DUSt3R_224_linear_feature_head2.pth",
            ],
            "regressor": None,
            "ckpt_path": f"{relative_checkpoint_path}/examples/original_dust3r/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth",
        },
    }

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

    model_configurations = ["dust3r_512_dpt", "dust3r_512_linear", "dust3r_224_linear", "dust3r_512_dpt_mast3r"]

    # Test different DUSt3R models using UniCeption modules
    for model_name in model_configurations:
        dust3r_model = DUSt3R(
            name=model_name,
            img_size=(512, 512) if "512" in model_name else (224, 224),
            patch_embed_cls="PatchEmbedDust3R",
            pred_head_type="linear" if "linear" in model_name else "dpt",
            pretrained_checkpoint_path=MODEL_TO_CHECKPOINT_PATH[model_name]["ckpt_path"],
            # pretrained_encoder_checkpoint_path=MODEL_TO_CHECKPOINT_PATH[model_name]["encoder"],
            # pretrained_decoder_checkpoint_path=MODEL_TO_CHECKPOINT_PATH[model_name]["decoder"],
            # pretrained_pred_head_checkpoint_paths=MODEL_TO_CHECKPOINT_PATH[model_name]["feature_head"],
            # pretrained_pred_head_regressor_checkpoint_paths=MODEL_TO_CHECKPOINT_PATH[model_name]["regressor"],
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
